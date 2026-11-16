"""Read-only R1 coverage audit and uniformly sampled human quality review."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import textwrap
import time

import numpy as np
from PIL import Image, ImageDraw

from train.pixie32.data import Holdouts, load_dataset, normalize, sha256


def contained(root, name):
    name = str(name).replace("\\", "/")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or ":" in name:
        raise ValueError("Dataset paths must be relative and contained")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes its root")
    return path


def read_rows(path):
    rows = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row["id"] in rows:
                    raise ValueError("Duplicate row id")
                rows[row["id"]] = row
    return rows


def distribution(values):
    counts = Counter(values)
    frequencies = list(counts.values())
    return {"unique": len(counts), "singletons": sum(n == 1 for n in frequencies),
            "max_rows_per_value": max(frequencies, default=0),
            "top_20": counts.most_common(20)}


def effective_rows(weights):
    weights = np.asarray(weights, dtype=np.float64)
    return float(weights.sum() ** 2 / np.square(weights).sum())


def load_population(args):
    root = args.data_root
    policy = Holdouts(args.blind50, args.eval50)
    keep = args.r1.read_text(encoding="utf-8-sig").splitlines()
    keep = [str(PurePosixPath(p.strip().replace("\\", "/"))) for p in keep if p.strip()]
    if not keep or len(set(keep)) != len(keep):
        raise ValueError("R1 list must be nonempty with unique paths")
    for path in keep:
        contained(root, path)
    keep = set(keep)
    metadata = read_rows(root / "final/metadata.jsonl")
    provenance = read_rows(root / "manifest.jsonl")
    population, found, excluded = [], set(), 0
    for rid, meta in sorted(metadata.items()):
        rel = "final/" + str(PurePosixPath(meta["file_name"].replace("\\", "/")))
        if rel not in keep:
            continue
        found.add(rel)
        prov = provenance.get(rid)
        if not prov or prov.get("final_path") != rel or meta.get("source") != prov.get("source") or PurePosixPath(rel).stem != rid:
            raise ValueError("R1 image identity differs from provenance")
        if prov.get("source") not in ("3060ti", "3080ti", "5070ti"):
            raise ValueError("R1 has unrecognized source")
        image_path = contained(root, rel)
        caption = image_path.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
        subject = normalize(prov.get("subject", ""))
        if not caption or not subject or normalize(caption) != normalize(meta["text"]) or normalize(caption) != normalize(prov["caption"]):
            raise ValueError("R1 caption/subject or sidecar differs from provenance")
        if policy.matches(caption, subject) or prov.get("holdout"):
            excluded += 1
            continue
        if not prov.get("final") or not prov.get("pass_extract_v2") or prov.get("dup_of"):
            raise ValueError("R1 contains rejected provenance")
        population.append(dict(id=rid, file=rel, caption=caption, subject=subject, source=prov["source"],
                               sheet=f"{prov['source']}:{prov['sheet_id']}"))
    if found != keep:
        raise ValueError("R1 paths are missing from metadata")
    if not population:
        raise ValueError("No non-holdout R1 rows")
    return population, policy, excluded


def image_stats(path):
    with Image.open(path) as image:
        rgba = np.array(image.convert("RGBA"))
    if rgba.shape != (32, 32, 4):
        raise ValueError("R1 requires native 32x32 images")
    alpha = rgba[..., 3]
    opaque = alpha > 0
    border = np.zeros((32, 32), dtype=bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    canonical = rgba.copy()
    canonical[~opaque, :3] = 0
    return dict(pixel_sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),
                empty=not bool(opaque.any()), binary_alpha=bool(np.isin(alpha, [0, 255]).all()),
                edge_px=int((opaque & border).sum()), fg_px=int(opaque.sum()))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(args):
    if args.out.exists():
        raise ValueError("Audit output exists; use a new folder")
    if not 1 <= args.sample <= 2000:
        raise ValueError("Sample must be between 1 and 2000")
    started = time.perf_counter()
    rows, policy, excluded = load_population(args)
    if getattr(args, "expect_rows", None) is not None and len(rows) != args.expect_rows:
        raise ValueError("Non-holdout R1 population differs from expected row count")
    if args.sample > len(rows):
        raise ValueError("Sample exceeds non-holdout R1 population")
    training_cache = None
    if getattr(args, "training_cache", None):
        cache = load_dataset(args.training_cache, args.blind50, args.eval50)
        if not set(cache.ids) <= {row["id"] for row in rows}:
            raise ValueError("Training cache contains rows outside audited R1 population")
        train = cache.arrays["split"] == 0
        actual_weights = cache.arrays["weight"][train].astype(np.float64)
        actual_edge = np.array([row["edge_heavy"] for row in cache.records])[train]
        training_cache = dict(fingerprint=cache.fingerprint, train_rows=int(train.sum()),
                              edge_heavy_rows=int(actual_edge.sum()), effective_rows=effective_rows(actual_weights),
                              edge_sampling_probability=float(actual_weights[actual_edge].sum() / actual_weights.sum()))
    pixels = Counter()
    for row in rows:
        row.update(image_stats(contained(args.data_root, row["file"])))
        pixels[row["pixel_sha256"]] += 1
    freq = Counter(row["subject"] for row in rows)
    balance = np.array([freq[row["subject"]] ** -0.35 for row in rows])
    balance = np.minimum(6., balance / balance.mean())
    edge = np.array([row["edge_px"] >= 32 for row in rows])
    weights = balance * np.where(edge, .5, 1.)
    chosen = random.Random(args.seed).sample(rows, args.sample)
    hashes = {"r1": sha256(args.r1), "metadata": sha256(args.data_root / "final/metadata.jsonl"),
              "manifest": sha256(args.data_root / "manifest.jsonl"), "holdouts": policy.hashes}
    report = dict(schema="pixie-r1-audit-v1", population_rows=len(rows), holdouts_excluded=excluded,
                  inputs=hashes, measurement="actual files and full non-holdout R1 population; no human junk labels yet",
                  source_counts=dict(Counter(row["source"] for row in rows)),
                  edge_heavy_by_source=dict(Counter(row["source"] for row in rows if row["edge_px"] >= 32)),
                  subjects=distribution(row["subject"] for row in rows),
                  captions=distribution(normalize(row["caption"]) for row in rows),
                  sheets=distribution(row["sheet"] for row in rows),
                  visible_pixel_duplicate_excess=sum(n - 1 for n in pixels.values()),
                  duplicate_groups=sum(n > 1 for n in pixels.values()),
                  empty_images=sum(row["empty"] for row in rows),
                  nonbinary_alpha_images=sum(not row["binary_alpha"] for row in rows),
                  edge_heavy_rows=int(edge.sum()),
                  weighting_scenario={"scope": "full R1 before train/probe/IID splitting; NOT actual trainer sampling weights",
                                      "balance_alpha": .35, "balance_cap": 6., "edge_multiplier": .5,
                                      "edge_probability_without_discount": float(balance[edge].sum() / balance.sum()),
                                      "edge_probability_with_discount": float(weights[edge].sum() / weights.sum()),
                                      "effective_rows": effective_rows(weights)},
                  annotation=dict(design="uniform without replacement from sorted IDs", seed=args.seed,
                                  sample_rows=len(chosen), population_rows=len(rows)),
                  training_cache=training_cache,
                  scan_seconds=time.perf_counter() - started)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "images").mkdir()
    write_json(args.out / "audit.json", report)
    write_json(args.out / "sample.json", [{k: row[k] for k in ("id", "caption", "subject", "source", "edge_px")} for row in chosen])
    with (args.out / "coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "rows", "unique_captions", "3060ti", "3080ti", "5070ti"])
        coverage = {}
        for row in rows:
            entry = coverage.setdefault(row["subject"], {"captions": set(), "sources": Counter()})
            entry["captions"].add(normalize(row["caption"]))
            entry["sources"][row["source"]] += 1
        for subject, entry in sorted(coverage.items()):
            writer.writerow([subject, freq[subject], len(entry["captions"]),
                             *(entry["sources"][source] for source in ("3060ti", "3080ti", "5070ti"))])
    with (args.out / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "junk", "caption_match", "defect"])
        writer.writeheader()
        writer.writerows(dict(id=row["id"]) for row in chosen)
    for start in range(0, len(chosen), 40):
        page = Image.new("RGB", (8 * 160, 5 * 188), (40, 40, 40))
        draw = ImageDraw.Draw(page)
        for offset, row in enumerate(chosen[start:start + 40]):
            i = start + offset
            x, y = offset % 8 * 160, offset // 8 * 188
            with Image.open(contained(args.data_root, row["file"])) as image:
                sprite = image.convert("RGBA").resize((128, 128), Image.Resampling.NEAREST)
            sprite.save(args.out / "images" / f"{i:04d}.png")
            background = Image.new("RGBA", (128, 128), (180, 180, 180, 255))
            background.alpha_composite(sprite)
            page.paste(background.convert("RGB"), (x, y))
            draw.text((x, y + 130), f"{i:04d}", fill="white")
            # Full captions and IDs are in sample.json; the sheet is only an index.
            caption = row["caption"].encode("ascii", "replace").decode()
            draw.text((x, y + 144), "\n".join(textwrap.wrap(caption, 24)[:3]), fill="white")
        page.save(args.out / f"contact-{start // 40:03d}.png")
    return report


def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    scale = 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return [0. if successes == 0 else max(0., center - half),
            1. if successes == n else min(1., center + half)]


def summarize(args):
    packet = json.loads((args.packet / "audit.json").read_text(encoding="utf-8"))
    sample = json.loads((args.packet / "sample.json").read_text(encoding="utf-8"))
    expected = {row["id"] for row in sample}
    with args.labels.open(encoding="utf-8-sig", newline="") as stream:
        labels = list(csv.DictReader(stream))
    if len(labels) != len(expected) or {row["id"] for row in labels} != expected:
        raise ValueError("Label IDs must match the entire packet exactly once")
    for row in labels:
        if row.get("junk", "") not in ("", "yes", "no", "borderline") or row.get("caption_match", "") not in ("", "yes", "no", "uncertain"):
            raise ValueError("Invalid label; use yes/no/borderline for junk and yes/no/uncertain for caption_match")
    result = dict(schema="pixie-r1-quality-v1", population_rows=packet["population_rows"], sample_rows=len(sample),
                  design=packet["annotation"], packet_sha256=sha256(args.packet / "sample.json"),
                  labels_sha256=sha256(args.labels), metrics={})
    for name, positive in (("junk", {"yes"}), ("junk_or_borderline", {"yes", "borderline"}),
                           ("caption_mismatch", {"no"}), ("caption_uncertain", {"uncertain"})):
        field = "caption_match" if name.startswith("caption_") else "junk"
        labeled = [row for row in labels if row[field]]
        count = sum(row[field] in positive for row in labeled)
        result["metrics"][name] = dict(positive=count, labeled=len(labeled), missing=len(labels) - len(labeled),
                                         fraction_among_labeled=count / len(labeled) if labeled else None,
                                         wilson_95_among_labeled=wilson(count, len(labeled)),
                                         population_inference_valid=len(labeled) == len(labels))
    result["interpretation"] = "Only a fully labeled uniform packet supports population estimates. Intervals are conservative binomial Wilson intervals; subjective labels need a consistent rubric. Uncertain caption labels are not mismatches."
    by_id = {row["id"]: row for row in sample}
    result["subgroups"] = {}
    for group, select in (("edge_heavy", lambda row: row["edge_px"] >= 32),
                          ("not_edge_heavy", lambda row: row["edge_px"] < 32),
                          *((source, lambda row, s=source: row["source"] == s)
                            for source in ("3060ti", "3080ti", "5070ti"))):
        selected = [row for row in labels if select(by_id[row["id"]])]
        labeled = [row for row in selected if row["junk"]]
        count = sum(row["junk"] == "yes" for row in labeled)
        result["subgroups"][group] = dict(sample_rows=len(selected), labeled=len(labeled), junk=count,
                                            wilson_95_among_labeled=wilson(count, len(labeled)),
                                            population_inference_valid=bool(selected) and len(labeled) == len(selected))
    if args.out.exists():
        raise ValueError("Summary output exists; use a new file")
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", allow_abbrev=False)
    for name in ("data-root", "r1", "blind50", "eval50", "out"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--sample", type=int, default=400)
    prep.add_argument("--seed", type=int, default=20261115)
    prep.add_argument("--expect-rows", type=int)
    prep.add_argument("--training-cache", type=Path, help="Optional verified P32 cache: enforce original holdout bytes and report actual train weights")
    summary = commands.add_parser("summarize", allow_abbrev=False)
    for name in ("packet", "labels", "out"):
        summary.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args) if args.command == "prepare" else summarize(args)
    # Print only numeric status; outputs deliberately do not include input paths.
    print(json.dumps({"population_rows": result["population_rows"],
                      "sample_rows": result.get("sample_rows", result.get("annotation", {}).get("sample_rows"))}))


if __name__ == "__main__":
    main()
