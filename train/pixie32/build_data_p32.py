"""Build a fresh native-32 dataset from reviewed combined Krea sprites."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

import numpy as np
from PIL import Image
from tokenizers import Tokenizer

try:
    from .data import DEFAULT_BLIND50, DEFAULT_EVAL50, Holdouts, load_dataset, normalize, sha256
except ImportError:
    from data import DEFAULT_BLIND50, DEFAULT_EVAL50, Holdouts, load_dataset, normalize, sha256

BOS, EOS, NO_PALETTE = 49406, 49407, 4097


def relative_path(value):
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or ":" in text:
        raise ValueError("Dataset paths must be relative and contained")
    return path.as_posix()


def checked_path(root, value):
    path = root / relative_path(value)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Dataset path escapes its root")
    return path


def encode(tokenizer, text):
    content = tokenizer.encode(text, add_special_tokens=False).ids[:16]
    if any(i < 0 or i >= BOS for i in content):
        raise ValueError("Caption contains invalid CLIP content ids")
    row = np.full(18, EOS, dtype=np.uint16)
    row[0] = BOS
    row[1:1 + len(content)] = content
    return row, tokenizer.decode(content, skip_special_tokens=False)


class Quantizer:
    def __init__(self, palette):
        self.palette = palette[1:].astype(np.int32)
        self.lookup = {}

    def __call__(self, rgba):
        if len(self.lookup) > 262144:
            self.lookup.clear()
        rgb = rgba[..., :3].astype(np.uint32)
        packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        opaque = rgba[..., 3] >= 128
        unique = np.unique(packed[opaque])
        missing = [int(x) for x in unique if int(x) not in self.lookup]
        for start in range(0, len(missing), 256):
            keys = np.array(missing[start:start + 256], dtype=np.uint32)
            colors = np.stack([(keys >> 16) & 255, (keys >> 8) & 255, keys & 255], 1).astype(np.int32)
            distance = ((colors[:, None] - self.palette[None]) ** 2).sum(2)
            self.lookup.update(zip(keys.tolist(), (distance.argmin(1) + 1).tolist()))
        out = np.zeros((32, 32), dtype=np.uint16)
        if unique.size:
            vals = np.array([self.lookup[int(x)] for x in unique], dtype=np.uint16)
            out[opaque] = vals[np.searchsorted(unique, packed[opaque])]
        return out.reshape(1024)


def conditioning(tokens, palette, color_thresholds):
    cond = np.zeros((len(tokens), 6), np.int64)
    for i, row in enumerate(tokens):
        counts = np.bincount(row, minlength=4097)
        order = np.lexsort((np.arange(1, 4097), -counts[1:]))[:3] + 1
        cond[i, :3] = np.where(counts[order] > 0, order, NO_PALETTE)
        opaque = row.reshape(32, 32) != 0
        fill = int(opaque.sum())
        cond[i, 3] = 1 if fill < 320 else (2 if fill <= 720 else 3)
        pad = np.pad(opaque, 1)
        boundary = opaque & (~pad[:-2, 1:-1] | ~pad[2:, 1:-1] | ~pad[1:-1, :-2] | ~pad[1:-1, 2:])
        dark = palette[row].astype(np.int32).sum(1).reshape(32, 32) < 90
        cond[i, 4] = 1 if boundary.any() and 2 * np.sum(dark & boundary) >= boundary.sum() else 2
        ncolors = np.count_nonzero(counts[1:])
        cond[i, 5] = 1 if ncolors <= color_thresholds[0] else (2 if ncolors <= color_thresholds[1] else 3)
    return cond


def build(args):
    root, out = Path(args.data_root), Path(args.out)
    if out.exists():
        raise ValueError("Dataset output already exists; use a new folder")
    policy = Holdouts(args.blind50, args.eval50)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    palette = np.load(args.palette, allow_pickle=False)
    if palette.shape != (4097, 3) or palette.dtype != np.uint8:
        raise ValueError("Palette must be uint8 [4097,3]")
    metadata_path = root / "final" / "metadata.jsonl"
    provenance_path = root / "manifest.jsonl"
    if not metadata_path.is_file() or not provenance_path.is_file():
        raise ValueError("Combined data requires final/metadata.jsonl and manifest.jsonl; sample provenance is not synthesized")
    r1 = {relative_path(x.strip()) for x in Path(args.r1).read_text(encoding="utf-8-sig").splitlines() if x.strip()}
    if not r1:
        raise ValueError("R1 keep-list is empty")
    edge = {relative_path(x.strip()) for x in Path(args.edge_list).read_text(encoding="utf-8-sig").splitlines() if x.strip()} if args.edge_list else set()
    sample_weights = {}
    if getattr(args, "sample_weights", None):
        from train.pixie32.quality_weights import load_sample_weights
        sample_weights = load_sample_weights(args.sample_weights, root, args.r1, policy.hashes)
    found_weights = set()
    provenance = {}
    with provenance_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] in provenance:
                raise ValueError("Duplicate provenance id")
            provenance[record["id"]] = record
    records, token_rows, texts, subjects = [], [], [], []
    drops = Counter()
    quantize = Quantizer(palette)
    seen_tokens, seen_ids, found_r1 = set(), set(), set()
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = [json.loads(line) for line in stream if line.strip()]
    for meta in sorted(metadata, key=lambda r: r["id"]):
        rid = meta["id"]
        if rid in seen_ids:
            raise ValueError("Duplicate metadata id")
        seen_ids.add(rid)
        rel = relative_path("final/" + relative_path(meta["file_name"]))
        if rel not in r1:
            drops["outside_r1"] += 1
            continue
        found_r1.add(rel)
        if rid in sample_weights:
            if sample_weights[rid]["file"] != rel:
                raise ValueError("Sample weight file differs from metadata")
            found_weights.add(rid)
        prov = provenance.get(rid)
        if prov is None:
            raise ValueError("Missing provenance for R1 row")
        if normalize(prov.get("source", "")) not in ("3060ti", "3080ti", "5070ti"):
            drops["unapproved_source"] += 1
            continue
        if meta.get("source") != prov["source"] or PurePosixPath(rel).stem != rid or relative_path(prov.get("final_path", "")) != rel:
            raise ValueError("Image id, source, or final path differs from provenance")
        if not prov.get("pass_extract_v2") or not prov.get("final") or prov.get("holdout") or prov.get("dup_of"):
            drops["provenance_reject"] += 1
            continue
        subject = normalize(prov.get("subject", ""))
        if not subject:
            raise ValueError("Missing provenance subject")
        image_path = checked_path(root, rel)
        caption = image_path.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
        if not caption or normalize(caption) != normalize(meta["text"]) or normalize(caption) != normalize(prov["caption"]):
            raise ValueError("Caption metadata and sidecar do not match")
        text, trimmed = encode(tokenizer, caption)
        subj_text, trimmed_subj = encode(tokenizer, subject)
        if policy.matches(caption, subject, trimmed, trimmed_subj):
            drops["holdout"] += 1
            continue
        with Image.open(image_path) as image:
            if image.size != (32, 32):
                raise ValueError("Expected native 32x32 image")
            rgba = np.asarray(image.convert("RGBA"))
        tokens = quantize(rgba)
        digest = hashlib.sha256(tokens.tobytes()).hexdigest()
        if digest in seen_tokens:
            drops["quantized_duplicate"] += 1
            continue
        seen_tokens.add(digest)
        sheet_id = prov.get("sheet_id")
        if not isinstance(sheet_id, int) or isinstance(sheet_id, bool) or sheet_id < 0:
            raise ValueError("Missing or invalid numeric provenance sheet identity")
        records.append(dict(id=rid, source=prov["source"], file=rel, caption=caption, subject=subject, trimmed_caption=trimmed,
                            trimmed_subject=trimmed_subj, sheet=f"{prov['source']}:{sheet_id}",
                            edge_heavy=rel in edge, pixel_sha256=digest))
        if sample_weights:
            records[-1]["quality_multiplier"] = sample_weights.get(rid, {}).get("multiplier", 1.)
        token_rows.append(tokens)
        texts.append(text)
        subjects.append(subj_text)
    if found_r1 != r1:
        raise ValueError("R1 list contains rows missing from metadata")
    if found_weights != set(sample_weights):
        raise ValueError("Sample weights contain rows outside the R1 population")
    if len(records) < 3:
        raise ValueError("Not enough admitted rows for train and evaluation splits")
    rng = np.random.default_rng(args.seed)
    unique_subjects = sorted({r["subject"] for r in records})
    probe_count = max(1, round(len(unique_subjects) * args.probe_fraction))
    if probe_count >= len(unique_subjects):
        raise ValueError("Not enough subjects for unseen probe")
    probe_subjects = set(rng.choice(unique_subjects, probe_count, replace=False).tolist())
    probe_sheets = {r["sheet"] for r in records if r["subject"] in probe_subjects}
    remaining_sheets = sorted({r["sheet"] for r in records} - probe_sheets)
    if len(remaining_sheets) < 2:
        raise ValueError("Not enough independent sheets for train and IID splits")
    iid_count = min(len(remaining_sheets) - 1, max(1, round(len(remaining_sheets) * args.iid_fraction)))
    iid_sheets = set(rng.choice(remaining_sheets, iid_count, replace=False).tolist())
    split, keep = [], []
    for i, record in enumerate(records):
        if record["subject"] in probe_subjects:
            value = 2
        else:
            value = 1 if record["sheet"] in iid_sheets else 0
        split.append(value)
        keep.append(i)
    eval_captions = {normalize(records[i]["caption"]) for i, s in zip(keep, split) if s != 0}
    filtered = [(i, s) for i, s in zip(keep, split) if s != 0 or normalize(records[i]["caption"]) not in eval_captions]
    drops["train_eval_caption"] = len(keep) - len(filtered)
    keep = [i for i, _ in filtered]
    split = np.array([s for _, s in filtered], dtype=np.uint8)
    records = [records[i] for i in keep]
    if not all(np.any(split == s) for s in (0, 1, 2)):
        raise ValueError("Filtering emptied a required split")
    tokens = np.stack([token_rows[i] for i in keep])
    ncolors = np.array([len(np.unique(row[row > 0])) for row in tokens])
    thresholds = np.maximum(1, np.quantile(ncolors[split == 0], [1 / 3, 2 / 3], method="nearest").astype(int))
    thresholds[1] = max(int(thresholds[1]), int(thresholds[0]) + 1)
    sub_ids = {subject: i for i, subject in enumerate(sorted({r["subject"] for r in records}))}
    train_freq = Counter(r["subject"] for r, s in zip(records, split) if s == 0)
    raw_balance = np.array([train_freq[r["subject"]] ** -0.35 for r, s in zip(records, split) if s == 0])
    balance_mean = raw_balance.mean()
    weights = np.array([min(6.0, (max(1, train_freq[r["subject"]]) ** -0.35) / balance_mean) * (0.5 if r["edge_heavy"] else 1.0) for r in records], dtype=np.float32)
    weights *= np.array([r.get("quality_multiplier", 1.) for r in records], dtype=np.float32)
    weights /= weights[split == 0].mean()
    arrays = dict(tokens=tokens, text=np.stack([texts[i] for i in keep]), subj_text=np.stack([subjects[i] for i in keep]),
                  subject_id=np.array([sub_ids[r["subject"]] for r in records], dtype=np.int32), weight=weights,
                  noflip=np.array([bool(re.search(r"\b(left|right|east|west|facing|letter|letters|text|word|words|writing|written|sign|logo|number|numbers|digit|digits)\b", normalize(r["caption"] + " " + r["subject"]))) for r in records]),
                  split=split, cond=conditioning(tokens, palette, thresholds))
    out.mkdir(parents=True, exist_ok=False)
    (out / "ids.txt").write_text("".join(r["id"] + "\n" for r in records), encoding="utf-8")
    (out / "records.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    np.save(out / "palette.npy", palette, allow_pickle=False)
    for key, value in arrays.items():
        np.save(out / f"{key}.npy", value, allow_pickle=False)
    names = ["ids.txt", "records.json", "palette.npy", *[f"{key}.npy" for key in arrays]]
    manifest = dict(schema="pixie32-v1", rows=len(records), files={name: sha256(out / name) for name in names},
                    arrays=[f"{key}.npy" for key in arrays],
                    array_ids_sha256={f"{key}.npy": sha256(out / "ids.txt") for key in arrays},
                    inputs={"palette": sha256(args.palette), "tokenizer": sha256(args.tokenizer), "r1": sha256(args.r1),
                            "edge_list": sha256(args.edge_list) if args.edge_list else None,
                            "metadata": sha256(metadata_path), "provenance": sha256(provenance_path)},
                    holdouts=policy.hashes, drops=dict(drops), split_counts={str(s): int(np.sum(split == s)) for s in (0, 1, 2)},
                    style={"fill_thresholds": [320, 720], "color_thresholds": thresholds.tolist(), "color_calibration": "training-set thirds, nearest quantile", "dark_rgb_sum": 90},
                    seed=args.seed, sampling={"balance_alpha": 0.35, "balance_cap": 6, "edge_multiplier": 0.5})
    if sample_weights:
        manifest["inputs"]["sample_weights"] = sha256(args.sample_weights)
        manifest["sampling"]["quality_multipliers"] = "hash-bound per-row weights multiplied after balance and edge weights"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    load_dataset(out, args.blind50, args.eval50)
    return manifest


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "r1", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--palette", type=Path, default=os.environ.get("PALETTE"))
    parser.add_argument("--tokenizer", type=Path, default=os.environ.get("TOKENIZER"))
    parser.add_argument("--blind50", type=Path, default=DEFAULT_BLIND50)
    parser.add_argument("--eval50", type=Path, default=DEFAULT_EVAL50)
    parser.add_argument("--edge-list", type=Path)
    parser.add_argument("--sample-weights", type=Path, help="optional hash-bound quality multiplier JSON; requires a new dataset cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-fraction", type=float, default=0.05)
    parser.add_argument("--iid-fraction", type=float, default=0.05)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    for name in ("palette", "tokenizer"):
        if getattr(args, name) is None:
            parser.error(f"--{name} or ${name.upper()} is required")
    if not 0 < args.probe_fraction < 1 or not 0 < args.iid_fraction < 1:
        parser.error("Split fractions must be between zero and one")
    try:
        manifest = build(args)
    except OSError:
        parser.exit(2, "Dataset build refused: unable to read or create a required file\n")
    except (ValueError, KeyError) as error:
        parser.exit(2, f"Dataset build refused: {type(error).__name__}: {error}\n")
    print(json.dumps({key: manifest[key] for key in ("rows", "drops", "split_counts", "style")}))


if __name__ == "__main__":
    main()
