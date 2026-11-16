"""Build reviewed, lineage-disjoint Krea animation LoRA training sheets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import (DEFAULT_BLIND50, DEFAULT_EVAL50, holdout_policy, load_collection,
                     new_output, relative_file, reviewed_sequences, sha256)
from train.pixie32.data import normalize_holdout

SCHEMA = "pixie-animation-sheets-v1"


def _kind(row):
    value = row["provenance"]
    return value.get("kind") if isinstance(value, dict) else value


def visible_sequence_hash(collection, row):
    digest = hashlib.sha256()
    visible_frames = []
    for frame in row["frames"]:
        with Image.open(relative_file(collection.root, frame["path"])) as image:
            rgba = np.array(image.convert("RGBA"), copy=True)
        rgba[rgba[..., 3] == 0, :3] = 0
        if np.any(rgba[..., 3]):
            visible_frames.append(hashlib.sha256(json.dumps(list(rgba.shape)).encode() + rgba.tobytes()).hexdigest())
        digest.update(json.dumps([*rgba.shape, frame["duration_ms"]]).encode())
        digest.update(rgba.tobytes())
    return digest.hexdigest(), visible_frames


def _groups(candidates):
    parents = list(range(len(candidates)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, entry in enumerate(candidates):
        row = entry["row"]
        keys = [("lineage", row["lineage_group"]), ("subject", normalize_holdout(row["subject"])),
                ("visible_sequence", entry["visible_hash"])]
        for frame_hash in entry["visible_frame_hashes"]:
            keys.append(("frame", frame_hash))
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    components = {}
    for index, entry in enumerate(candidates):
        components.setdefault(find(index), []).append(entry)
    groups = []
    for entries in components.values():
        names = sorted({entry["row"]["lineage_group"] for entry in entries})
        group_id = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()
        groups.append((group_id, entries))
    return groups


def build(inputs, output, *, grid=2, cell_px=256, reviews=(), eval_fraction=0.1,
          seed=0, include_procedural=False, include_generated=False,
          eval_subjects=None, blind50=DEFAULT_BLIND50, eval50=DEFAULT_EVAL50, logical_px=32,
          eval_min_lineages=None, max_procedural_fraction=0.5):
    if type(logical_px) is not int or logical_px not in (32, 64):
        raise ValueError("logical_px must be 32 or 64")
    if grid not in (2, 3, 4) or type(cell_px) is not int or cell_px < logical_px or cell_px % logical_px:
        raise ValueError(f"Grid must be 2, 3 or 4 and cell_px a positive multiple of {logical_px}")
    if not isinstance(eval_fraction, (int, float)) or not 0 <= eval_fraction < 1:
        raise ValueError("eval_fraction must be in [0,1)")
    if eval_min_lineages is None:
        eval_min_lineages = 2 if logical_px == 64 else 1
    if type(eval_min_lineages) is not int or eval_min_lineages < 0:
        raise ValueError("eval_min_lineages must be a nonnegative integer")
    if not isinstance(max_procedural_fraction, (int, float)) or not 0 <= max_procedural_fraction < 1:
        raise ValueError("max_procedural_fraction must be in [0,1)")
    if not inputs or type(seed) is not int:
        raise ValueError("At least one input collection and an integer seed are required")
    policy = holdout_policy(blind50, eval50)
    collections = [load_collection(path) for path in inputs]
    roots = {item.root for item in collections}
    if len(roots) != len(collections):
        raise ValueError("Duplicate input collection")
    for root in roots:
        if root == Path(output).resolve() or root in Path(output).resolve().parents:
            raise ValueError("Output must be separate from all input collections")
    review_map = {}
    for path in reviews:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        digest = document.get("collection_sha256")
        if digest in review_map:
            raise ValueError("More than one review file names the same collection")
        review_map[digest] = Path(path)
    input_hashes = {sha256(item.root / "manifest.json") for item in collections}
    if set(review_map) - input_hashes:
        raise ValueError("Review file does not match any input collection")
    eval_names = set()
    if eval_subjects is not None:
        values = json.loads(Path(eval_subjects).read_text(encoding="utf-8"))
        if not isinstance(values, list) or not values or any(not isinstance(value, str) or not normalize_holdout(value) for value in values):
            raise ValueError("eval_subjects must be a nonempty JSON array of subjects")
        if any(policy.matches(value) for value in values):
            raise ValueError("Animation eval subjects overlap frozen holdouts")
        eval_names = {normalize_holdout(value) for value in values}
    candidates, excluded = [], []
    allowed = {"artist", "real"}
    if include_procedural:
        allowed.add("procedural")
    if include_generated:
        allowed.add("generated")
    for collection in collections:
        collection_hash = sha256(collection.root / "manifest.json")
        if collection.manifest["holdouts"] != policy.hashes:
            raise ValueError("Input collection uses different holdout bytes")
        sources = {item["source_id"]: item for item in collection.sources}
        for row in reviewed_sequences(collection, review_map.get(collection_hash)):
            n = grid * grid
            direction = "" if row["direction"] in (None, "unknown") else f", {row['direction']}"
            caption = (f"pixie_anim, {row['subject']}, {row['action']}{direction}, {n}-frame animation sheet, "
                       f"{grid}x{grid} grid, chronological frames left to right then top to bottom, "
                       f"{'looping' if row['loop'] else 'non-looping'}, fixed camera and scale, magenta background")
            reason = None
            if policy.matches(row["subject"], row["caption"], row["action"], str(row["direction"]), caption):
                reason = "frozen_holdout"
            elif row["split"] == "test" or collection.manifest.get("purpose") == "animation_baseline_eval":
                reason = "audit_only"
            elif row["review_status"] != "accepted":
                reason = "not_accepted"
            elif _kind(row) not in allowed:
                reason = "source_kind_not_enabled"
            elif len(row["frames"]) != n:
                reason = "frame_count_mismatch"
            elif row["canvas_px"] != [logical_px, logical_px] or row["anchor_px"] is None:
                reason = f"requires_shared_{logical_px}_canvas_and_anchor"
            elif row["alpha_mode"] != "binary" or row["blend_mode"] != "alpha_over":
                reason = "requires_opaque_cutout"
            if reason:
                excluded.append({"collection_sha256": collection_hash, "sequence_id": row["sequence_id"], "reason": reason})
                continue
            visible_hash, visible_frames = visible_sequence_hash(collection, row)
            candidates.append({"collection": collection, "collection_hash": collection_hash,
                               "row": row, "source": sources[row["source_id"]], "caption": caption,
                               "visible_hash": visible_hash, "visible_frame_hashes": visible_frames})
    if not candidates:
        raise ValueError("No accepted sequences match the selected sheet format and source kinds")
    groups = _groups(candidates)
    splits = {}
    for group_id, entries in groups:
        forced_eval = any(entry["row"]["split"] == "validation" or
                          normalize_holdout(entry["row"]["subject"]) in eval_names for entry in entries)
        value = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest(), 16) / 2**256
        split = "eval" if forced_eval or value < eval_fraction else "train"
        for entry in entries:
            entry["group_id"], entry["split"] = group_id, split
        splits[group_id] = split
    if "train" not in splits.values():
        raise ValueError("All lineage groups are evaluation-only; no training sheets")
    if eval_fraction > 0 and sum(value == "eval" for value in splits.values()) < eval_min_lineages:
        raise ValueError("No evaluation lineage groups meet the minimum; provide more groups or lower eval_min_lineages")
    if eval_fraction > 0 and "eval" not in splits.values():
        raise ValueError("No evaluation lineage group selected; provide frozen eval subjects or more lineages")
    train_entries = [entry for entry in candidates if entry["split"] == "train"]
    artist_entries = [entry for entry in train_entries if _kind(entry["row"]) in {"artist", "real"}]
    nonartist_entries = [entry for entry in train_entries if _kind(entry["row"]) in {"procedural", "generated"}]
    if nonartist_entries and artist_entries:
        max_nonartist = int(len(artist_entries) * max_procedural_fraction / (1 - max_procedural_fraction))
        if len(nonartist_entries) > max_nonartist:
            ordered = sorted(nonartist_entries, key=lambda entry: hashlib.sha256(
                f"{seed}:{entry['collection_hash']}:{entry['row']['sequence_id']}".encode()).hexdigest())
            keep = {id(entry) for entry in ordered[:max_nonartist]}
            candidates = [entry for entry in candidates if entry["split"] != "train" or
                          id(entry) in keep or entry in artist_entries]
            for entry in nonartist_entries:
                if id(entry) not in keep:
                    excluded.append({"collection_sha256": entry["collection_hash"],
                                     "sequence_id": entry["row"]["sequence_id"],
                                     "reason": "procedural_training_cap"})
    root = new_output(output)
    for name in ("train", "eval"):
        (root / name).mkdir()
    records, emitted = [], set()
    for entry in sorted(candidates, key=lambda item: (item["collection_hash"], item["row"]["sequence_id"])):
        row = entry["row"]
        if entry["visible_hash"] in emitted:
            excluded.append({"collection_sha256": entry["collection_hash"], "sequence_id": row["sequence_id"], "reason": "exact_visible_sequence_duplicate"})
            continue
        emitted.add(entry["visible_hash"])
        sheet_id = hashlib.sha256(f"{entry['collection_hash']}:{row['sequence_id']}".encode()).hexdigest()[:24]
        image = Image.new("RGB", (grid * cell_px, grid * cell_px), (255, 0, 255))
        for index, frame in enumerate(row["frames"]):
            with Image.open(relative_file(entry["collection"].root, frame["path"])) as original:
                rgba = original.resize((cell_px, cell_px), Image.Resampling.NEAREST)
                image.paste(rgba.convert("RGB"), ((index % grid) * cell_px, (index // grid) * cell_px), rgba.getchannel("A"))
        relative = f"{entry['split']}/{sheet_id}"
        image.save(root / (relative + ".png"))
        (root / (relative + ".txt")).write_text(entry["caption"] + "\n", encoding="utf-8", newline="\n")
        records.append({"sheet_id": sheet_id, "sequence_id": row["sequence_id"], "parent_collection_sha256": entry["collection_hash"],
                        "lineage_group": row["lineage_group"], "split_group": entry["group_id"], "split": entry["split"],
                        "subject": row["subject"], "action": row["action"], "provenance": row["provenance"],
                        "source": entry["source"], "grid": [grid, grid], "cell_px": cell_px,
                        "anchor_px": row["anchor_px"], "canvas_px": row["canvas_px"],
                        "frame_order": list(range(len(row["frames"]))),
                        "frames": row["frames"], "loop": row["loop"], "caption": entry["caption"],
                        "image": relative + ".png", "caption_file": relative + ".txt", "visible_sequence_sha256": entry["visible_hash"]})
    manifest = {"schema": SCHEMA, "purpose": "animation_lora_sheets", "grid": [grid, grid],
                "frame_count": grid * grid, "cell_px": cell_px, "logical_px": logical_px, "row_major": True, "seed": seed,
                "eval_fraction": eval_fraction, "lineage_splits": splits, "holdouts": policy.hashes,
                "eval_min_lineages": eval_min_lineages, "max_procedural_fraction": max_procedural_fraction,
                "input_manifests": sorted(input_hashes), "review_files": sorted(sha256(path) for path in reviews),
                "eval_subjects_sha256": sha256(eval_subjects) if eval_subjects else None,
                "records": records, "excluded": excluded,
                "counts": {name: sum(row["split"] == name for row in records) for name in ("train", "eval")},
                "files": {path.relative_to(root).as_posix(): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, action="append", default=[])
    parser.add_argument("--grid", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--cell-px", type=int, default=256)
    parser.add_argument("--logical-px", type=int, choices=(32, 64), default=32,
                        help="Required input canvas size; keep 32 and 64 corpora separate")
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--eval-subjects", type=Path)
    parser.add_argument("--eval-min-lineages", type=int,
                        help="Minimum evaluation lineage groups; defaults to 2 for logical 64 and 1 otherwise")
    parser.add_argument("--max-procedural-fraction", type=float, default=0.5,
                        help="Maximum procedural share relative to artist training sheets (default: 0.5)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-procedural", action="store_true")
    parser.add_argument("--include-generated", action="store_true")
    parser.add_argument("--blind50", type=Path, default=DEFAULT_BLIND50)
    parser.add_argument("--eval50", type=Path, default=DEFAULT_EVAL50)
    args = parser.parse_args(argv)
    result = build(args.input, args.output, grid=args.grid, cell_px=args.cell_px,
                   reviews=args.reviews, eval_fraction=args.eval_fraction, seed=args.seed,
                   include_procedural=args.include_procedural, include_generated=args.include_generated,
                   eval_subjects=args.eval_subjects, blind50=args.blind50, eval50=args.eval50, logical_px=args.logical_px,
                   eval_min_lineages=args.eval_min_lineages, max_procedural_fraction=args.max_procedural_fraction)
    print(json.dumps({"status": "created", "counts": result["counts"], "excluded": len(result["excluded"])}))


if __name__ == "__main__":
    main()
