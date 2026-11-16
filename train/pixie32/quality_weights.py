"""Conservative simple-tile signals and hash-bound sampling multipliers."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from train.pixie32.audit_r1 import contained, load_population, write_json
from train.pixie32.data import sha256


def simple_tile(rgba):
    opaque = rgba[..., 3] >= 128
    yy, xx = np.nonzero(opaque)
    if not len(yy):
        return dict(simple_tile=False, rectangle_fill=0., dominant_fraction=0., transition_fraction=0.)
    x0, x1, y0, y1 = int(xx.min()), int(xx.max()) + 1, int(yy.min()), int(yy.max()) + 1
    pixels = rgba[y0:y1, x0:x1, :3]
    mask = opaque[y0:y1, x0:x1]
    rectangle_fill = float(mask.mean())
    # Ignore up to a two-pixel border so an outline does not hide a flat tile.
    interior = pixels[2:-2, 2:-2] if min(mask.shape) >= 8 else pixels
    core_mask = mask[2:-2, 2:-2] if min(mask.shape) >= 8 else mask
    colors, counts = np.unique(interior[core_mask], axis=0, return_counts=True)
    dominant = float(counts.max() / core_mask.sum()) if len(counts) else 0.
    changed = adjacent = 0
    for axis in (0, 1):
        left = [slice(None)] * 2
        right = [slice(None)] * 2
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        valid = core_mask[tuple(left)] & core_mask[tuple(right)]
        delta = np.any(interior[tuple(left)] != interior[tuple(right)], axis=-1)
        changed += int((delta & valid).sum())
        adjacent += int(valid.sum())
    transitions = changed / adjacent if adjacent else 0.
    # This narrow geometry signal deliberately leaves textured rectangles alone.
    flagged = (mask.size >= 144 and min(mask.shape) >= 8 and rectangle_fill >= .98
               and dominant >= .92 and transitions <= .10)
    return dict(simple_tile=bool(flagged), rectangle_fill=rectangle_fill,
                dominant_fraction=dominant, transition_fraction=transitions, visible_colors=len(colors))


def load_sample_weights(path, root, r1, holdout_hashes):
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {"metadata": sha256(root / "final/metadata.jsonl"),
                "provenance": sha256(root / "manifest.jsonl"), "r1": sha256(r1), "holdouts": holdout_hashes}
    if value.get("schema") != "pixie32-sample-weights-v1" or value.get("inputs") != expected:
        raise ValueError("Sample weights do not match dataset/R1/original holdout bytes")
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Sample weights require nonempty rows")
    result = {}
    for row in rows:
        rid, multiplier = row["id"], row["multiplier"]
        if not isinstance(rid, str) or rid in result or type(multiplier) not in (float, int) or not math.isfinite(multiplier) or not 0 < multiplier <= 1:
            raise ValueError("Sample weight IDs must be unique and multipliers finite in (0,1]")
        if not isinstance(row["file"], str):
            raise ValueError("Sample weight file must be relative")
        image = contained(root, row["file"])
        if image.stem != rid or sha256(image) != row["image_sha256"] or sha256(image.with_suffix(".txt")) != row["caption_sha256"]:
            raise ValueError("Sample weight image/caption identity changed")
        result[rid] = row
    return result


def prepare(args):
    if args.out.exists() or not math.isfinite(args.simple_weight) or not 0 < args.simple_weight <= 1:
        raise ValueError("Choose a fresh output folder and simple weight in (0,1]")
    rows, policy, excluded = load_population(args)
    if args.expect_rows is not None and len(rows) != args.expect_rows:
        raise ValueError("Non-holdout population differs from expected rows")
    output, flagged = [], []
    for row in rows:
        image_path = contained(args.data_root, row["file"])
        with Image.open(image_path) as image:
            rgba = np.array(image.convert("RGBA"))
        if rgba.shape != (32, 32, 4):
            raise ValueError("Quality weights require native 32x32 images")
        signal = simple_tile(rgba)
        weighted = dict(id=row["id"], file=row["file"], image_sha256=sha256(image_path),
                        caption_sha256=sha256(image_path.with_suffix(".txt")),
                        multiplier=args.simple_weight if signal["simple_tile"] else 1., **signal)
        output.append(weighted)
        if signal["simple_tile"]:
            flagged.append(row)
    args.out.mkdir(parents=True, exist_ok=False)
    weights = dict(schema="pixie32-sample-weights-v1", inputs={
        "metadata": sha256(args.data_root / "final/metadata.jsonl"), "provenance": sha256(args.data_root / "manifest.jsonl"),
        "r1": sha256(args.r1), "holdouts": policy.hashes}, rows=output)
    write_json(args.out / "sample_weights.json", weights)
    write_json(args.out / "review.json", flagged)
    for start in range(0, len(flagged), 40):
        page = Image.new("RGB", (8 * 144, 5 * 160), (180, 180, 180))
        draw = ImageDraw.Draw(page)
        for offset, row in enumerate(flagged[start:start + 40]):
            x, y = offset % 8 * 144, offset // 8 * 160
            with Image.open(contained(args.data_root, row["file"])) as image:
                sprite = image.convert("RGBA").resize((128, 128), Image.Resampling.NEAREST)
            page.paste(sprite, (x, y), sprite)
            draw.text((x, y + 130), str(start + offset), fill="black")
        page.save(args.out / f"simple-tiles-{start // 40:03d}.png")
    result = dict(schema="pixie32-quality-signals-v1", rows=len(rows), simple_tiles=len(flagged),
                  holdouts_excluded=excluded, flagged_subjects=dict(Counter(row["subject"] for row in flagged)),
                  proposed_multiplier=args.simple_weight, recommendation_status="untested heuristic; review before building a fresh cache",
                  criteria={"bbox_area_min": 144, "bbox_side_min": 8, "rectangle_fill_min": .98,
                            "interior_border_ignored_px": 2, "interior_dominant_rgb_fraction_min": .92,
                            "interior_adjacent_rgb_transition_fraction_max": .10},
                  interpretation="Simple rectangles can be valid assets. This flags visual simplicity, not junk or caption mismatch. Weights do not alter split selection.")
    write_json(args.out / "quality-summary.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("data-root", "r1", "blind50", "eval50", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--simple-weight", type=float, default=.25, help="untested multiplier for reviewed simple tiles; not a junk threshold")
    p.add_argument("--expect-rows", type=int)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        result = prepare(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        p.error(f"quality weighting refused ({type(error).__name__})")
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
