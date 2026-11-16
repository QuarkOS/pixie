"""Create explicitly procedural motion from reviewed static sprites."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw

from .common import (DEFAULT_BLIND50, DEFAULT_EVAL50, holdout_policy, load_collection,
                     new_output, relative_file, reviewed_sequences, save_frames, sha256, write_collection)

MOTIONS = ("bob", "blink", "pulse", "recoil", "spin", "fade")
VERSION = "pixie-motion-v2"


class MotionClippingError(ValueError):
    """The requested transform would discard visible foreground."""

    def __init__(self, message, *, motion=None, frame_index=None, params=None):
        super().__init__(message)
        self.motion = motion
        self.frame_index = frame_index
        self.params = deepcopy(params)


def _number(params, name, default, minimum, maximum):
    value = params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _warp(image, anchor, dx=0, dy=0, scale=1.0, angle=0.0, allow_clipping=False):
    width, height = image.size
    padded = Image.new("RGBA", (width * 3, height * 3))
    padded.paste(image, (width, height))
    cx, cy = anchor[0] + width, anchor[1] + height
    radians = math.radians(angle)
    a, b = math.cos(radians) / scale, math.sin(radians) / scale
    d, e = -b, a
    matrix = (a, b, cx - a * (cx + dx) - b * (cy + dy),
              d, e, cy - d * (cx + dx) - e * (cy + dy))
    result = padded.transform(padded.size, Image.Transform.AFFINE, matrix,
                              resample=Image.Resampling.NEAREST)
    box = result.getchannel("A").getbbox()
    if not allow_clipping:
        if box is None or box[0] < width or box[1] < height or box[2] > width * 2 or box[3] > height * 2:
            raise MotionClippingError("Motion would clip foreground; provide a padded parent canvas")
    return result.crop((width, height, width * 2, height * 2))


def render_motion(image, motion, frame_count, params, seed, anchor):
    """Return same-canvas RGBA frames and the complete effective parameters."""
    if motion not in MOTIONS or frame_count not in (4, 9, 16):
        raise ValueError("Choose a supported motion and 4, 9 or 16 frames")
    if not isinstance(seed, int) or isinstance(seed, bool) or not isinstance(params, dict):
        raise ValueError("Seed must be an integer and params an object")
    image = image.convert("RGBA")
    if (not isinstance(anchor, (list, tuple)) or len(anchor) != 2
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in anchor)
            or not 0 <= anchor[0] <= image.width or not 0 <= anchor[1] <= image.height):
        raise ValueError("A shared anchor inside the canvas is required")
    if image.getchannel("A").getbbox() is None:
        raise ValueError("Parent sprite has no visible pixels")
    allowed = {"allow_clipping"}
    allowed.update({"bob": {"amplitude_px", "phase"}, "blink": {"eye_boxes", "fill_rgba", "ink_rgba", "closed_frames"},
                    "pulse": {"amplitude", "phase"}, "recoil": {"amplitude_px", "axis"},
                    "spin": {"turns"}, "fade": {"start_opacity", "end_opacity"}}[motion])
    if set(params) - allowed:
        raise ValueError("Unknown motion parameter")
    effective = deepcopy(params)
    allow_clipping = effective.setdefault("allow_clipping", False)
    if not isinstance(allow_clipping, bool):
        raise ValueError("allow_clipping must be boolean")
    if motion == "spin" and allow_clipping:
        raise ValueError("Spin never permits clipping; use a padded parent canvas")
    rng = random.Random(seed)
    if motion in ("bob", "pulse"):
        phase = _number(params, "phase", rng.random() * 2 * math.pi, -2 * math.pi, 2 * math.pi)
        effective["phase"] = phase
    if motion == "blink":
        boxes = params.get("eye_boxes")
        if not isinstance(boxes, list) or not boxes:
            raise ValueError("Blink needs explicit eye_boxes; eyes are never inferred")
        for box in boxes:
            if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                    or not 0 <= box[0] < box[2] <= image.width or not 0 <= box[1] < box[3] <= image.height):
                raise ValueError("eye_boxes must be in-canvas [left, top, right, bottom] rectangles")
        for name in ("fill_rgba", "ink_rgba"):
            color = params.get(name)
            if not isinstance(color, list) or len(color) != 4 or any(type(v) is not int or not 0 <= v <= 255 for v in color):
                raise ValueError("Blink requires explicit fill_rgba and ink_rgba colors")
        closed = params.get("closed_frames", [rng.randrange(frame_count)])
        if (not isinstance(closed, list) or not closed or len(set(closed)) != len(closed)
                or any(type(v) is not int or not 0 <= v < frame_count for v in closed)
                or len(closed) == frame_count):
            raise ValueError("closed_frames must select some, but not all, frames")
        effective["closed_frames"] = closed
    def transform(index, **kwargs):
        try:
            return _warp(image, anchor, **kwargs)
        except MotionClippingError as error:
            raise MotionClippingError(str(error), motion=motion, frame_index=index,
                                      params=effective) from error

    frames = []
    for index in range(frame_count):
        t = index / frame_count
        if motion == "bob":
            amplitude = _number(params, "amplitude_px", 1, 0.1, min(image.size))
            effective["amplitude_px"] = amplitude
            result = transform(index, dy=round(amplitude * math.sin(2 * math.pi * t + phase)), allow_clipping=allow_clipping)
        elif motion == "pulse":
            amplitude = _number(params, "amplitude", 0.15, 0.01, 0.8)
            effective["amplitude"] = amplitude
            result = transform(index, scale=1 + amplitude * math.sin(2 * math.pi * t + phase), allow_clipping=allow_clipping)
        elif motion == "recoil":
            amplitude = _number(params, "amplitude_px", 2, 0.1, min(image.size))
            axis = params.get("axis", "x")
            if axis not in ("x", "y"):
                raise ValueError("Recoil axis must be x or y")
            effective.update(amplitude_px=amplitude, axis=axis)
            offset = -round(amplitude * math.sin(math.pi * index / (frame_count - 1)))
            result = transform(index, dx=offset if axis == "x" else 0,
                               dy=offset if axis == "y" else 0, allow_clipping=allow_clipping)
        elif motion == "spin":
            turns = _number(params, "turns", 1, -4, 4)
            if turns == 0 or not float(turns).is_integer():
                raise ValueError("Spin turns must be a nonzero integer for a complete loop")
            effective["turns"] = turns
            result = transform(index, angle=360 * turns * t, allow_clipping=False)
        elif motion == "fade":
            start = _number(params, "start_opacity", 1, 0, 1)
            end = _number(params, "end_opacity", 0, 0, 1)
            if start == end:
                raise ValueError("Fade needs different endpoint opacities")
            effective.update(start_opacity=start, end_opacity=end)
            result = image.copy()
            opacity = start + (end - start) * index / (frame_count - 1)
            alpha = np.rint(np.asarray(image.getchannel("A"), dtype=np.float64) * opacity).astype(np.uint8)
            result.putalpha(Image.fromarray(alpha))
        else:
            result = image.copy()
            if index in closed:
                draw = ImageDraw.Draw(result)
                for left, top, right, bottom in boxes:
                    draw.rectangle((left, top, right - 1, bottom - 1), fill=tuple(params["fill_rgba"]))
                    middle = (top + bottom - 1) // 2
                    draw.line((left, middle, right - 1, middle), fill=tuple(params["ink_rgba"]))
        frames.append(result)
    return frames, effective


def derive(input_dir, output_dir, *, motion, frame_count=4, duration_ms=100,
           seed=0, params=None, sequence_ids=None, reviews=None, blind50=DEFAULT_BLIND50, eval50=DEFAULT_EVAL50):
    collection = load_collection(input_dir)
    policy = holdout_policy(blind50, eval50)
    if collection.root in Path(output_dir).resolve().parents or collection.root == Path(output_dir).resolve():
        raise ValueError("Output must be separate from the input collection")
    if collection.manifest["holdouts"] != policy.hashes:
        raise ValueError("Collection holdout byte hashes differ; use the original holdouts")
    if type(duration_ms) is not int or duration_ms <= 0:
        raise ValueError("duration_ms must be a positive integer")
    selected = set(sequence_ids) if sequence_ids is not None else None
    parents = [row for row in reviewed_sequences(collection, reviews) if selected is None or row["sequence_id"] in selected]
    if not parents or selected is not None and selected != {row["sequence_id"] for row in parents}:
        raise ValueError("No matching parents, or a requested parent is missing")
    sources_by_id = {row["source_id"]: row for row in collection.sources}
    planned, skipped = [], []
    requested_params = {} if params is None else params
    for parent in parents:
        if parent["review_status"] != "accepted" or len(parent["frames"]) != 1:
            raise ValueError("Procedural parents must be explicitly accepted one-frame sprites")
        if policy.matches(parent["subject"], parent["caption"], parent["action"], motion):
            raise ValueError("Frozen holdout appears in procedural input")
        anchor = parent["anchor_px"]
        if anchor is None:
            raise ValueError("Procedural parent needs an explicit shared anchor")
        file = relative_file(collection.root, parent["frames"][0]["path"])
        recipe = {"version": VERSION, "parent_id": parent["sequence_id"], "parent_sha256": sha256(file),
                  "motion": motion, "frame_count": frame_count, "duration_ms": duration_ms,
                  "anchor_px": anchor, "params": deepcopy(requested_params), "seed": seed}
        try:
            with Image.open(file) as image:
                images, effective = render_motion(image, motion, frame_count, requested_params, seed, anchor)
        except MotionClippingError as error:
            recipe["params"] = error.params
            skipped.append({"parent_id": parent["sequence_id"], "parent_source_id": parent["source_id"],
                            "source_url": sources_by_id[parent["source_id"]]["source_url"],
                            "reason": "foreground_clipping", "motion": error.motion,
                            "frame_index": error.frame_index, "recipe": recipe})
            continue
        recipe["params"] = effective
        digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        planned.append((parent, file, images, recipe, digest))
    root = new_output(output_dir)
    sequences, sources = [], []
    for parent, file, images, recipe, digest in planned:
        sequence_id = "motion-" + digest[:24]
        source_id = "procedural-" + digest[:24]
        source_url = sources_by_id[parent["source_id"]]["source_url"]
        original = root / "originals" / source_id / "parent.png"
        original.parent.mkdir(parents=True)
        shutil.copyfile(file, original)
        sources.append({"source_id": source_id, "kind": "procedural", "source_url": source_url,
                        "sha256": sha256(original), "assets": [{"path": original.relative_to(root).as_posix(),
                        "source_url": source_url, "sha256": sha256(original)}], "recipe": recipe,
                        "parent_collection_sha256": sha256(collection.root / "manifest.json"),
                        "parent_source": deepcopy(sources_by_id[parent["source_id"]])})
        row = deepcopy(parent)
        row.update(sequence_id=sequence_id, source_id=source_id, action=motion,
                   caption=f"{parent['subject']}, {motion}", frame_count=frame_count,
                   frames=save_frames(root, sequence_id, images, [duration_ms] * frame_count),
                   loop=motion in ("bob", "blink", "pulse", "spin"), review_status="unreviewed",
                   provenance={"kind": "procedural", "parent_id": parent["sequence_id"]}, parent_id=parent["sequence_id"],
                   normalization={"operation": "procedural", "recipe": recipe}, timing_origin="procedural")
        row.pop("original_masters", None)
        row["alpha_mode"] = "partial" if any(np.any((np.asarray(image)[..., 3] > 0) & (np.asarray(image)[..., 3] < 255)) for image in images) else "binary"
        sequences.append(row)
    status = "all_skipped" if not planned else "complete_with_skips" if skipped else "complete"
    write_collection(root, sequences, sources, extra={
        "operation": VERSION, "holdouts": policy.hashes,
        "review_sha256": sha256(reviews) if reviews else None,
        "parent_collection_sha256": sha256(collection.root / "manifest.json"),
        "status": status, "counts": {"requested": len(parents), "created": len(planned), "skipped": len(skipped)},
        "skipped": skipped,
    })
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion", choices=MOTIONS, required=True)
    parser.add_argument("--frames", type=int, choices=(4, 9, 16), default=4)
    parser.add_argument("--duration-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--params", type=Path, help="Motion parameters JSON; blink requires eye boxes and colors")
    parser.add_argument("--ids", type=Path, help="Optional JSON array of accepted static sequence IDs")
    parser.add_argument("--reviews", type=Path, help="Manifest-bound human review JSON")
    parser.add_argument("--blind50", type=Path, default=DEFAULT_BLIND50)
    parser.add_argument("--eval50", type=Path, default=DEFAULT_EVAL50)
    args = parser.parse_args(argv)
    params = json.loads(args.params.read_text()) if args.params else {}
    ids = json.loads(args.ids.read_text()) if args.ids else None
    if ids is not None and (not isinstance(ids, list) or any(not isinstance(item, str) for item in ids) or len(ids) != len(set(ids))):
        parser.error("--ids must contain a JSON array of unique sequence IDs")
    root = derive(args.input, args.output, motion=args.motion, frame_count=args.frames,
           duration_ms=args.duration_ms, seed=args.seed, params=params, sequence_ids=ids, reviews=args.reviews,
           blind50=args.blind50, eval50=args.eval50)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps({"status": manifest["status"], "counts": manifest["counts"], "motion": args.motion,
                      "provenance": "procedural", "review_status": "unreviewed"}))
    return 3 if manifest["status"] == "all_skipped" else 0


if __name__ == "__main__":
    raise SystemExit(main())
