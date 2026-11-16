"""Import explicit pack sequences without per-frame alignment or palette fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import shutil
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from .common import (SCHEMA, _safe_id, holdout_policy, new_output, relative_file,
                     save_frames, sha256, write_collection)


def public_url(value):
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("source URL must be public HTTP(S), without credentials, query or fragment")
    return value


def provenance_details(value):
    if not isinstance(value, dict):
        raise ValueError("provenance must be an object")
    for field, item in value.items():
        if not isinstance(field, str):
            raise ValueError("provenance keys must be strings")
        if isinstance(item, dict):
            provenance_details(item)
        elif isinstance(item, list):
            for nested in item:
                provenance_details({"value": nested})
        elif isinstance(item, str):
            if re.search(r"\b[A-Za-z]:[\\/]|^/|\\\\", item):
                raise ValueError("provenance must not contain absolute or private paths")
            if item.startswith(("http://", "https://")):
                public_url(item)
        elif item is not None and type(item) not in (int, float, bool):
            raise ValueError("provenance must contain JSON values")
    return value


def duration(value):
    if type(value) is not int or value <= 0:
        raise ValueError("frame durations must be explicit positive integer milliseconds")
    return value


def durations(spec, count, default_duration_ms=None):
    values = spec.get("durations_ms", [spec.get("duration_ms", default_duration_ms)] * count)
    if not isinstance(values, list) or len(values) != count:
        raise ValueError("duration count must match explicit frame order")
    return [duration(value) for value in values]


def rgba(path):
    with Image.open(path) as image:
        return image.convert("RGBA")


def _rect(value, label):
    if isinstance(value, dict):
        value = [value.get(key) for key in ("x", "y", "w", "h")]
    if not isinstance(value, list) or len(value) != 4 or any(type(v) is not int for v in value):
        raise ValueError(f"{label} must be integer x, y, width, height")
    x, y, width, height = value
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError(f"invalid {label}")
    return x, y, width, height


def _crop(image, rect):
    x, y, width, height = rect
    if x + width > image.width or y + height > image.height:
        raise ValueError("frame rectangle exceeds sheet bounds")
    return image.crop((x, y, x + width, y + height))


def source_duration(value, spec, default_duration_ms):
    if value is None or type(value) is int and value == 0:
        value = spec.get("duration_ms", default_duration_ms)
    return duration(value)


def fallback_origin(spec):
    return "annotated" if spec.get("duration_ms") is not None or spec.get("durations_ms") is not None else "default"


def decode_sequence(input_root, spec, default_duration_ms=None):
    kind = spec["format"]
    assets = []
    if kind == "folder":
        entries = spec.get("frames")
        if not isinstance(entries, list) or not entries:
            raise ValueError("folder frames must give explicit ordered paths and durations")
        paths = [relative_file(input_root, entry["path"]) for entry in entries]
        images = [rgba(path) for path in paths]
        timing = [duration(entry.get("duration_ms", spec.get("duration_ms", default_duration_ms))) for entry in entries]
        assets = paths
        origin = "annotated" if all("duration_ms" in entry or spec.get("duration_ms") is not None for entry in entries) else "default"
    elif kind == "sheet":
        path = relative_file(input_root, spec["path"])
        sheet = rgba(path)
        order = spec.get("order")
        cell = spec.get("cell_px")
        if not isinstance(order, list) or not order or not isinstance(cell, list) or len(cell) != 2:
            raise ValueError("sheet requires cell_px and explicit ordered [column,row] cells")
        if any(type(value) is not int or value <= 0 for value in cell):
            raise ValueError("invalid cell dimensions")
        images = []
        for point in order:
            if not isinstance(point, list) or len(point) != 2 or any(type(v) is not int or v < 0 for v in point):
                raise ValueError("invalid sheet cell coordinate")
            images.append(_crop(sheet, (point[0] * cell[0], point[1] * cell[1], *cell)))
        timing = durations(spec, len(images), default_duration_ms)
        assets, origin = [path], fallback_origin(spec)
    elif kind == "gif":
        path = relative_file(input_root, spec["path"])
        images, timing = [], []
        fallback_used = False
        with Image.open(path) as gif:
            if gif.format != "GIF":
                raise ValueError("GIF import requires a GIF file")
            for index in range(gif.n_frames):
                gif.seek(index)
                images.append(gif.convert("RGBA").copy())
                fallback_used |= gif.info.get("duration") in (None, 0)
                timing.append(source_duration(gif.info.get("duration"), spec, default_duration_ms))
        assets, origin = [path], "source_metadata"
        if fallback_used:
            origin = fallback_origin(spec)
    elif kind == "aseprite":
        path = relative_file(input_root, spec["path"])
        data = json.loads(path.read_text(encoding="utf-8"))
        values = data.get("frames")
        if isinstance(values, dict):
            names = spec.get("frame_order", list(values))
            values = [values[name] for name in names]
        if not isinstance(values, list) or not values:
            raise ValueError("Aseprite JSON must have ordered frame metadata")
        sheet_path = relative_file(input_root, spec.get("image", str(Path(spec["path"]).parent / data["meta"]["image"])))
        sheet = rgba(sheet_path)
        indices = list(range(len(values)))
        if len(data.get("meta", {}).get("frameTags", [])) > 1 and "tag" not in spec:
            raise ValueError("Aseprite export has multiple actions; select an explicit tag")
        if "tag" in spec:
            tags = [tag for tag in data.get("meta", {}).get("frameTags", []) if tag.get("name") == spec["tag"]]
            if len(tags) != 1:
                raise ValueError("Aseprite tag must identify one action")
            tag = tags[0]
            if type(tag.get("from")) is not int or type(tag.get("to")) is not int or not 0 <= tag["from"] <= tag["to"] < len(values):
                raise ValueError("invalid Aseprite tag frame range")
            indices = list(range(tag["from"], tag["to"] + 1))
            mode = tag.get("direction", "forward")
            if mode == "reverse":
                indices.reverse()
            elif mode == "pingpong":
                indices += indices[-2:0:-1]
            elif mode == "pingpong_reverse":
                indices.reverse()
                indices += indices[-2:0:-1]
            elif mode != "forward":
                raise ValueError("unsupported Aseprite tag direction")
        images, timing = [], []
        for index in indices:
            entry = values[index]
            if entry.get("rotated", False):
                raise ValueError("rotated Aseprite packing is not supported; export without rotation")
            cut = _crop(sheet, _rect(entry["frame"], "packed frame"))
            size = entry.get("sourceSize")
            if not isinstance(size, dict) or any(type(size.get(k)) is not int or size[k] <= 0 for k in ("w", "h")):
                raise ValueError("sourceSize is required for each Aseprite frame")
            offset = _rect(entry.get("spriteSourceSize", {"x": 0, "y": 0, "w": size["w"], "h": size["h"]}), "source frame")
            x, y, width, height = offset
            if cut.size != (width, height) or x + width > size["w"] or y + height > size["h"]:
                raise ValueError("trimmed frame does not fit source canvas")
            full = Image.new("RGBA", (size["w"], size["h"]))
            full.paste(cut, (x, y))
            images.append(full)
            timing.append(source_duration(entry.get("duration"), spec, default_duration_ms))
        assets, origin = [path, sheet_path], "source_metadata"
        if any(values[index].get("duration") in (None, 0) for index in indices):
            origin = fallback_origin(spec)
    else:
        raise ValueError("format must be folder, sheet, gif or aseprite")
    if len({image.size for image in images}) != 1:
        raise ValueError("frames must share one source canvas; restore source offsets before importing")
    return images, timing, list(dict.fromkeys(assets)), origin


def working_frames(images, anchor, key_rgb=None, crop_to_content=False, crop_padding=0, alpha_threshold=None):
    if key_rgb is not None and (not isinstance(key_rgb, (list, tuple)) or len(key_rgb) != 3
                               or any(type(value) is not int or not 0 <= value <= 255 for value in key_rgb)):
        raise ValueError("key_rgb must contain three integer RGB values from 0 to 255")
    if type(crop_to_content) is not bool or type(crop_padding) is not int or crop_padding < 0:
        raise ValueError("crop_to_content must be boolean and crop_padding a nonnegative integer")
    if crop_padding and not crop_to_content:
        raise ValueError("crop_padding requires crop_to_content")
    if alpha_threshold is not None and (type(alpha_threshold) is not int or not 1 <= alpha_threshold <= 255):
        raise ValueError("alpha_threshold must be an integer from 1 to 255")
    result, keyed_counts = [], []
    thresholded_counts = []
    for image in images:
        pixels = np.array(image, copy=True)
        keyed = np.zeros(pixels.shape[:2], dtype=bool)
        if key_rgb is not None:
            keyed = np.all(pixels[..., :3] == key_rgb, axis=-1) & (pixels[..., 3] > 0)
            pixels[keyed, 3] = 0
        keyed_counts.append(int(keyed.sum()))
        if alpha_threshold is None:
            thresholded_counts.append(0)
        else:
            alpha = pixels[..., 3].copy()
            pixels[..., 3] = np.where(alpha >= alpha_threshold, 255, 0).astype(np.uint8)
            thresholded_counts.append(int(np.count_nonzero(alpha != pixels[..., 3])))
        result.append(Image.fromarray(pixels))
    width, height = images[0].size
    box = [0, 0, width, height]
    boxes = [image.getchannel("A").getbbox() for image in result]
    visible = [value for value in boxes if value is not None]
    if not visible and (key_rgb is not None or crop_to_content):
        raise ValueError("sequence has no visible pixels after keying; check key_rgb and source frames")
    if crop_to_content:
        box = [min(value[0] for value in visible), min(value[1] for value in visible),
               max(value[2] for value in visible), max(value[3] for value in visible)]
        if anchor is not None:
            if not 0 <= anchor[0] <= width or not 0 <= anchor[1] <= height:
                raise ValueError("crop-to-content requires the shared anchor inside the source canvas")
            box = [min(box[0], math.floor(anchor[0])), min(box[1], math.floor(anchor[1])),
                   max(box[2], math.ceil(anchor[0])), max(box[3], math.ceil(anchor[1]))]
        box = [max(0, box[0] - crop_padding), max(0, box[1] - crop_padding),
               min(width, box[2] + crop_padding), min(height, box[3] + crop_padding)]
        result = [image.crop(box) for image in result]
    adjusted_anchor = None if anchor is None else [anchor[0] - box[0], anchor[1] - box[1]]
    metadata = {"key_rgb": list(key_rgb) if key_rgb is not None else None,
                "key_tolerance": 0, "keyed_pixels_per_frame": keyed_counts,
                "alpha_threshold": alpha_threshold,
                "alpha_thresholded_pixels_per_frame": thresholded_counts,
                "crop_to_content": crop_to_content, "crop_padding": crop_padding,
                "crop_box_px": box, "original_canvas_px": [width, height]}
    return result, adjusted_anchor, metadata


def normalize_frames(images, canvas, anchor):
    source_width, source_height = images[0].size
    width, height = canvas
    scale = min(width / source_width, height / source_height)
    resized = [max(1, round(source_width * scale)), max(1, round(source_height * scale))]
    offset = [(width - resized[0]) // 2, (height - resized[1]) // 2]
    normalized = []
    for image in images:
        target = Image.new("RGBA", canvas)
        target.paste(image.resize(resized, Image.Resampling.NEAREST), tuple(offset))
        normalized.append(target)
    def snap(value):
        rounded = round(value)
        return rounded if abs(value - rounded) < 1e-9 else value
    transformed_anchor = None if anchor is None else [
        snap(anchor[0] * resized[0] / source_width + offset[0]),
        snap(anchor[1] * resized[1] / source_height + offset[1]),
    ]
    return normalized, transformed_anchor, {"source_canvas_px": [source_width, source_height], "target_canvas_px": list(canvas), "resized_px": resized, "offset_px": offset, "resample": "nearest", "tool_version": "animation.ingest-v2"}


def ingest(input_root, spec_path, output, canvas=(32, 32), blind50=None, eval50=None,
           *, key_rgb=None, crop_to_content=False, crop_padding=0, default_duration_ms=None,
           alpha_threshold=None):
    output_path = Path(output).resolve()
    for source_path in (Path(input_root).resolve(), Path(spec_path).resolve().parent):
        if output_path == source_path or source_path in output_path.parents:
            raise ValueError("output must be outside the input pack and spec directories")
    policy = holdout_policy(blind50, eval50)
    document = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    specs = document.get("sequences")
    if not isinstance(specs, list) or not specs:
        raise ValueError("spec must contain a nonempty sequences list")
    if len(canvas) != 2 or any(type(v) is not int or v <= 0 for v in canvas):
        raise ValueError("canvas must contain positive integer width and height")
    if default_duration_ms is not None:
        duration(default_duration_ms)
    prepared, ids = [], set()
    for spec in specs:
        sid = _safe_id(spec["sequence_id"], "sequence_id")
        source_id = _safe_id(spec.get("source_id", sid), "source_id")
        if sid in ids:
            raise ValueError("duplicate sequence ID")
        ids.add(sid)
        for key in ("subject", "action", "lineage_group"):
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                raise ValueError(f"{key} is required")
        for key in ("caption", "direction", "color_space"):
            if key in spec and (not isinstance(spec[key], str) or not spec[key].strip()):
                raise ValueError(f"{key} must be a nonempty string")
        if policy.matches(spec["subject"], spec.get("caption", ""), spec["action"]):
            raise ValueError("frozen holdout matches an import subject, caption or action")
        source_url = public_url(spec["source_url"])
        if spec.get("kind", "artist") not in {"artist", "generated"}:
            raise ValueError("import kind must be artist or generated")
        details = provenance_details(spec.get("provenance", {}))
        if policy.matches(json.dumps(details, ensure_ascii=False)):
            raise ValueError("frozen holdout matches supplied provenance")
        if type(spec.get("loop")) is not bool:
            raise ValueError("loop must be explicitly true or false")
        if "anchor_px" not in spec:
            raise ValueError("anchor_px must be an explicit shared [x,y] or null")
        anchor = spec["anchor_px"]
        if anchor is not None and (not isinstance(anchor, list) or len(anchor) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in anchor)):
            raise ValueError("invalid shared anchor")
        images, timing, assets, origin = decode_sequence(input_root, spec, default_duration_ms)
        working, adjusted_anchor, preprocessing = working_frames(
            images, anchor, spec.get("key_rgb", key_rgb),
            spec.get("crop_to_content", crop_to_content), spec.get("crop_padding", crop_padding),
            spec.get("alpha_threshold", alpha_threshold))
        normalized, anchor, transform = normalize_frames(working, canvas, adjusted_anchor)
        transform.update(preprocessing)
        transform["default_duration_ms"] = default_duration_ms
        prepared.append((spec, sid, source_id, source_url, images, timing, assets, origin, normalized, anchor, transform))
    if len({item[2] for item in prepared}) != len(prepared):
        raise ValueError("each import entry needs a unique source_id")
    root = new_output(output)
    sequences, sources = [], []
    for spec, sid, source_id, source_url, images, timing, assets, origin, normalized, anchor, transform in prepared:
        original_dir = root / "originals" / source_id
        original_dir.mkdir(parents=True)
        asset_records = []
        for index, path in enumerate(assets):
            target = original_dir / f"{index:04d}-{path.name}"
            shutil.copyfile(path, target)
            asset_records.append({"path": target.relative_to(root).as_posix(), "source_url": source_url, "sha256": sha256(target)})
        masters_dir = root / "masters" / sid
        masters_dir.mkdir(parents=True)
        masters = []
        for index, image in enumerate(images):
            path = masters_dir / f"{index:04d}.png"
            image.save(path)
            masters.append({"path": path.relative_to(root).as_posix(), "file_sha256": sha256(path), "rgba_sha256": hashlib.sha256(image.tobytes()).hexdigest(), "duration_ms": timing[index]})
        frames = save_frames(root, sid, normalized, timing)
        partial = any(any(0 < value < 255 for value in image.getchannel("A").tobytes()) for image in normalized)
        source_hash = asset_records[0]["sha256"] if len(asset_records) == 1 else hashlib.sha256(json.dumps([a["sha256"] for a in asset_records], separators=(",", ":")).encode()).hexdigest()
        kind = spec.get("kind", "artist")
        sources.append({"source_id": source_id, "kind": kind, "source_url": source_url, "sha256": source_hash, "assets": asset_records, "format": spec["format"]})
        sequences.append({"schema_version": SCHEMA, "sequence_id": sid, "source_id": source_id, "lineage_group": spec["lineage_group"], "subject": spec["subject"], "caption": spec.get("caption", spec["subject"]), "action": spec["action"], "direction": spec.get("direction", "unknown"), "loop": spec["loop"], "canvas_px": list(canvas), "anchor_px": anchor, "frames": frames, "frame_count": len(frames), "alpha_mode": "partial" if partial else "binary", "alpha_association": "straight", "color_space": spec.get("color_space", "source_unspecified"), "blend_mode": "alpha_over", "normalization": transform, "original_masters": masters, "timing_origin": origin, "palette_derivation": None, "review_status": "unreviewed", "provenance": {**spec.get("provenance", {}), "kind": kind, "format": spec["format"], "source_tag": spec.get("tag")}, "split": "unassigned"})
    write_collection(root, sequences, sources, {"holdouts": policy.hashes, "ingest_spec_sha256": sha256(spec_path), "purpose": "animation_lora_collection"})
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canvas", type=int, nargs=2, default=[32, 32])
    parser.add_argument("--key-rgb", type=int, nargs=3, metavar=("R", "G", "B"), help="Exact color key on working frames; masters remain unchanged")
    parser.add_argument("--crop-to-content", action="store_true", help="Crop one sequence-wide alpha union, including the shared anchor")
    parser.add_argument("--crop-padding", type=int, default=0, help="Source pixels around the shared content crop")
    parser.add_argument("--default-duration-ms", type=int, help="Explicit annotation for frames without timing; source timing takes priority")
    parser.add_argument("--alpha-threshold", type=int, help="Opt-in alpha cutout: alpha below this value becomes transparent and the rest opaque")
    parser.add_argument("--blind50", type=Path)
    parser.add_argument("--eval50", type=Path)
    args = parser.parse_args(argv)
    ingest(args.input, args.spec, args.output, tuple(args.canvas), args.blind50, args.eval50,
           key_rgb=args.key_rgb, crop_to_content=args.crop_to_content, crop_padding=args.crop_padding,
           default_duration_ms=args.default_duration_ms, alpha_threshold=args.alpha_threshold)
    print("Imported sequences into a new hash-bound collection")


if __name__ == "__main__":
    main()
