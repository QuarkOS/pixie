"""Score animation sequences and write uncalibrated review artifacts.

The scorer reports independent measurements. It deliberately does not combine
them into an admission score: thresholds must be calibrated against labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import load_collection, new_output, relative_file, sha256


SCHEMA = "pixie-animation-score-v1"
STATUS = "UNCALIBRATED"


def _json_value(value: Any) -> Any:
    """Convert numpy scalars and arrays to JSON-safe values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _frame_path(root: Path, frame: Mapping[str, Any]) -> Path:
    return relative_file(root, frame.get("path", ""))


def _load_frames(root: Path, sequence: Mapping[str, Any]) -> list[Image.Image]:
    frames = sequence.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("sequence has no frames")
    images: list[Image.Image] = []
    for frame in frames:
        image_path = _frame_path(root, frame)
        with Image.open(image_path) as source:
            images.append(source.convert("RGBA"))
    return images


def _bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    alpha = np.asarray(image, dtype=np.uint8)[..., 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _rgba_hash(image: Image.Image) -> str:
    return hashlib.sha256(_visible_rgba(image).tobytes()).hexdigest()


def _visible_rgba(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    pixels[pixels[..., 3] == 0, :3] = 0
    return pixels


def _bbox_metrics(images: list[Image.Image], anchor: Any) -> dict[str, Any]:
    boxes = [_bbox(image) for image in images]
    centers = [
        None if box is None else [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
        for box in boxes
    ]
    valid = np.asarray([center for center in centers if center is not None], dtype=np.float64)
    anchor_xy: list[float] | None = None
    if isinstance(anchor, (list, tuple)) and len(anchor) == 2:
        try:
            anchor_xy = [float(anchor[0]), float(anchor[1])]
        except (TypeError, ValueError):
            anchor_xy = None
    offsets = None if anchor_xy is None else [
        None if center is None else [center[0] - anchor_xy[0], center[1] - anchor_xy[1]]
        for center in centers
    ]
    # This is a geometric proxy. It includes intended motion and is not a
    # tracked-camera judgment. Human review calibrates whether it is jitter.
    steps = np.asarray([np.asarray(right) - np.asarray(left) for left, right in zip(centers, centers[1:])
                        if left is not None and right is not None], dtype=np.float64)
    if len(valid) >= 2:
        jitter = (float(np.sqrt(np.mean(np.sum(np.square(steps - np.median(steps, axis=0)), axis=1))))
                  if len(steps) else None)
        center_span = float(np.linalg.norm(valid.max(axis=0) - valid.min(axis=0)))
    else:
        jitter = 0.0 if len(valid) else None
        center_span = 0.0 if len(valid) else None
    return {
        "bbox": boxes,
        "bbox_center": centers,
        "anchor_px": anchor_xy,
        "bbox_anchor_offset": offsets,
        "bbox_jitter_px": jitter,
        "bbox_center_span_px": center_span,
        "metric_note": "bbox jitter is RMS residual displacement from the median adjacent step; it includes intended motion and is not tracked anchor jitter",
    }


def _palette_metrics(images: list[Image.Image]) -> dict[str, Any]:
    palettes: list[set[tuple[int, int, int]]] = []
    for image in images:
        pixels = np.asarray(image, dtype=np.uint8)
        visible = pixels[..., 3] > 0
        colors = pixels[..., :3][visible]
        palettes.append({tuple(map(int, color)) for color in colors})
    union = set().union(*palettes)
    intersection = set(palettes[0]).intersection(*palettes[1:]) if palettes else set()
    jaccards: list[float] = []
    for left, right in zip(palettes, palettes[1:]):
        denominator = len(left | right)
        jaccards.append(len(left & right) / denominator if denominator else 1.0)
    return {
        "visible_palette_size": [len(palette) for palette in palettes],
        "visible_palette_union": len(union),
        "visible_palette_intersection": len(intersection),
        "palette_consistency_jaccard": len(intersection) / len(union) if union else 1.0,
        "adjacent_palette_jaccard": jaccards,
        "metric_note": "transparent RGB values are ignored; no palette refit is performed",
    }


def _motion_metrics(images: list[Image.Image]) -> dict[str, Any]:
    hashes = [_rgba_hash(image) for image in images]
    changes: list[float] = []
    for before, after in zip(images, images[1:]):
        left = _visible_rgba(before)
        right = _visible_rgba(after)
        changes.append(float(np.mean(np.any(left != right, axis=-1))))
    return {
        "visible_rgba_sha256": hashes,
        "adjacent_changed_fraction": changes,
        "all_frames_identical": len(set(hashes)) == 1,
        "identical_adjacent_fraction": (
            sum(a == b for a, b in zip(hashes, hashes[1:])) / max(len(hashes) - 1, 1)
        ),
    }


def _loop_metrics(images: list[Image.Image], loop: Any) -> dict[str, Any]:
    if not bool(loop):
        return {"applicable": False, "status": "not_applicable"}
    first = _visible_rgba(images[0]).astype(np.int16)
    last = _visible_rgba(images[-1]).astype(np.int16)
    difference = np.abs(first - last)
    return {
        "applicable": True,
        "status": "measured",
        "rgba_pixel_match": float(np.all(first == last, axis=-1).mean()),
        "rgba_mean_abs_error": float(difference.mean()),
        "closing_transition_changed_fraction": float(np.any(first != last, axis=-1).mean()),
        "metric_note": "endpoint comparison is a loop-closure proxy; it does not judge apparent velocity",
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if not math.isfinite(left_norm) or not math.isfinite(right_norm) or left_norm == 0 or right_norm == 0:
        raise ValueError("identity feature vectors must be finite and nonzero")
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("identity feature vectors must share one nonempty dimension")
    return float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))


def _identity_metrics(sequence: Mapping[str, Any], root: Path, features: Mapping[tuple[str, str], np.ndarray] | None) -> dict[str, Any]:
    if features is None:
        return {
            "status": "unavailable",
            "embedding_similarity": None,
            "metric_note": "no semantic embedding model or hash-bound feature sidecar was supplied",
        }
    sequence_id = str(sequence.get("sequence_id", ""))
    vectors: list[np.ndarray] = []
    missing: list[str] = []
    for frame in sequence.get("frames", []):
        path = str(frame.get("path", ""))
        key = (sequence_id, path)
        vector = features.get(key)
        if vector is None:
            missing.append(path)
        else:
            vectors.append(np.asarray(vector, dtype=np.float64))
    if missing or len(vectors) != len(sequence.get("frames", [])):
        return {
            "status": "invalid_features",
            "embedding_similarity": None,
            "missing_frame_paths": missing,
            "metric_note": "every ordered frame needs one hash-bound feature vector",
        }
    try:
        adjacent = [_cosine(left, right) for left, right in zip(vectors, vectors[1:])]
        all_pairs = [_cosine(vectors[0], vector) for vector in vectors[1:]]
    except ValueError as error:
        return {"status": "invalid_features", "embedding_similarity": None, "error": str(error)}
    return {
        "status": "available",
        "embedding_similarity": {
            "adjacent": adjacent,
            "adjacent_mean": float(np.mean(adjacent)) if adjacent else 1.0,
            "adjacent_min": float(np.min(adjacent)) if adjacent else 1.0,
            "from_first": all_pairs,
            "from_first_mean": float(np.mean(all_pairs)) if all_pairs else 1.0,
            "from_first_min": float(np.min(all_pairs)) if all_pairs else 1.0,
        },
        "metric_note": "semantic identity is measured from a supplied local model or verified feature sidecar",
    }


def score_sequence(
    root: Path,
    sequence: Mapping[str, Any],
    features: Mapping[tuple[str, str], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Return independent uncalibrated metrics for one sequence."""
    images = _load_frames(root, sequence)
    normalization = sequence.get("normalization", {})
    source_canvas = normalization.get("source_canvas_px")
    resized = normalization.get("resized_px")
    resampled = source_canvas is not None and resized is not None and source_canvas != resized
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "sequence_id": str(sequence.get("sequence_id", "")),
        "source_id": str(sequence.get("source_id", "")),
        "lineage_group": str(sequence.get("lineage_group", "")),
        "subject": sequence.get("subject"),
        "action": sequence.get("action"),
        "frame_count": len(images),
        "measurement_space": {
            "canvas_px": list(images[0].size),
            "frame_source": "stored_collection_frames_before_preview_resize",
            "source_canvas_px": source_canvas,
            "source_resampled_during_ingest": resampled,
            "palette_comparability": "compare_only_matching_key_crop_and_normalization_policies",
            "normalization": normalization,
        },
        "identity": _identity_metrics(sequence, root, features),
        "geometry": _bbox_metrics(images, sequence.get("anchor_px")),
        "palette": _palette_metrics(images),
        "motion": _motion_metrics(images),
        "loop": _loop_metrics(images, sequence.get("loop")),
        "limitations": [
            "uncalibrated metrics are for review and ranking only",
            "bbox jitter is a geometric proxy and may measure intended motion",
            "endpoint loop error does not assess timing or apparent velocity",
        ],
    }


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _background(size: tuple[int, int], mode: str) -> Image.Image:
    if mode == "light":
        return Image.new("RGBA", size, (244, 244, 244, 255))
    if mode == "dark":
        return Image.new("RGBA", size, (32, 32, 32, 255))
    image = Image.new("RGBA", size, (220, 220, 220, 255))
    draw = ImageDraw.Draw(image)
    tile = max(1, min(size) // 16)
    for y in range(0, size[1], tile):
        for x in range(0, size[0], tile):
            if (x // tile + y // tile) % 2:
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(170, 170, 170, 255))
    return image


def _compose(image: Image.Image, mode: str, size: int) -> Image.Image:
    ratio = size / max(image.size)
    dimensions = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    source = image.convert("RGBA").resize(dimensions, Image.Resampling.NEAREST)
    canvas = _background((size, size), mode)
    canvas.alpha_composite(source, ((size - source.width) // 2, (size - source.height) // 2))
    return canvas.convert("RGB")


def export_review(root: Path, sequence: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Write contact sheets and GIFs with explicit playback timing."""
    output.mkdir(parents=True, exist_ok=True)
    images = _load_frames(root, sequence)
    tile = 128
    label = 24
    columns = len(images)
    sheet = Image.new("RGB", (columns * tile, (tile + label) * 3), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(12)
    for row, background in enumerate(("checker", "light", "dark")):
        for index, image in enumerate(images):
            y = row * (tile + label)
            sheet.paste(_compose(image, background, tile), (index * tile, y))
            draw.text((index * tile + 4, y + tile + 4), f"{background}: frame {index + 1}", fill="black", font=font)
    contact_name = f"{sequence['sequence_id']}.contact.png"
    sheet.save(output / contact_name)
    artifacts = {
        "contact_sheet": contact_name,
        "gif": None,
        "gif_source_durations_ms": None,
        "gif_durations_ms": None,
        "gif_requested_durations_ms": None,
        "gif_loop": bool(sequence.get("loop")),
    }
    if images:
        source_durations = [int(frame["duration_ms"]) for frame in sequence["frames"]]
        if any(duration <= 0 for duration in source_durations):
            raise ValueError("GIF frame durations must be positive")
        # GIF stores durations in 10 ms units. Keep both values so review can
        # see the source timing and the timing encoded in the preview.
        durations = [max(10, (duration // 10) * 10) for duration in source_durations]
        gif_frames = [_compose(image, "checker", tile) for image in images]
        gif_name = f"{sequence['sequence_id']}.preview.gif"
        playback = {"loop": 0} if bool(sequence.get("loop")) else {}
        gif_frames[0].save(
            output / gif_name,
            save_all=True,
            append_images=gif_frames[1:],
            duration=durations,
            disposal=2,
            **playback,
        )
        artifacts["gif"] = gif_name
        artifacts["gif_source_durations_ms"] = source_durations
        artifacts["gif_requested_durations_ms"] = durations
        encoded_durations = []
        with Image.open(output / gif_name) as encoded:
            for index in range(encoded.n_frames):
                encoded.seek(index)
                encoded_durations.append(encoded.info["duration"])
        artifacts["gif_durations_ms"] = encoded_durations
    return artifacts


def _load_feature_sidecar(path: Path, root: Path, sequences: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], np.ndarray]:
    """Load an NPZ with exact sequence_id, frame path, and file hash columns."""
    with np.load(path, allow_pickle=False) as data:
        required = {"features", "sequence_ids", "frame_paths", "frame_sha256"}
        if not required <= set(data.files):
            raise ValueError("feature sidecar needs features, sequence_ids, frame_paths and frame_sha256")
        vectors = np.asarray(data["features"], dtype=np.float64)
        ids = np.asarray(data["sequence_ids"])
        paths = np.asarray(data["frame_paths"])
        hashes = np.asarray(data["frame_sha256"])
    if any(values.ndim != 1 or values.dtype.kind != "U" for values in (ids, paths, hashes)):
        raise ValueError("feature identity columns must be one-dimensional unicode arrays")
    if vectors.ndim != 2 or vectors.shape[1] == 0 or len(ids) != len(paths) or len(ids) != len(hashes) or len(ids) != len(vectors):
        raise ValueError("feature sidecar rows do not have matching lengths")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("identity feature vectors must be finite and nonzero")
    expected: dict[tuple[str, str], str] = {}
    for sequence in sequences:
        for frame in sequence.get("frames", []):
            key = (str(sequence.get("sequence_id", "")), str(frame.get("path", "")))
            expected[key] = str(frame.get("file_sha256", ""))
    result: dict[tuple[str, str], np.ndarray] = {}
    for vector, sequence_id, frame_path, digest in zip(vectors, ids, paths, hashes):
        relative_file(root, str(frame_path))
        key = (sequence_id, frame_path)
        if key not in expected or expected[key] != digest:
            raise ValueError("feature sidecar does not match collection frame hashes")
        if key in result:
            raise ValueError("feature sidecar contains duplicate frame keys")
        result[key] = np.asarray(vector, dtype=np.float64)
    if set(result) != set(expected):
        raise ValueError("feature sidecar is missing collection frames")
    return result


def _vision_features(root: Path, sequences: list[Mapping[str, Any]], model_path: Path, device: str) -> dict[tuple[str, str], np.ndarray]:
    """Extract image embeddings from a local Transformers model only."""
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise RuntimeError("--vision-model requires local torch and transformers packages") from error
    if not model_path.is_dir():
        raise ValueError("vision model must be an existing local model directory")
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    result: dict[tuple[str, str], np.ndarray] = {}
    with torch.inference_mode():
        for sequence in sequences:
            for frame in sequence.get("frames", []):
                with Image.open(_frame_path(root, frame)) as source:
                    rgba = source.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (127, 127, 127, 255))
                background.alpha_composite(rgba)
                image = background.convert("RGB")
                batch = processor(images=image, return_tensors="pt")
                batch = {name: value.to(device) for name, value in batch.items()}
                output = model.get_image_features(**batch) if hasattr(model, "get_image_features") else model(**batch)
                if isinstance(output, torch.Tensor):
                    embedding = output
                elif hasattr(output, "pooler_output") and output.pooler_output is not None:
                    embedding = output.pooler_output
                elif hasattr(output, "last_hidden_state"):
                    embedding = output.last_hidden_state.mean(dim=1)
                else:
                    raise RuntimeError("local vision model has no pooler_output or last_hidden_state")
                result[(str(sequence["sequence_id"]), str(frame["path"]))] = embedding[0].detach().cpu().numpy()
    return result


def _write_hash_manifest(output: Path, metadata: dict) -> None:
    files: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != output / "manifest.json":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files[path.relative_to(output).as_posix()] = digest
    (output / "manifest.json").write_text(
        json.dumps({"schema": SCHEMA, "status": STATUS, "files": files, **metadata}, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    input_root = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if output_path == input_root or input_root in output_path.parents:
        raise ValueError("review output must be separate from the input collection")
    collection = load_collection(input_root)
    sequences = list(collection.sequences)
    features = None
    feature_source = {"status": "unavailable"}
    if args.features:
        features = _load_feature_sidecar(Path(args.features), input_root, sequences)
        feature_source = {"kind": "precomputed", "sha256": sha256(Path(args.features)),
                          "model_provenance": "external sidecar; model identity is not inferred"}
    elif args.vision_model:
        features = _vision_features(input_root, sequences, Path(args.vision_model), args.device)
        model_path = Path(args.vision_model)
        inventory = {path.relative_to(model_path).as_posix(): sha256(path)
                     for path in sorted(model_path.rglob("*")) if path.is_file()}
        digest = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
        feature_source = {"kind": "local_vision_model", "model_inventory_sha256": digest,
                          "device": args.device, "alpha_composite": "neutral_srgb_127"}
    output_root = new_output(output_path)
    previews_dir = output_root / "previews"
    previews_dir.mkdir(parents=True)
    with (output_root / "score.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for sequence in sequences:
            score = score_sequence(input_root, sequence, features)
            score["source_manifest_sha256"] = sha256(input_root / "manifest.json")
            score["identity_feature_source"] = feature_source
            artifacts = export_review(input_root, sequence, previews_dir)
            for field in ("contact_sheet", "gif"):
                if artifacts[field] is not None:
                    artifacts[field] = "previews/" + artifacts[field]
            score["review_artifacts"] = artifacts
            stream.write(json.dumps(score, sort_keys=True, default=_json_value) + "\n")
    _write_hash_manifest(output_root, {"source_manifest_sha256": sha256(input_root / "manifest.json"),
                                       "identity_feature_source": feature_source,
                                       "sequence_count": len(sequences)})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write uncalibrated animation consistency metrics and review previews")
    parser.add_argument("--input", required=True, type=Path, help="normalized animation collection")
    parser.add_argument("--output", required=True, type=Path, help="new review output directory")
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument("--features", type=Path, help="hash-bound NPZ semantic feature sidecar")
    identity.add_argument("--vision-model", type=Path, help="local Transformers image model; no network access")
    parser.add_argument("--device", default="cpu", help="vision model device, such as cpu or cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
