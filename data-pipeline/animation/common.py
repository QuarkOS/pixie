"""Shared on-disk records and integrity checks for animation collections."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
import copy
from pathlib import Path, PureWindowsPath
from typing import Mapping, Sequence

SCHEMA = "pixie-animation-alpha-v1"
FROZEN_HOLDOUT_HASHES = {
    "blind50": "7524db9cabd2449f5ec6b48b454bc10e2fed38825ba8c8d1e749311f3b5afb0a",
    "eval50": "a1d6893e8c3c45ab99a845c1726c55547dcde29ff6f7a94b022f067c27a7872f",
}
REQUIRED_SEQUENCE_FIELDS = {
    "sequence_id", "source_id", "lineage_group", "subject", "caption", "action",
    "direction", "loop", "canvas_px", "anchor_px", "frames", "frame_count",
    "alpha_mode", "alpha_association", "color_space", "blend_mode", "normalization",
    "review_status", "provenance", "split",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from train.pixie32.data import Holdouts, DEFAULT_BLIND50, DEFAULT_EVAL50


def holdout_policy(blind50: Path | None = None, eval50: Path | None = None):
    policy = Holdouts(blind50 or DEFAULT_BLIND50, eval50 or DEFAULT_EVAL50)
    if policy.hashes != FROZEN_HOLDOUT_HASHES:
        raise ValueError("Frozen Blind-50/Eval-50 bytes differ; supply the original files. "
                         "Git newline conversion and JSON reformatting change their fingerprints. "
                         "This command never rewrites holdouts")
    return policy


def new_output(path: Path) -> Path:
    """Create an exclusive output directory and refuse an existing path."""
    path = Path(path)
    if not path or path.exists():
        raise FileExistsError("output already exists; choose a new folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def relative_file(root: Path, relative: str | Path) -> Path:
    """Resolve a collection-relative file while rejecting traversal and links."""
    root = Path(root).resolve()
    text = relative.as_posix() if isinstance(relative, Path) else relative
    value = Path(text)
    if value.is_absolute() or PureWindowsPath(text).drive or str(value) in {"", "."} or ".." in value.parts or "\\" in text:
        raise ValueError(f"invalid collection-relative path: {relative}")
    candidate = root / value
    resolved = candidate.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes collection: {relative}")
    current = root
    for part in value.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink is not allowed in collection path: {relative}")
    return candidate


def _safe_id(value: str, label: str = "id") -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"invalid {label}")
    if Path(value).name != value or any(char in value for char in "/\\"):
        raise ValueError(f"invalid {label}: {value}")
    return value


def _jsonl(path: Path, values: Sequence[Mapping]) -> None:
    path.write_text("".join(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
                                 for value in values), encoding="utf-8", newline="\n")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _file_hashes(root: Path) -> dict[str, str]:
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("collection cannot contain symlinks")
    return {str(path.relative_to(root)).replace(os.sep, "/"): sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file() and path != root / "manifest.json"}


def _frame_rgba_hash(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest()


@dataclass(frozen=True)
class Collection:
    root: Path
    sources: list[dict]
    sequences: list[dict]
    manifest: dict


def reviewed_sequences(collection: Collection, review_file: Path | None = None) -> list[dict]:
    """Apply a manifest-bound human review overlay without changing input files."""
    sequences = copy.deepcopy(collection.sequences)
    if review_file is None:
        return sequences
    overlay = json.loads(Path(review_file).read_text(encoding="utf-8"))
    if not isinstance(overlay, dict) or overlay.get("collection_sha256") != sha256(collection.root / "manifest.json"):
        raise ValueError("review overlay does not match this collection manifest")
    reviews = overlay.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("review overlay needs a reviews list")
    by_id = {sequence["sequence_id"]: sequence for sequence in sequences}
    seen = set()
    for review in reviews:
        if not isinstance(review, dict) or review.get("sequence_id") not in by_id:
            raise ValueError("review references an unknown sequence")
        sequence_id = review["sequence_id"]
        if sequence_id in seen or review.get("status") not in {"accepted", "repair", "rejected", "unreviewed"}:
            raise ValueError("duplicate sequence or invalid status in review overlay")
        seen.add(sequence_id)
        by_id[sequence_id]["review_status"] = review["status"]
    return sequences


def save_frames(root: Path, sequence_id: str, images: Sequence, durations: Sequence[int]) -> list[dict]:
    """Write normalized RGBA frame masters and return frame records."""
    from PIL import Image

    sequence_id = _safe_id(sequence_id, "sequence_id")
    if len(images) != len(durations) or not images:
        raise ValueError("frames and durations must be nonempty and have equal length")
    frame_dir = relative_file(root, Path("frames") / sequence_id)
    frame_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for index, (value, duration) in enumerate(zip(images, durations)):
        if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
            raise ValueError("frame duration must be a positive integer")
        image = value.convert("RGBA") if hasattr(value, "convert") else Image.fromarray(value, "RGBA")
        path = frame_dir / f"{index:04d}.png"
        image.save(path, format="PNG", optimize=False)
        records.append({
            "path": str(path.relative_to(root)).replace(os.sep, "/"),
            "file_sha256": sha256(path),
            "rgba_sha256": _frame_rgba_hash(path),
            "duration_ms": duration,
            "size_px": [image.width, image.height],
        })
    return records


def write_collection(root: Path, sequences: Sequence[Mapping], sources: Sequence[Mapping],
                     extra: Mapping | None = None) -> dict:
    """Write JSONL records and a hash-bound manifest into an existing new folder."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    sequence_values = [dict(item) for item in sequences]
    source_values = [dict(item) for item in sources]
    for source in source_values:
        _safe_id(source.get("source_id"), "source_id")
        if not source.get("source_url"):
            raise ValueError("source_url is required for every source")
    for sequence in sequence_values:
        missing = REQUIRED_SEQUENCE_FIELDS - set(sequence)
        if missing:
            raise ValueError(f"sequence missing fields: {sorted(missing)}")
        _safe_id(sequence["sequence_id"], "sequence_id")
        if sequence["frame_count"] != len(sequence["frames"]):
            raise ValueError("frame_count does not match frames")
        if sequence["alpha_mode"] not in {"binary", "partial"}:
            raise ValueError("alpha_mode must be binary or partial")
        if sequence["review_status"] not in {"unreviewed", "accepted", "repair", "rejected"}:
            raise ValueError("invalid review_status")
        if sequence["split"] not in {"unassigned", "train", "validation", "test"}:
            raise ValueError("invalid split")
    if any((root / name).exists() for name in ("sources.jsonl", "sequences.jsonl", "manifest.json")):
        raise FileExistsError("collection records already exist")
    _jsonl(root / "sources.jsonl", source_values)
    _jsonl(root / "sequences.jsonl", sequence_values)
    holdouts = holdout_policy().hashes
    files = _file_hashes(root)
    manifest = {
        "schema": SCHEMA,
        "files": files,
        "source_count": len(source_values),
        "sequence_count": len(sequence_values),
        "holdouts": holdouts,
    }
    if extra:
        if set(extra) & {"schema", "files", "source_count", "sequence_count"}:
            raise ValueError("extra metadata cannot replace collection integrity fields")
        manifest.update(dict(extra))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8", newline="\n")
    load_collection(root)
    return manifest


def load_collection(root: Path) -> Collection:
    """Load and verify a collection manifest, records, PNG hashes and paths."""
    from PIL import Image

    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported animation collection schema")
    if manifest.get("holdouts") != FROZEN_HOLDOUT_HASHES:
        raise ValueError("Collection was not built with the frozen holdout bytes; do not edit its manifest")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("manifest files must be a mapping")
    for relative in files:
        relative_file(root, relative)
    actual = _file_hashes(root)
    if actual != files:
        raise ValueError("collection file hash mismatch")
    sources = _load_jsonl(root / "sources.jsonl")
    sequences = _load_jsonl(root / "sequences.jsonl")
    if manifest.get("source_count") != len(sources) or manifest.get("sequence_count") != len(sequences):
        raise ValueError("collection record counts differ from manifest")
    source_ids = {source.get("source_id") for source in sources}
    if len(source_ids) != len(sources):
        raise ValueError("duplicate source IDs")
    for source in sources:
        _safe_id(source.get("source_id"), "source_id")
        if not source.get("source_url") or not source.get("sha256"):
            raise ValueError("each source needs a URL and hash")
        for asset in source.get("assets", []):
            asset_path = relative_file(root, asset["path"])
            if sha256(asset_path) != asset.get("sha256") or files.get(asset["path"]) != asset["sha256"]:
                raise ValueError("source asset hash mismatch")
            if not asset.get("source_url", source["source_url"]):
                raise ValueError("source asset URL is missing")
    sequence_ids = set()
    for sequence in sequences:
        missing = REQUIRED_SEQUENCE_FIELDS - set(sequence)
        if missing:
            raise ValueError(f"sequence missing fields: {sorted(missing)}")
        sequence_id = _safe_id(sequence["sequence_id"], "sequence_id")
        if sequence_id in sequence_ids or sequence["source_id"] not in source_ids:
            raise ValueError("duplicate sequence or unknown source")
        sequence_ids.add(sequence_id)
        if not sequence["frames"] or sequence["frame_count"] != len(sequence["frames"]):
            raise ValueError("frame_count does not match nonempty frames")
        canvas = sequence["canvas_px"]
        if not isinstance(canvas, list) or len(canvas) != 2 or any(type(v) is not int or v <= 0 for v in canvas):
            raise ValueError("invalid sequence canvas")
        if type(sequence["loop"]) is not bool:
            raise ValueError("loop must be a boolean")
        anchor = sequence["anchor_px"]
        import math
        if anchor is not None and (not isinstance(anchor, list) or len(anchor) != 2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in anchor)):
            raise ValueError("invalid sequence anchor")
        if sequence["review_status"] not in {"unreviewed", "accepted", "repair", "rejected"} or sequence["split"] not in {"unassigned", "train", "validation", "test"}:
            raise ValueError("invalid sequence review status or split")
        if sequence["alpha_association"] != "straight" or sequence["blend_mode"] not in {"alpha_over", "additive"}:
            raise ValueError("unsupported alpha association or blend mode")
        expected_size = tuple(canvas)
        has_partial = False
        for frame in sequence["frames"]:
            path = relative_file(root, frame["path"])
            if not path.is_file() or sha256(path) != frame.get("file_sha256"):
                raise ValueError(f"frame file hash mismatch: {frame.get('path')}")
            with Image.open(path) as image:
                if image.size != expected_size or image.mode != "RGBA":
                    raise ValueError("frame canvas or mode mismatch")
                has_partial |= any(0 < value < 255 for value in image.getchannel("A").tobytes())
                if _frame_rgba_hash(path) != frame.get("rgba_sha256"):
                    raise ValueError("decoded frame hash mismatch")
            if type(frame.get("duration_ms")) is not int or frame["duration_ms"] <= 0:
                raise ValueError("invalid frame duration")
        if sequence["alpha_mode"] != ("partial" if has_partial else "binary"):
            raise ValueError("alpha_mode does not match actual channel values")
    return Collection(root, sources, sequences, manifest)
