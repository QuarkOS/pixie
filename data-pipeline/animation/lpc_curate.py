"""Validate a downloaded LPC curation manifest and ingest its sequences."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import DEFAULT_BLIND50, DEFAULT_EVAL50
from .ingest import ingest, public_url

# Official LPC fish sheet (lpc_fish.png): rows are up, left, right, down.
LPC_DIRECTION_ROWS = {"up": 0, "left": 1, "right": 2, "down": 3}
_DIRECTION_ALIASES = {"north": "up", "south": "down", "east": "right", "west": "left"}
# Official LPC horse sheet: four directional rows per motion. Group 0 is gallop, group 1 is walk.
LPC_HORSE_MOTION_GROUPS = {0: "gallop", 1: "walk"}
LPC_HORSE_ROWS_PER_GROUP = 4


def _sha256_text(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("archive_sha256 must be lowercase hexadecimal SHA256")
    return value


def _validate_pack_conventions(pack_id: str, row: dict) -> None:
    direction = _DIRECTION_ALIASES.get(row.get("direction"), row.get("direction"))
    if "fish" in pack_id and row.get("format") == "sheet":
        order = row.get("order")
        expected = LPC_DIRECTION_ROWS.get(direction)
        valid_order = isinstance(order, list) and bool(order) and all(
            isinstance(point, list) and len(point) == 2 and all(type(value) is int for value in point)
            for point in order)
        if expected is None or not valid_order or any(point[1] != expected for point in order):
            raise ValueError("LPC fish sheets use rows up, left, right, down; correct the direction/order pair")
    if "horse" in pack_id:
        anchor = row.get("anchor_px")
        if row.get("anchor_policy") != "source_ground_center":
            raise ValueError("LPC horse rows need anchor_policy source_ground_center")
        if not isinstance(anchor, list) or len(anchor) != 2 or any(type(value) is not int for value in anchor):
            raise ValueError("LPC horse anchors must be integer source-canvas coordinates")
        if row.get("source_motion") != row.get("action"):
            raise ValueError("LPC horse action must match source_motion; label side gallop rows as gallop")
        if row.get("format") == "sheet":
            _validate_horse_sheet_group(row)


def _validate_horse_sheet_group(row: dict) -> None:
    order = row.get("order")
    valid_order = isinstance(order, list) and bool(order) and all(
        isinstance(point, list) and len(point) == 2 and all(type(value) is int and value >= 0 for value in point)
        for point in order)
    if not valid_order:
        raise ValueError("LPC horse sheets need an explicit [column, row] order")
    groups = {point[1] // LPC_HORSE_ROWS_PER_GROUP for point in order}
    if len(groups) != 1:
        raise ValueError("LPC horse frames must stay inside one motion group")
    expected = LPC_HORSE_MOTION_GROUPS.get(next(iter(groups)))
    if expected is not None and row.get("action") != expected:
        raise ValueError("LPC horse sheet group 0 is gallop and group 1 is walk")


def load_curation(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("LPC curation must be an object")
    pack_id = data.get("pack_id")
    if not isinstance(pack_id, str) or not pack_id or any(char in pack_id for char in "/\\"):
        raise ValueError("pack_id must be a nonempty portable identifier")
    source_url = public_url(data.get("source_url", ""))
    archive_sha256 = _sha256_text(data.get("archive_sha256"))
    rows = data.get("sequences")
    if not isinstance(rows, list) or not rows:
        raise ValueError("curation must list at least one sequence")
    seen = set()
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each LPC sequence must be an object")
        sequence_id = row.get("sequence_id")
        if not isinstance(sequence_id, str) or not sequence_id or sequence_id in seen:
            raise ValueError("sequence IDs must be unique nonempty strings")
        seen.add(sequence_id)
        if row.get("kind", "artist") != "artist":
            raise ValueError("LPC curation accepts artist sequences only")
        _validate_pack_conventions(pack_id, row)
        entry = dict(row)
        entry["source_url"] = source_url
        entry["provenance"] = {
            **(entry.get("provenance") or {}),
            "lpc_pack_id": pack_id,
            "archive_sha256": archive_sha256,
            "curation_schema": "pixie-lpc-curation-v1",
        }
        normalized.append(entry)
    return {"pack_id": pack_id, "source_url": source_url,
            "archive_sha256": archive_sha256, "sequences": normalized}


def curate(input_root: Path, curation_path: Path, output: Path, *, canvas=(32, 32),
           blind50=DEFAULT_BLIND50, eval50=DEFAULT_EVAL50, default_duration_ms=None,
           alpha_threshold=None):
    input_root = input_root.resolve()
    curation_path = curation_path.resolve()
    if not input_root.is_dir():
        raise ValueError("LPC input must be a downloaded/extracted directory")
    if output.exists() or output.resolve() == input_root or input_root in output.resolve().parents:
        raise ValueError("output must be a new folder outside the downloaded pack")
    manifest = load_curation(curation_path)
    curation_hash = hashlib.sha256(curation_path.read_bytes()).hexdigest()
    sequences = []
    for row in manifest["sequences"]:
        item = dict(row)
        item["provenance"] = {**item.get("provenance", {}), "curation_sha256": curation_hash}
        sequences.append(item)
    spec = {"sequences": sequences}
    spec_dir = output.parent / (output.name + ".ingest-spec")
    spec_path = spec_dir / "ingest.json"
    if spec_dir.exists():
        raise FileExistsError("refusing to overwrite the generated ingest spec")
    spec_dir.mkdir(parents=True)
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    try:
        root = ingest(input_root, spec_path, output, canvas, blind50, eval50,
                      default_duration_ms=default_duration_ms, alpha_threshold=alpha_threshold)
    except Exception:
        if spec_dir.exists():
            import shutil
            shutil.rmtree(spec_dir)
        raise
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--curation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canvas", type=int, nargs=2, default=[32, 32])
    parser.add_argument("--default-duration-ms", type=int)
    parser.add_argument("--alpha-threshold", type=int,
                        help="Opt-in alpha cutout for soft-edged LPC sprites (1-255)")
    parser.add_argument("--blind50", type=Path, default=DEFAULT_BLIND50)
    parser.add_argument("--eval50", type=Path, default=DEFAULT_EVAL50)
    args = parser.parse_args(argv)
    root = curate(args.input, args.curation, args.output, canvas=tuple(args.canvas),
                  blind50=args.blind50, eval50=args.eval50,
                  default_duration_ms=args.default_duration_ms,
                  alpha_threshold=args.alpha_threshold)
    count = sum(1 for line in (root / "sequences.jsonl").read_text(encoding="utf-8").splitlines() if line)
    print(json.dumps({"status": "created", "output": str(root), "sequences": count}))


if __name__ == "__main__":
    main()
