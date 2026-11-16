"""Compare scored animation sheets against a frozen pilot baseline."""
from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from pathlib import Path


def normalize_holdout(value: str) -> str:
    return " ".join("".join(char if char.isalnum() else " " for char in
                              unicodedata.normalize("NFKC", value).casefold()).split())

CANONICAL_HOLDS = {
    "blind50": "7524db9cabd2449f5ec6b48b454bc10e2fed38825ba8c8d1e749311f3b5afb0a",
    "eval50": "a1d6893e8c3c45ab99a845c1726c55547dcde29ff6f7a94b022f067c27a7872f",
}


def _rows(score_dir: Path) -> list[dict]:
    path = score_dir / "score.jsonl"
    if not path.is_file():
        raise ValueError(f"missing score file: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("score rows must be objects")
            rows.append(row)
    if not rows:
        raise ValueError("score file is empty")
    keys = [(normalize_holdout(str(row.get("subject", ""))), str(row.get("action", ""))) for row in rows]
    if any(not key[0] or not key[1] for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("score rows need unique nonempty subject/action keys")
    return rows


def _summary(rows: list[dict]) -> dict:
    structural = [row for row in rows if row.get("frame_count") == 4 and row.get("status") == "UNCALIBRATED"]
    moving = [row for row in structural if not row.get("motion", {}).get("all_frames_identical", True)]
    jitters = [row["geometry"]["bbox_jitter_px"] for row in structural
               if isinstance(row.get("geometry", {}).get("bbox_jitter_px"), (int, float))]
    loops = [row["loop"]["rgba_mean_abs_error"] for row in structural
             if row.get("loop", {}).get("applicable") and isinstance(row["loop"].get("rgba_mean_abs_error"), (int, float))]
    identities = [row["identity"]["embedding_similarity"]["from_first_mean"] for row in structural
                  if row.get("identity", {}).get("status") == "available"
                  and isinstance(row.get("identity", {}).get("embedding_similarity", {}).get("from_first_mean"), (int, float))]
    return {
        "scored": len(rows), "structurally_valid": len(structural),
        "structural_valid_fraction": len(structural) / len(rows),
        "moving_fraction": len(moving) / len(structural) if structural else 0.0,
        "bbox_jitter_mean_px": statistics.fmean(jitters) if jitters else None,
        "loop_error_mean": statistics.fmean(loops) if loops else None,
        "identity_from_first_mean": statistics.fmean(identities) if identities else None,
        "identity_available": len(identities),
        "metric_note": "uncalibrated scorer values; thresholds require human labels",
    }


def compare(baseline: Path, candidate: Path, candidate_sheets: Path, output: Path,
            *, allow_below_target=False, blind50=None, eval50=None):
    if output.exists():
        raise FileExistsError("output already exists; choose a new folder")
    manifest = json.loads((candidate_sheets / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("holdouts") != CANONICAL_HOLDS:
        raise ValueError("candidate sheet manifest does not use canonical holdout bytes")
    records = manifest.get("records", [])
    train_artist = sum(row.get("split") == "train" and row.get("provenance", {}).get("kind") in {"artist", "real"}
                       for row in records)
    if train_artist < 400 and not allow_below_target:
        raise ValueError(f"candidate has {train_artist} artist training sheets; need at least 400")
    baseline_rows, candidate_rows = _rows(baseline), _rows(candidate)
    base_keys = {(normalize_holdout(str(row["subject"])), str(row["action"])) for row in baseline_rows}
    cand_keys = {(normalize_holdout(str(row["subject"])), str(row["action"])) for row in candidate_rows}
    if base_keys != cand_keys:
        raise ValueError("baseline and candidate must use the same subject/action keys")
    holds = []
    for path in (blind50, eval50):
        if path is not None:
            values = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(values, list) or len(values) != 50:
                raise ValueError("evaluation holdout must contain exactly 50 subjects")
            holds.extend(normalize_holdout(value) for value in values)
    if any(any(hold and hold in normalize_holdout(str(row.get("subject", ""))) for hold in holds)
           for row in candidate_rows):
        raise ValueError("candidate evaluation contains a frozen holdout subject")
    base_summary, cand_summary = _summary(baseline_rows), _summary(candidate_rows)
    deltas = {}
    for key in ("structural_valid_fraction", "moving_fraction", "bbox_jitter_mean_px", "loop_error_mean", "identity_from_first_mean"):
        left, right = base_summary[key], cand_summary[key]
        deltas[key] = None if left is None or right is None else right - left
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": "pixie-animation-lora-eval-v1", "status": "UNCALIBRATED",
                                  "candidate_artist_train_sheets": train_artist,
                                  "baseline": base_summary, "candidate": cand_summary,
                                  "candidate_minus_baseline": deltas,
                                  "same_keys": len(base_keys)}, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8", newline="\n")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sheets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blind50", type=Path)
    parser.add_argument("--eval50", type=Path)
    parser.add_argument("--allow-below-target", action="store_true", help="CPU fixture mode only")
    args = parser.parse_args(argv)
    print(compare(args.baseline, args.candidate, args.candidate_sheets, args.output,
                  allow_below_target=args.allow_below_target, blind50=args.blind50, eval50=args.eval50))


if __name__ == "__main__":
    main()
