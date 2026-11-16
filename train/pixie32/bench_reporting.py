"""Measure training and reporting phases in repeated fresh-folder smoke segments."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from train.pixie32.clip_cache import idle_device
from train.pixie32 import train_p32


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    supplied = args.trainer_args[1:] if args.trainer_args[:1] == ["--"] else args.trainer_args
    if any(token.split("=", 1)[0] in ("--out", "--smoke", "--smoke-steps") for token in supplied):
        parser.error("benchmark owns output and smoke settings")
    if args.out.exists() or not 1 <= args.steps <= 200 or not 5 <= args.trials <= 10:
        parser.error("select a new output folder, 1..200 steps and 5..10 trials")
    config = train_p32.parser().parse_args(supplied + ["--out", str(args.out), "--smoke", "--smoke-steps", str(args.steps)])
    if config.resume is None:
        parser.error("supply --resume and its unchanged recipe after --")
    idle_device(config.device)
    args.out.mkdir(parents=True, exist_ok=False)
    trials, errors = [], []
    for index in range(args.trials):
        out = args.out / f"trial-{index:02d}"
        command = [sys.executable, "-m", "train.pixie32.train_p32", *supplied,
                   "--out", str(out), "--smoke", "--smoke-steps", str(args.steps)]
        try:
            idle_device(config.device)
            subprocess.run(command, capture_output=True, check=True)
            summary = out / "paused.json"
            if not summary.is_file():
                raise ValueError("smoke did not pause")
            measured = json.loads(summary.read_text(encoding="utf-8"))
            if measured["status"] != "smoke" or measured["measured_steps"] != args.steps:
                raise ValueError("incomplete smoke workload")
            row = {key: measured[key] for key in (
                "measured_steps", "segment_images", "training_seconds", "segment_wall_seconds",
                "images_per_second", "wall_images_per_second", "phase_seconds", "phase_counts",
                "unclassified_seconds")}
            row["trial"] = index
            trials.append(row)
            print(json.dumps(row), flush=True)
        except (OSError, subprocess.SubprocessError, KeyError, ValueError) as error:
            errors.append({"trial": index, "error_type": type(error).__name__})
            break
    summary = {}
    for key in ("images_per_second", "wall_images_per_second", "segment_wall_seconds"):
        values = [row[key] for row in trials]
        summary[key] = {"median": statistics.median(values), "range": [min(values), max(values)]} if values else None
    report = {"scope": "fresh-process smoke segments; segment clock excludes startup and includes compile/reporting/checkpoints",
              "warmup": "none excluded; this measures complete bounded segments, not steady-state step speed",
              "trials": trials, "errors": errors, "summary": summary,
              "comparison_valid": len(trials) == args.trials and not errors}
    (args.out / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if report["comparison_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
