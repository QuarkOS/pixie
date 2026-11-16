"""Run the six requested b80/cache comparisons separately; never save weights."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


SUITE = ("fused-adamw", "sdpa-flash", "sdpa-efficient", "tf32",
         "compile-reduce-overhead", "compile-max-autotune")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    owned = {"--out", "--experiment", "--batch", "--trials", "--worker", "--result", "--validate-only"}
    if args.out.exists() or any(token.split("=", 1)[0] in owned for token in forwarded):
        parser.error("choose a new folder; suite owns experiment, batch80, five trials and output")
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for experiment in SUITE:
        out = args.out / experiment
        command = [sys.executable, "-m", "train.pixie32.bench_compute", *forwarded,
                   "--batch", "80", "--trials", "5", "--experiment", experiment, "--out", str(out)]
        completed = subprocess.run(command, check=False)
        path = out / "benchmark.json"
        report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        rows.append({"experiment": experiment, "exit_code": completed.returncode,
                     "comparison_valid": report.get("comparison_valid", False),
                     "adoption_gate_passed": report.get("adoption_gate_passed", False),
                     "measured_throughput_change_percent": report.get("measured_throughput_change_percent")})
    result = {"scope": "six independent b80/cache five-trial comparisons; no combined-setting claim", "experiments": rows}
    (args.out / "suite.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if all(row["exit_code"] == 0 and row["comparison_valid"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
