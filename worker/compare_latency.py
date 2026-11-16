"""Compare matched offline GPU reports; never infer browser or queue latency."""
import argparse
import json
import math
from pathlib import Path
import statistics

from latency_identity import PROTOCOL, json_hash


PHASES = {"prepare", "clip", "sample", "readback_rgba", "score_rank", "png"}


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("timings must be finite and positive")
    return value


def metric(value):
    middle, low, high = (positive(value[key]) for key in ("median", "min", "max"))
    if not low <= middle <= high:
        raise ValueError("timing range does not contain median")
    return {"median": middle, "min": low, "max": high}


def validate(report):
    if not isinstance(report, dict):
        raise ValueError("source report must be an object")
    if report.get("comparison_valid") is not True:
        raise ValueError("source report failed its local exact-output gate")
    identity = report["workload_identity"]
    required = {"protocol", "ordered_prompt_sha256", "seed_policy", "batch", "show", "stored_candidates",
                "trials", "warmup_calls", "checkpoint_sha256", "clip_weights_sha256", "tokenizer_sha256",
                "palette_state_sha256", "implementation_sha256", "loaded_model_implementation_sha256",
                "loaded_runtime_implementation_sha256",
                "sampling_settings", "png_scale", "png_mode", "reward", "runtime", "math"}
    if set(identity) != required or identity["protocol"] != PROTOCOL:
        raise ValueError("missing or unsupported workload identity")
    if not identity["ordered_prompt_sha256"] or not identity["reward"] or not identity["sampling_settings"] or not identity["runtime"] or not identity["math"]:
        raise ValueError("workload identity is incomplete")
    if report["workload_sha256"] != json_hash(identity):
        raise ValueError("workload identity hash mismatch")
    for field in ("batch", "show", "stored_candidates", "trials", "warmup_calls"):
        if type(identity[field]) is not int or identity[field] <= 0 or report[field] != identity[field]:
            raise ValueError("workload budget mismatch")
    if not 1 <= identity["show"] <= identity["stored_candidates"] <= identity["batch"]:
        raise ValueError("invalid candidate budget")
    if identity["trials"] < 5:
        raise ValueError("require at least five paired trials")
    expected_rows = identity["trials"] * len(identity["ordered_prompt_sha256"])
    if len(report["rows"]) != expected_rows:
        raise ValueError("incomplete trials")
    phases = report["phase_ms"]
    if set(phases) != PHASES:
        raise ValueError("require the complete generation phase set")
    seen = set()
    for row in report["rows"]:
        point = (row["trial"], row["prompt_index"])
        if point in seen or not 0 <= point[0] < identity["trials"] or not 0 <= point[1] < len(identity["ordered_prompt_sha256"]):
            raise ValueError("invalid trial indices")
        seen.add(point)
        if row["exact_outputs"] is not True:
            raise ValueError("source output mismatch")
        positive(row["unprofiled_ms"])
        positive(row["profiled_ms"])
        if set(row["profile"]["phase_ms"]) != set(phases):
            raise ValueError("phase sets differ inside source report")
        for value in row["profile"]["phase_ms"].values():
            positive(value)
        total = positive(row["profile"]["total_ms"])
        if total > row["profiled_ms"] + 1e-6:
            raise ValueError("profile scope exceeds complete instrumented generation")
        residual = row["profile"]["unclassified_ms"]
        if isinstance(residual, bool) or not isinstance(residual, (int, float)) or not math.isfinite(residual) or residual < -1e-6:
            raise ValueError("invalid unclassified profile interval")
        if not math.isclose(sum(row["profile"]["phase_ms"].values()) + residual, total, rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError("exclusive phase times do not account for profile total")
    def checked_summary(summary, values):
        validated = metric(summary)
        actual = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        if validated != actual:
            raise ValueError("summary differs from completed raw trials")
        return actual
    return {"clean": checked_summary(report["unprofiled_ms"], [row["unprofiled_ms"] for row in report["rows"]]),
            "profiled": checked_summary(report["profiled_ms"], [row["profiled_ms"] for row in report["rows"]]),
            "phases": {key: checked_summary(value, [row["profile"]["phase_ms"][key] for row in report["rows"]])
                       for key, value in phases.items()}}


def compare(left, right):
    try:
        values = [validate(value) for value in (left, right)]
        if left["workload_identity"] != right["workload_identity"]:
            return {"comparison_valid": False, "verdict": "inconclusive", "reason": "workload or runtime identities differ"}
        if values[0]["phases"].keys() != values[1]["phases"].keys():
            raise ValueError("phase sets differ")
        def pair(a, b):
            return {"reference_ms": a, "candidate_ms": b,
                    "candidate_over_reference_median": b["median"] / a["median"],
                    "ranges_overlap": max(a["min"], b["min"]) <= min(a["max"], b["max"])}
        clean = pair(values[0]["clean"], values[1]["clean"])
        return {"comparison_valid": True, "verdict": "matched offline measurements",
                "scope": "complete offline generation on each GPU; excludes queue, polling, JSON serialization and HTTP; no browser latency or cross-GPU exact-sprite claim",
                "reference_gpu": left["gpu_model"], "candidate_gpu": right["gpu_model"],
                "workload_sha256": left["workload_sha256"], "clean_total": clean,
                "profiled_total": pair(values[0]["profiled"], values[1]["profiled"]),
                "phase_ratios": {key: pair(values[0]["phases"][key], values[1]["phases"][key]) for key in values[0]["phases"]},
                "interpretation": "ratios are measured median ratios, not causal attribution; overlapping ranges make a small difference inconclusive; never sum phase medians"}
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"comparison_valid": False, "verdict": "inconclusive", "reason": "missing, malformed, nonfinite or incomplete source evidence"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("choose a new output folder")
    try:
        result = compare(json.loads(args.reference.read_text(encoding="utf-8")), json.loads(args.candidate.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        result = {"comparison_valid": False, "verdict": "inconclusive", "reason": "cannot read source reports"}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "latency-comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))
    return 0 if result["comparison_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
