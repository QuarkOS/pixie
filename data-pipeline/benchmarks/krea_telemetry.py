"""Record selected-GPU telemetry beside a running Krea job, or analyze streaks."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import time


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def runner_event(value, wall_origin, inflight):
    end = number(value.get("t"))
    if end is None or end < wall_origin:
        return None
    row = {"phase": "measured", "inflight": inflight, "end_s": end - wall_origin,
           "ok": "error" not in value, "source": "runner"}
    if not row["ok"]:
        return row
    generation, extraction = number(value.get("gen_s")), number(value.get("extract_s"))
    if generation is None or extraction is None or min(generation, extraction) < 0:
        raise ValueError("invalid runner timing")
    row.update(request_s=generation, extract_write_s=extraction,
               start_s=row["end_s"] - generation - extraction,
               request_end_s=row["end_s"] - extraction)
    for key in ("sprites", "usable", "purity_fail", "rejected", "edge_touch"):
        count = value.get(key)
        if type(count) is int and count >= 0:
            row[key] = count
    return row


def record(args):
    from krea_unique import GpuLog

    if not re.fullmatch(r"\d+", args.gpu) or not 0 < args.seconds <= 86400:
        raise ValueError("choose one numeric GPU and a duration up to 86400 seconds")
    if args.events is not None and not args.events.is_file():
        raise ValueError("supply an existing runner events file")
    args.out.mkdir(parents=True, exist_ok=False)
    origin, wall_origin = time.perf_counter(), time.time()
    offset = args.events.stat().st_size if args.events else 0
    gpu = GpuLog(args.out / "gpu.csv", args.gpu, origin)
    gpu.thread.start()
    count, errors = 0, 0
    try:
        with (args.out / "requests.jsonl").open("w", encoding="utf-8") as dest:
            while time.perf_counter() - origin < args.seconds:
                if args.events:
                    if args.events.stat().st_size < offset:
                        raise ValueError("runner event file was truncated")
                    with args.events.open("rb") as source:
                        source.seek(offset)
                        for line in source:
                            if not line.endswith(b"\n"):
                                break
                            offset += len(line)
                            try:
                                row = runner_event(json.loads(line), wall_origin, args.inflight)
                            except (ValueError, TypeError, AttributeError):
                                errors += 1
                                continue
                            if row is not None:
                                dest.write(json.dumps(row) + "\n")
                                count += 1
                    dest.flush()
                time.sleep(min(1., max(0., args.seconds - (time.perf_counter() - origin))))
    except KeyboardInterrupt:
        pass
    finally:
        gpu.stop.set()
        gpu.thread.join(timeout=6)
        summary = {"scope": "read-only selected-GPU monitoring and new runner events",
                   "elapsed_s": time.perf_counter() - origin, "events": count,
                   "event_errors": errors, "gpu_samples": gpu.samples, "gpu_errors": gpu.errors,
                   "gpu_index": int(args.gpu), "inflight_operator_declared": args.inflight,
                   "request_scope": "runner generation includes queueing and PNG recompression; extraction separate"}
        (args.out / "record-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if not errors and gpu.samples else 1


def describe(values):
    return {"count": len(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)} if values else None


def gpu_summary(rows, samples, fields):
    selected = [sample for sample in samples if any(
        row["start_s"] <= sample["elapsed_s"] <= row.get("request_end_s", row["end_s"])
        for row in rows)]
    return {"samples": len(selected), "fields": {field: describe([
        sample[field] for sample in selected if sample.get(field) is not None]) for field in fields}}


def analyze(args):
    if not math.isfinite(args.slow_ratio) or args.slow_ratio <= 1:
        raise ValueError("slow ratio must exceed one")
    if args.baseline_seconds is not None and (not math.isfinite(args.baseline_seconds) or args.baseline_seconds <= 0):
        raise ValueError("baseline seconds must be finite and positive")
    rows = [json.loads(line) for line in (args.run / "requests.jsonl").read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("phase") == "measured"]
    errors = sum(not row.get("ok") for row in rows)
    successes = [row for row in rows if row.get("ok")]
    for row in successes:
        if any(number(row.get(key)) is None for key in ("start_s", "end_s", "request_s", "inflight")):
            raise ValueError("recording lacks aligned request timestamps; rerun with telemetry")
        if row["request_s"] <= 0 or row["end_s"] < row["start_s"]:
            raise ValueError("invalid request timing")
    with (args.run / "gpu.csv").open(newline="") as source:
        reader = csv.DictReader(source)
        allowed = {"utilization.gpu", "memory.used", "power.draw", "temperature.gpu", "clocks.sm", "clocks.mem"}
        fields = [field for field in (reader.fieldnames or []) if field in allowed or
                  re.fullmatch(r"clocks_(?:event|throttle)_reasons\.(?:sw_power_cap|hw_slowdown|sw_thermal_slowdown|hw_thermal_slowdown|hw_power_brake_slowdown)", field)]
        samples = []
        for raw in reader:
            sample = {key: number(raw.get(key)) for key in ["elapsed_s", *fields]}
            if sample["elapsed_s"] is not None:
                samples.append(sample)
    groups = []
    for inflight in sorted({row["inflight"] for row in successes}):
        group = sorted([row for row in successes if row["inflight"] == inflight], key=lambda row: row["end_s"])
        times = sorted(row["request_s"] for row in group)
        baseline = args.baseline_seconds or times[int((len(times) - 1) * .25)]
        threshold = baseline * args.slow_ratio
        slow = [row for row in group if row["request_s"] >= threshold]
        normal = [row for row in group if row["request_s"] < threshold]
        streaks, streak = [], []
        for row in group:
            if streak and row.get("trial") != streak[-1].get("trial"):
                streaks.append(streak)
                streak = []
            if row["request_s"] >= threshold:
                streak.append(row)
            elif streak:
                streaks.append(streak)
                streak = []
        if streak:
            streaks.append(streak)
        groups.append({"inflight": inflight, "completed": len(group), "slow": len(slow),
                       "slow_fraction": len(slow) / len(group), "baseline_request_s": baseline,
                       "baseline_method": "operator supplied" if args.baseline_seconds else "estimated lower quartile",
                       "slow_threshold_s": threshold, "request_seconds": describe(times),
                       "slow_streaks": [{"requests": len(streak), "start_s": min(r["start_s"] for r in streak),
                                         "end_s": max(r["end_s"] for r in streak)} for streak in streaks],
                       "normal_gpu": gpu_summary(normal, samples, fields),
                       "slow_gpu": gpu_summary(slow, samples, fields)})
    report = {"scope": "queue-inclusive request-latency classification, grouped by declared concurrency",
              "completed": len(successes), "errors": errors, "slow_ratio": args.slow_ratio,
              "groups": groups, "gpu_samples": len(samples),
              "limits": ["Latency is not wall seconds per sheet under concurrency.",
                         "Overlapping normal/slow requests can share GPU samples.",
                         "Correlated clocks, power or reasons do not prove a cause.",
                         "A run dominated by slow requests needs a controlled baseline.",
                         "Two-second telemetry can miss short events; CPU and offload causes remain unmeasured."]}
    with args.out.open("x", encoding="utf-8") as dest:
        dest.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if successes and not errors else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("record")
    collect.add_argument("--gpu", required=True)
    collect.add_argument("--events", type=Path, help="existing runner events.jsonl; copy new numeric events only")
    collect.add_argument("--inflight", type=int, choices=(1, 2, 3), required=True)
    collect.add_argument("--seconds", type=float, default=1800)
    collect.add_argument("--out", type=Path, required=True)
    inspect = commands.add_parser("analyze")
    inspect.add_argument("--run", type=Path, required=True)
    inspect.add_argument("--out", type=Path, required=True)
    inspect.add_argument("--slow-ratio", type=float, default=1.75)
    inspect.add_argument("--baseline-seconds", type=float)
    args = parser.parse_args()
    return record(args) if args.command == "record" else analyze(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"telemetry failed: {type(error).__name__}")
        raise SystemExit(1)
