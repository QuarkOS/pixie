"""Read one completed segment summary and project reporting costs; never train."""
import argparse
import json
import math
from pathlib import Path


PHASES = ("training", "evaluation", "sampling", "checkpoint")


def number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} has an invalid value")
    return value


def report(summary, *, remaining_images, eval_every, sample_every, checkpoint_cycle_seconds=None,
           save_every=100_000, batch=80, steady_update_seconds=None):
    """Event means include finalization; projections are explicitly estimates."""
    if summary.get("status") not in ("complete", "smoke", "interrupt", "max_hours"):
        raise ValueError("require a completed trainer segment summary")
    images = number(summary["segment_images"], "segment_images", positive=True)
    wall = number(summary["segment_wall_seconds"], "segment_wall_seconds", positive=True)
    seconds = {p: number(summary["phase_seconds"][p], p) for p in PHASES}
    counts = {p: summary["phase_counts"][p] for p in PHASES}
    for phase, count in counts.items():
        if type(count) is not int or count < 0 or (seconds[phase] > 0 and count == 0):
            raise ValueError("phase counts must be nonnegative integers consistent with times")
    if sum(seconds.values()) > wall + max(1e-6, wall * 1e-8):
        raise ValueError("phase times exceed segment wall time")
    end = number(summary["images_seen"], "images_seen")
    if end < images:
        raise ValueError("global images cannot be smaller than segment images")
    if type(remaining_images) is not int or remaining_images <= 0:
        raise ValueError("remaining images must be a positive integer")
    if type(batch) is not int or batch <= 0:
        raise ValueError("batch must be a positive integer")
    for interval in (eval_every, sample_every, save_every):
        if type(interval) is not int or interval < batch:
            raise ValueError("cadence must be an integer at least as large as batch")
    if checkpoint_cycle_seconds is not None:
        number(checkpoint_cycle_seconds, "checkpoint cycle seconds", positive=True)
    if steady_update_seconds is not None:
        number(steady_update_seconds, "steady update seconds", positive=True)
    updates = math.ceil(remaining_images / batch)
    # Existing trainer aligns branch cadences to global image multiples. Final
    # eval/grid are mandatory unless a periodic event already covered that step.
    def events(interval):
        total = int((end + remaining_images) // interval - end // interval)
        final_start = end + ((remaining_images - 1) // batch) * batch
        final_has_periodic = (end + remaining_images) // interval > final_start // interval
        return total + int(not final_has_periodic)
    estimates = {}
    for phase, interval in (("evaluation", eval_every), ("sampling", sample_every)):
        count = events(interval)
        estimates[phase] = {"events_including_final": count,
                            "seconds": seconds[phase] / counts[phase] * count if counts[phase] else None}
    cycles = int((end + remaining_images) // save_every - end // save_every)
    estimates["checkpoint"] = {
        "periodic_cycles": cycles,
        "periodic_seconds": cycles * checkpoint_cycle_seconds if checkpoint_cycle_seconds is not None else None,
        "note": "One cycle writes a model snapshot and full last.pt. Final and best.pt writes are additional and unprojected."}
    return {
        "measured_segment": {
            "images": images, "wall_seconds": wall, "phase_seconds": seconds, "phase_counts": counts,
            "wall_share": {p: seconds[p] / wall for p in PHASES},
            "reporting_wall_share": sum(seconds[p] for p in PHASES[1:]) / wall,
            "unclassified_seconds": wall - sum(seconds.values()),
            "scope": summary.get("wall_scope", "trainer segment; asset loading excluded")},
        "estimated_remaining": {
            "images": remaining_images,
            "updates": updates,
            "training_seconds": (steady_update_seconds * updates if steady_update_seconds is not None
                                 else seconds["training"] / images * remaining_images),
            "training_time_basis": ("operator-supplied steady seconds per completed update; partial final update charged as full"
                                    if steady_update_seconds is not None else "measured segment mean, including compilation"),
            "steady_update_seconds": steady_update_seconds,
            "batch": batch,
            "evaluation_every_images": eval_every, "sampling_every_images": sample_every,
            "save_every_images": save_every, "phases": estimates,
            "scope": "Training uses the stated rate basis; reporting uses segment event means. Projections are estimates, not measured long-run performance. No total wall-time forecast without checkpoint and startup costs."},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remaining-images", type=int, required=True)
    parser.add_argument("--batch", type=int, default=80, help="batch size of the projected continuation")
    parser.add_argument("--eval-every-images", type=int, default=250_000)
    parser.add_argument("--sample-every-images", type=int, default=500_000)
    parser.add_argument("--save-every-images", type=int, default=100_000)
    parser.add_argument("--checkpoint-cycle-seconds", type=float)
    parser.add_argument("--steady-update-seconds", type=float, help="measured steady update time, e.g. 0.221 for the accepted b80 run; do not reuse at another batch")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("choose a new output folder")
    try:
        result = report(json.loads(args.summary.read_text(encoding="utf-8")),
                        remaining_images=args.remaining_images, eval_every=args.eval_every_images,
                        sample_every=args.sample_every_images, save_every=args.save_every_images,
                        checkpoint_cycle_seconds=args.checkpoint_cycle_seconds, batch=args.batch,
                        steady_update_seconds=args.steady_update_seconds)
    except (OSError, KeyError, ValueError, TypeError) as error:
        parser.error(f"invalid segment or projection inputs ({type(error).__name__})")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "cadence.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
