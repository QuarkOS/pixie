"""Isolated single-lever P32 compute comparisons, with no training-run writes."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
from dataclasses import asdict, dataclass, replace
import gc
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from train.pixie32.clip_cache import FeatureCache, idle_device
from train.pixie32.data import DEFAULT_BLIND50, DEFAULT_EVAL50, load_dataset, sha256
from train.pixie32.p32_model import P32, P32Config
from train.pixie32.train_p32 import Recipe, autocast, epoch_plan, load_clip, prepared_batch


@dataclass(frozen=True)
class Compute:
    compile_mode: str = "default"
    sdpa: str = "auto"
    fused_adamw: bool = False
    tf32: bool = False


EXPERIMENTS = {
    "fused-adamw": {"fused_adamw": True},
    "sdpa-flash": {"sdpa": "flash"},
    "sdpa-efficient": {"sdpa": "efficient"},
    "sdpa-math": {"sdpa": "math"},
    "compile-reduce-overhead": {"compile_mode": "reduce-overhead"},
    "compile-max-autotune": {"compile_mode": "max-autotune"},
    "compile-max-autotune-no-cudagraphs": {"compile_mode": "max-autotune-no-cudagraphs"},
    "tf32": {"tf32": True},
}


def variants(experiment):
    baseline = Compute()
    return baseline, replace(baseline, **EXPERIMENTS[experiment])


def attention(setting):
    backends = {"flash": SDPBackend.FLASH_ATTENTION, "efficient": SDPBackend.EFFICIENT_ATTENTION,
                "math": SDPBackend.MATH}
    return nullcontext() if setting.sdpa == "auto" else sdpa_kernel(backends[setting.sdpa])


def configure(model, setting, device):
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = setting.tf32
    return torch.compile(model.loss_fwd, mode=setting.compile_mode) if device.type == "cuda" else model.loss_fwd


def optimizer_for(model, saved, setting):
    optimizer = torch.optim.AdamW(model.parameters(), fused=True if setting.fused_adamw else None)
    state = copy.deepcopy(saved["optimizer"])
    for group in state["param_groups"]:
        group["fused"] = True if setting.fused_adamw else None
        group["foreach"] = None
    optimizer.load_state_dict(state)
    return optimizer


def differences(reference, actual, *, atol, rtol, absolute_gate=True):
    maximum, error2, reference2, valid = 0., 0., 0., True
    for name, expected in reference.items():
        value = actual[name].detach().float().cpu()
        expected = expected.float()
        delta = value - expected
        maximum = max(maximum, float(delta.abs().max()))
        error2 += float(delta.double().square().sum())
        reference2 += float(expected.double().square().sum())
        finite = bool(torch.isfinite(value).all() and torch.isfinite(expected).all())
        valid &= finite and (not absolute_gate or bool(torch.allclose(value, expected, atol=atol, rtol=rtol)))
    relative_l2 = (error2 / max(reference2, 1e-30)) ** .5
    return {"valid": valid and math.isfinite(relative_l2) and relative_l2 <= rtol,
            "max_abs_error": maximum if math.isfinite(maximum) else None,
            "relative_l2_error": relative_l2 if math.isfinite(relative_l2) else None,
            "atol": atol if absolute_gate else None, "rtol": rtol,
            "gate": "elementwise allclose and global relative L2" if absolute_gate else "finite and global relative L2; max_abs diagnostic only"}


def parity(model, saved, settings, device, inputs, chunks, alpha_weight):
    reference = None
    reports = []
    for setting in settings:
        model.load_state_dict(saved["model"], strict=True)
        optimizer = optimizer_for(model, saved, setting)
        model.eval()
        loss_fn = configure(model, setting, device)
        optimizer.zero_grad(set_to_none=True)
        with attention(setting), autocast(device):
            loss, nll = loss_fn(*inputs[:2], alpha_weight, cond=inputs[2], chunks=chunks)
        loss.backward()
        values = {name: p.grad.detach().float().cpu().clone() for name, p in model.named_parameters()}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses = torch.stack((loss.detach(), nll.detach())).float().cpu()
        state = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
        if reference is None:
            reference = losses.clone(), values, state
        else:
            loss_report = differences({"loss_and_nll": reference[0]}, {"loss_and_nll": losses}, atol=2e-3, rtol=2e-3)
            gradient_report = differences(reference[1], values, atol=2e-4, rtol=2e-2)
            update_report = differences(reference[2], state, atol=0., rtol=1e-5, absolute_gate=False)
            reports.append({"valid": all(x["valid"] for x in (loss_report, gradient_report, update_report)),
                            "loss": loss_report, "gradients": gradient_report, "updated_parameters": update_report,
                            "baseline_loss_nll": reference[0].tolist(), "candidate_loss_nll": losses.tolist()})
        del optimizer, values, state, loss_fn, loss, nll
        model.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == "cuda":
            torch._dynamo.reset()
            torch.cuda.empty_cache()
    return {"valid": all(row["valid"] for row in reports), "comparisons": reports,
            "scope": "one full-batch gradient and AdamW update from saved weights/moments; dropout disabled on both sides"}


def curve_comparison(reference, candidate, *, steps, block=20, atol=1e-3, rtol=1e-3, noise_sigma=3.):
    """Compare paired loss block means with a declared sampling-noise allowance."""
    left, right = np.asarray(reference["training_losses"]), np.asarray(candidate["training_losses"])
    complete = (reference.get("completed_updates") == candidate.get("completed_updates") == steps
                and len(left) == len(right) == steps and np.isfinite(left).all() and np.isfinite(right).all())
    blocks = []
    if complete:
        for start in range(0, steps, block):
            a, b = left[start:start + block], right[start:start + block]
            differences = b - a
            error = abs(float(differences.mean()))
            standard_error = float(differences.std(ddof=1) / math.sqrt(len(differences))) if len(differences) > 1 else 0.
            floor = atol + rtol * abs(float(a.mean()))
            tolerance = floor + noise_sigma * standard_error
            blocks.append({"first_update": start + 1, "updates": len(a), "baseline_mean": float(a.mean()),
                           "candidate_mean": float(b.mean()), "mean_abs_difference": error,
                           "paired_standard_error": standard_error, "tolerance": tolerance, "valid": error <= tolerance})
    evaluations = []
    left_eval, right_eval = reference.get("evaluation_nll", []), candidate.get("evaluation_nll", [])
    if len(left_eval) == len(right_eval) == math.ceil(steps / block) + 1:
        for a, b in zip(left_eval, right_eval):
            same_step = a["update"] == b["update"]
            keys = set(a["nll"])
            valid = same_step and keys == set(b["nll"]) and bool(keys)
            errors = {}
            for key in keys & set(b["nll"]):
                x, y = a["nll"][key], b["nll"][key]
                finite = math.isfinite(x) and math.isfinite(y)
                error = abs(y - x) if finite else None
                tolerance = 1e-4 + 1e-4 * abs(x) if finite else None
                errors[key] = {"absolute_error": error, "tolerance": tolerance}
                valid &= finite and error <= tolerance
            evaluations.append({"update": a["update"], "valid": bool(valid), "errors": errors})
    valid = (complete and bool(blocks) and all(row["valid"] for row in blocks)
             and len(evaluations) == math.ceil(steps / block) + 1 and all(row["valid"] for row in evaluations))
    return {"valid": bool(valid), "completed_updates_per_side": steps if complete else None,
            "loss_blocks": blocks, "evaluation_nll": evaluations,
            "loss_atol": atol, "loss_rtol": rtol, "noise_sigma": noise_sigma,
            "evaluation_atol": 1e-4, "evaluation_rtol": 1e-4,
            "scope": "paired short training trajectories with original dropout; deterministic fixed IID/probe batch NLL; noise allowance is a heuristic, not a long-run quality guarantee"}


def run_curve(model, saved, setting, device, batch, evaluation_batch, a, alpha_weight):
    model.load_state_dict(saved["model"], strict=True)
    torch.random.default_generator.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)
    optimizer = optimizer_for(model, saved, setting)
    loss_fn = configure(model, setting, device)
    losses, evaluations = [], []
    def observe(update):
        model.eval()
        nlls = {}
        with torch.no_grad(), attention(setting), autocast(device):
            for name, inputs in evaluation_batch().items():
                _, nll = model.loss_fwd(*inputs[:2], alpha_weight, cond=inputs[2], chunks=a.ce_chunks)
                nlls[name] = float(nll)
        if not nlls or not all(math.isfinite(value) for value in nlls.values()):
            raise ValueError("missing or non-finite evaluation NLL")
        evaluations.append({"update": update, "nll": nlls})
        model.train()
    observe(0)
    with attention(setting):
        for step in range(a.validate_steps):
            if device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            features, image, cond = batch(step)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                loss, _ = loss_fn(features, image, alpha_weight, cond=cond, chunks=a.ce_chunks)
            if not torch.isfinite(loss):
                raise ValueError("non-finite validation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if (step + 1) % 20 == 0 or step + 1 == a.validate_steps:
                observe(step + 1)
    return {"valid": True, "completed_updates": len(losses), "training_losses": losses,
            "evaluation_nll": evaluations}


def run_updates(model, saved, setting, device, batch, a, alpha_weight):
    model.load_state_dict(saved["model"], strict=True)
    model.train()
    torch.random.default_generator.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)
    optimizer = optimizer_for(model, saved, setting)
    loss_fn = configure(model, setting, device)
    losses, update_ms = [], []
    warmup_started = time.perf_counter()
    with attention(setting):
        for step in range(a.warmup + a.steps):
            if step == a.warmup:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                began = time.perf_counter()
                warmup_seconds = began - warmup_started
            started = time.perf_counter()
            if device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            features, image, cond = batch(step)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                loss, _ = loss_fn(features, image, alpha_weight, cond=cond, chunks=a.ce_chunks)
            if not torch.isfinite(loss):
                raise ValueError("non-finite benchmark loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if step >= a.warmup:
                losses.append(float(loss.detach()))
                update_ms.append((time.perf_counter() - started) * 1000)
    seconds = time.perf_counter() - began
    return {"valid": True, "completed_updates": a.steps, "completed_images": a.steps * a.batch,
            "seconds": seconds, "images_per_second": a.steps * a.batch / seconds,
            "warmup_seconds_including_compile": warmup_seconds,
            "median_update_ms": statistics.median(update_ms), "training_losses": losses,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}


def child(a):
    device = idle_device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = load_dataset(a.data, a.blind50, a.eval50)
    cache = FeatureCache(a.clip_cache, data, sha256(a.clip), a.clip_cache_max_gb)
    if cache.compute != ("bf16" if device.type == "cuda" else "float32"):
        raise ValueError("CLIP cache precision differs from selected device")
    checkpoint_sha = sha256(a.checkpoint)
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") != "pixie32-full-v1" or "optimizer" not in saved:
        raise ValueError("a full P32 checkpoint with AdamW state is required")
    model = P32(P32Config(**saved["config"])).to(device)
    model.gradient_checkpointing = False
    clip = load_clip(a.clip, device)
    fields = saved["fingerprint"]["fields"]
    if fields["dataset"] != data.fingerprint:
        raise ValueError("checkpoint and dataset fingerprints differ")
    if fields.get("clip_sha256") != sha256(a.clip):
        raise ValueError("checkpoint and CLIP weight fingerprints differ")
    recipe = Recipe(**{**fields, "batch": a.batch})
    rows = torch.from_numpy(np.flatnonzero(data.arrays["split"] == 0))
    weights = torch.from_numpy(np.asarray(data.arrays["weight"])[rows.numpy()].astype(np.float64))
    plan = epoch_plan(rows, weights, a.batch, recipe.seed, 0)
    if len(plan["rows"]) < a.batch:
        raise ValueError("training split cannot form a complete benchmark batch")

    def batch(step):
        start = (step % (len(plan["rows"]) // a.batch)) * a.batch
        text, image, cond = prepared_batch(data.arrays, plan, start, a.batch, recipe, device, text_on_cpu=True)
        with torch.no_grad():
            features = cache(text, device)
        return features, image, cond

    def evaluation_batch():
        batches = {}
        for split, name in ((1, "iid"), (2, "probe")):
            chosen = np.flatnonzero(data.arrays["split"] == split)[:a.batch]
            if not len(chosen):
                raise ValueError("multi-step validation needs nonempty IID and probe splits")
            image = torch.as_tensor(np.asarray(data.arrays["tokens"][chosen]).astype(np.int64), device=device)
            text = torch.as_tensor(np.asarray(data.arrays["text"][chosen]).astype(np.int64))
            cond = torch.as_tensor(np.asarray(data.arrays["cond"][chosen]).astype(np.int64), device=device)
            with torch.no_grad():
                features = cache(text, device)
            batches[name] = features, image, cond
        return batches

    settings = variants(a.experiment)
    if a.worker == "parity":
        result = parity(model, saved, settings, device, batch(0), a.ce_chunks, recipe.alpha_weight)
    elif a.worker.startswith("curve-"):
        result = run_curve(model, saved, settings[int(a.worker[-1])], device, batch,
                           evaluation_batch, a, recipe.alpha_weight)
    else:
        result = run_updates(model, saved, settings[int(a.worker)], device, batch, a, recipe.alpha_weight)
    result["clip_resident"] = next(clip.parameters()).device.type == device.type
    result["torch_version"] = torch.__version__
    result["cuda_version"] = torch.version.cuda
    result["cache_resident_bytes"] = cache.resident_bytes
    if sha256(a.checkpoint) != checkpoint_sha:
        raise ValueError("checkpoint changed during benchmark")
    result["checkpoint_sha256"] = checkpoint_sha
    result["dataset_fingerprint"] = data.fingerprint
    result["clip_cache_fingerprint"] = cache.fingerprint
    a.result.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize(rows, parity_valid, trials):
    valid = parity_valid and len(rows) == trials * 2 and all(row.get("valid", False) for row in rows)
    identities = {tuple(row.get(key) for key in ("checkpoint_sha256", "dataset_fingerprint", "clip_cache_fingerprint"))
                  for row in rows if row.get("valid")}
    valid &= len(identities) == 1
    summary = {}
    for side in (0, 1):
        values = [row["images_per_second"] for row in rows if row["side"] == side and row.get("valid")]
        if values:
            summary[str(side)] = {"median_img_s": statistics.median(values), "range_img_s": [min(values), max(values)]}
    result = {"comparison_valid": valid, "summary": summary, "trials": rows}
    if valid:
        result["measured_throughput_change_percent"] = (summary["1"]["median_img_s"] / summary["0"]["median_img_s"] - 1) * 100
        result["candidate_range_above_baseline"] = summary["1"]["range_img_s"][0] > summary["0"]["range_img_s"][1]
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("data", "clip-cache", "checkpoint", "clip", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--blind50", type=Path, default=DEFAULT_BLIND50)
    p.add_argument("--eval50", type=Path, default=DEFAULT_EVAL50)
    p.add_argument("--device", required=True)
    p.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    p.add_argument("--batch", type=int, default=80)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--ce-chunks", type=int, default=16)
    p.add_argument("--clip-cache-max-gb", type=float, default=4)
    p.add_argument("--validate-steps", type=int, default=200, help="1..200 paired training updates for any positive timing result")
    p.add_argument("--validate-only", action="store_true", help="run the short trajectory check without timing trials")
    p.add_argument("--curve-atol", type=float, default=1e-3)
    p.add_argument("--curve-rtol", type=float, default=1e-3)
    p.add_argument("--curve-noise-sigma", type=float, default=3.)
    p.add_argument("--worker", choices=("parity", "0", "1", "curve-0", "curve-1"), help=argparse.SUPPRESS)
    p.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if not a.device.startswith("cuda:") or not a.device[5:].isdigit():
        p.error("hardware comparisons require one explicit cuda index")
    if not 1 <= a.steps <= 100 or not 1 <= a.warmup <= 100 or not 5 <= a.trials <= 10 or a.batch <= 0 or a.ce_chunks <= 0:
        p.error("use positive batch/chunks, 1..100 warmup/steps and 5..10 trials")
    if not 1 <= a.validate_steps <= 200 or any(not math.isfinite(value) or value < 0 for value in (a.curve_atol, a.curve_rtol, a.curve_noise_sigma)):
        p.error("validation needs 1..200 steps and finite nonnegative curve tolerances")
    if a.worker:
        if a.result is None or a.result.parent.resolve() != a.out.resolve() or a.result.exists():
            p.error("worker result must be a new file directly inside the benchmark output folder")
        try:
            child(a)
        except Exception as error:
            a.result.write_text(json.dumps({"valid": False, "error_type": type(error).__name__}) + "\n", encoding="utf-8")
            return 1
        return 0
    if a.out.exists():
        p.error("select a new output folder")
    idle_device(a.device)
    a.out.mkdir(parents=True, exist_ok=False)
    forwarded = list(sys.argv[1:] if argv is None else argv)

    def launch(worker, name):
        path = a.out / name
        command = [sys.executable, "-m", "train.pixie32.bench_compute", *forwarded,
                   "--worker", worker, "--result", str(path)]
        completed = subprocess.run(command, check=False)
        result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"valid": False, "error_type": "MissingResult"}
        if completed.returncode:
            result["valid"] = False
        print(json.dumps({"worker": worker, **result}), flush=True)
        return result

    parity_report = launch("parity", "parity.json")
    rows = []
    if parity_report.get("valid") and not a.validate_only:
        for trial in range(a.trials):
            for side in ((0, 1) if trial % 2 == 0 else (1, 0)):
                result = launch(str(side), f"trial-{trial}-{side}.json")
                rows.append({"trial": trial, "side": side, **result})
                if not result.get("valid"):
                    break
            if any(not row.get("valid") for row in rows):
                break
    same_inputs = all(all(row.get(key) == parity_report.get(key) for key in
                          ("checkpoint_sha256", "dataset_fingerprint", "clip_cache_fingerprint"))
                      for row in rows if row.get("valid"))
    report = summarize(rows, parity_report.get("valid", False) and same_inputs, a.trials)
    needs_curve = a.validate_only or (report["comparison_valid"] and report["measured_throughput_change_percent"] > 0)
    curve = {"valid": None, "status": "not run; no positive timing result"}
    if parity_report.get("valid") and needs_curve:
        trajectories = [launch(f"curve-{side}", f"curve-{side}.json") for side in (0, 1)]
        same_curve_inputs = all(all(row.get(key) == parity_report.get(key) for key in
                               ("checkpoint_sha256", "dataset_fingerprint", "clip_cache_fingerprint")) for row in trajectories)
        if all(row.get("valid") for row in trajectories) and same_curve_inputs:
            curve = curve_comparison(*trajectories, steps=a.validate_steps,
                                     atol=a.curve_atol, rtol=a.curve_rtol, noise_sigma=a.curve_noise_sigma)
        else:
            curve = {"valid": False, "status": "trajectory failure or mismatched input identity"}
    if a.validate_only:
        report["comparison_valid"] = False
        report["validation_valid"] = bool(parity_report.get("valid") and curve["valid"])
    report["multi_step_parity"] = curve
    report["adoption_gate_passed"] = bool(report["comparison_valid"] and report.get("candidate_range_above_baseline")
                                          and needs_curve and curve["valid"])
    report.update({"experiment": a.experiment, "variants": [asdict(v) for v in variants(a.experiment)],
                   "parity": parity_report, "batch": a.batch, "warmup": a.warmup, "steps": a.steps,
                   "trials_per_side": a.trials, "ce_chunks": a.ce_chunks,
                   "scope": "completed training updates including data/cache, forward, backward, clipping and AdamW; resident CLIP; excludes startup, compile warmup, eval, sampling, checkpoints",
                   "verdict": ("validation only; no timing claim" if a.validate_only else
                               "invalid" if not report["comparison_valid"] else
                               "multi-step parity failed; do not adopt" if needs_curve and not curve["valid"] else
                               "timing ranges overlap or candidate slower; no speed winner" if not report.get("candidate_range_above_baseline") else
                               "compare ranges and assess the measured limiter before adopting")})
    (a.out / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if (report.get("validation_valid") if a.validate_only else report["comparison_valid"] and (not needs_curve or curve["valid"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
