"""Compare completed online/cached training updates; never writes checkpoints."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from train.pixie32.clip_cache import FeatureCache, idle_device
from train.pixie32.data import DEFAULT_BLIND50, DEFAULT_EVAL50, load_dataset, sha256
from train.pixie32.p32_model import P32, P32Config
from train.pixie32.train_p32 import Recipe, autocast, epoch_plan, load_clip, prepared_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "clip-cache", "checkpoint", "clip", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--blind50", type=Path, default=DEFAULT_BLIND50)
    p.add_argument("--eval50", type=Path, default=DEFAULT_EVAL50)
    p.add_argument("--device", required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--ce-chunks", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--clip-cache-max-gb", type=float, default=4)
    a = p.parse_args()
    if a.out.exists() or not 1 <= a.steps <= 100 or not 1 <= a.warmup <= 100 or not 5 <= a.trials <= 10 or a.batch <= 0 or a.ce_chunks <= 0:
        p.error("choose a new output folder, 1..100 steps/warmup, 5..10 trials and positive batch/chunks")
    device = idle_device(a.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = load_dataset(a.data, a.blind50, a.eval50)
    cache = FeatureCache(a.clip_cache, data, sha256(a.clip), a.clip_cache_max_gb)
    if cache.compute != ("bf16" if device.type == "cuda" else "float32"):
        p.error("rebuild CLIP cache on the benchmark device type")
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    saved = {"model": saved["model"], "config": saved["config"]}
    model = P32(P32Config(**saved["config"])).to(device)
    model.gradient_checkpointing = a.gradient_checkpointing
    clip = load_clip(a.clip, device)
    loss_fn = torch.compile(model.loss_fwd) if a.compile else model.loss_fwd
    recipe = Recipe(a.batch, 42, data.fingerprint, model.config.to_dict(), 1000000, 0, 0, 1e-4, 1e-5)
    rows = torch.from_numpy(np.flatnonzero(data.arrays["split"] == 0))
    if not len(rows):
        p.error("empty training split")
    weights = torch.from_numpy(np.asarray(data.arrays["weight"])[rows.numpy()].astype(np.float64))
    plan = epoch_plan(rows, weights, a.batch, 42, 0)
    def batch(step, cached):
        start = (step * a.batch) % len(plan["rows"])
        return prepared_batch(data.arrays, plan, start, a.batch, recipe, device, text_on_cpu=cached)
    text, _, _ = batch(0, True)
    with torch.no_grad(), autocast(device):
        online = clip(text.to(device)).float()
        cached = cache(text, device).float()
    feature_error = float((online - cached).abs().max())
    # CUDA batch/kernel differences can change BF16 rounding. Refuse clearly stale/wrong features.
    feature_relative_error = float((online - cached).norm() / online.norm().clamp_min(1e-9))
    if not torch.isfinite(online).all() or not torch.isfinite(cached).all() or feature_relative_error > .02:
        raise ValueError("cached/online CLIP feature parity failed (relative L2 > 0.02)")
    del online, cached
    trials = []
    a.out.mkdir(parents=True, exist_ok=False)
    for trial in range(a.trials):
        for use_cache in ((False, True) if trial % 2 == 0 else (True, False)):
            model.load_state_dict(saved["model"], strict=True)
            torch.random.default_generator.manual_seed(42)
            if device.type == "cuda":
                torch.cuda.manual_seed(42)
            optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
            model.train()
            losses = []
            for step in range(a.warmup + a.steps):
                if step == a.warmup:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    began = time.perf_counter()
                text, image, cond = batch(step, use_cache)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(), autocast(device):
                    features = cache(text, device) if use_cache else clip(text)
                with autocast(device):
                    loss, _ = loss_fn(features, image, recipe.alpha_weight, cond=cond, chunks=a.ce_chunks)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite benchmark loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                if step >= a.warmup:
                    losses.append(loss.detach())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - began
            trials.append({"trial": trial, "cached": use_cache, "updates": a.steps,
                           "images": a.steps * a.batch, "seconds": seconds,
                           "images_per_second": a.steps * a.batch / seconds,
                           "mean_loss": float(torch.stack(losses).mean()),
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0})
            del optimizer
            print(json.dumps(trials[-1]), flush=True)
    summary = {}
    for mode in (False, True):
        values = [t["images_per_second"] for t in trials if t["cached"] == mode]
        summary["cached" if mode else "online"] = {"median_img_s": statistics.median(values), "range_img_s": [min(values), max(values)]}
    result = {"summary": summary, "trials": trials, "feature_max_abs_error": feature_error,
              "feature_relative_l2_error": feature_relative_error, "cache_sha256": cache.fingerprint,
              "config": {key: getattr(a, key) for key in ("batch", "steps", "warmup", "trials", "ce_chunks", "compile", "gradient_checkpointing")},
              "torch_version": torch.__version__, "cache_budget_bytes": cache.resident_bytes,
              "scope": "augmentation, CLIP or RAM transfer, forward, backward, clipping and AdamW; excludes startup/eval/sampling",
              "verdict": "inconclusive until limiter and variability are assessed on target hardware"}
    (a.out / "benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
