#!/usr/bin/env python3
"""Offline CLIP + FastSampler comparison; excludes reward, PNG, polling and HTTP.

Run only while the chat worker and other compute users of the selected GPU are
stopped. Assets are explicit local inputs; no worker config or key is loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import time
from pathlib import Path

import torch


def load_weights(path):
    # Match the worker's loader for checkpoints containing cross-platform Paths.
    saved = pathlib.WindowsPath, pathlib.PosixPath
    pathlib.WindowsPath, pathlib.PosixPath = pathlib.PureWindowsPath, pathlib.PurePosixPath
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        pathlib.WindowsPath, pathlib.PosixPath = saved


def require_idle_gpu(index):
    """Check the physical index before initializing CUDA; never stop processes."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("unset CUDA_VISIBLE_DEVICES so --device selects a physical GPU")
    result = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
    for line in result.stdout.splitlines():
        if line.strip() and line.strip() != str(os.getpid()):
            raise RuntimeError("selected GPU has another compute process or unverified occupancy; stop it before benchmarking")


def summary(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def compare_outputs(original, cached):
    pairs = list(zip(original[:2], cached[:2]))
    finite = all(bool(torch.isfinite(a).all() and torch.isfinite(b).all()) for a, b in pairs)
    return {"finite": finite,
            "features_exact": all(torch.equal(a, b) for a, b in pairs),
            "max_error": max(float((a - b).abs().max().item()) for a, b in pairs) if finite else None,
            "matched_sprites": int((original[2] == cached[2]).all(dim=1).sum().item())}


def write_report(report, out):
    replay = report.get("seed_replay")
    replay_valid = replay is None or replay["exact_replay_matches"] == replay["ignored_tail_matches"] == replay["sprites_checked"]
    report["comparison_valid"] = (report["finite_features"] and report["features_exact"]
                                  and report["exact_sprite_match_fraction"] == 1.0 and replay_valid)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
    print(encoded)
    return 0 if report["comparison_valid"] else 1


def main(argv=None):
    from tokenizers import Tokenizer
    import v3_model as M
    from fast_sample import FastSampler
    from blank_features import BlankFeatures
    p = argparse.ArgumentParser(description=__doc__)
    for flag, env in (("checkpoint", "V3_CKPT"), ("clip-weights", "CLIP_WEIGHTS"),
                      ("tokenizer", "TOKENIZER")):
        p.add_argument(f"--{flag}", default=os.environ.get(env), help=f"local asset or ${env}")
    p.add_argument("--device", required=True, help="explicit physical CUDA device, e.g. cuda:0")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--prompt", action="append", help="repeat to test several prompts")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--verify-seed-replay", action="store_true",
                   help="also verify complete-batch replay and the ignored per-row seed tail")
    p.add_argument("--palette-ids", type=int, nargs=3, help="optional palette condition")
    p.add_argument("--style", type=int, nargs=3, help="optional fill/outline/colour condition")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--out", type=Path, required=True, help="new JSON file; refuses overwrite")
    args = p.parse_args(argv)
    for flag in ("checkpoint", "clip_weights", "tokenizer"):
        if not getattr(args, flag) or not Path(getattr(args, flag)).is_file():
            p.error(f"supply an existing local --{flag.replace('_', '-')} asset")
    if args.out and args.out.exists():
        p.error("--out must be a new file")
    if not 1 <= args.batch <= 64 or not 5 <= args.warmup <= 20 or not 5 <= args.trials <= 100:
        p.error("batch must be 1..64; warmup 5..20 and trials 5..100")
    if not 0 <= args.seed <= 2**31 - 1:
        p.error("seed must be 0..2147483647")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        p.error("--device must include an explicit CUDA index")
    if args.palette_ids and any(x != M.NO_PALETTE and not 1 <= x <= 4096 for x in args.palette_ids):
        p.error("palette ids must be 1..4096 or NO_PALETTE (4097)")
    if args.style and any(not 0 <= x < limit for x, limit in zip(args.style, M.STYLE_SIZES)):
        p.error("invalid style condition")
    require_idle_gpu(device.index)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    ck = load_weights(args.checkpoint)
    config = M.V3Config(**ck["config"])
    if config.arch != "p50":
        p.error("benchmark requires the worker's P50 checkpoint")
    model = M.build(config)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if unexpected or any(k not in ("pr", "pc", "vmask") for k in missing):
        raise ValueError("checkpoint does not match the P50 worker model")
    del ck
    model.to(device).eval()
    clip = M.ClipTextPenultimate()
    clip.load_hf(load_weights(args.clip_weights))
    clip.to(device=device, dtype=torch.bfloat16).eval()
    features = BlankFeatures(clip)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    prompts = args.prompt or ["copper golem", "tiny azure jelly", "rusted cutlass"]
    rows = []
    for prompt in prompts:
        # Match worker_v3.t26 + trim, including literal special-token inputs.
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids[:24]
        full = [M.CLIP_BOS] + ids + [M.CLIP_EOS] * (25 - len(ids))
        row = full[:config.text_len]
        if sum(x != M.CLIP_EOS for x in full[1:]) > config.text_len - 2:
            row[config.text_len - 1] = M.CLIP_EOS
        rows.append(row)
    blank = [M.CLIP_BOS] + [M.CLIP_EOS] * (config.text_len - 1)
    cond = None
    if args.palette_ids or args.style:
        if config.n_pal != 3 or not config.style:
            p.error("explicit conditions require the worker's three palette slots and style slot")
        cond = torch.tensor([(args.palette_ids or [M.NO_PALETTE] * 3) + (args.style or [0] * 3)],
                            device=device, dtype=torch.long).expand(args.batch, -1).contiguous()
    seeds = torch.arange(args.seed, args.seed + args.batch, device=device, dtype=torch.int64)
    sampler = FastSampler(model, device, max_batch=args.batch * 2, use_graph=not args.no_graph,
                          cfg_scale=2.0, temperature=0.8, top_k=32)
    with torch.no_grad():
        sampler.prepare([args.batch * 2])

    def encode(row, optimized):
        if optimized:
            return features.batch(row, blank, args.batch)
        return (features.encode_rows([row] * args.batch),
                features.encode_rows([blank] * args.batch))

    def sync():
        torch.cuda.synchronize(device)

    @torch.no_grad()
    def run(row, optimized):
        sync()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        h, u = encode(row, optimized)
        sync()
        t1 = time.perf_counter()
        indices = sampler.sample(h, u, seeds=seeds, cond=cond)
        sync()
        t2 = time.perf_counter()
        return (h, u, indices), {"encode_ms": (t1 - t0) * 1000, "sample_ms": (t2 - t1) * 1000,
                         "encode_sample_ms": (t2 - t0) * 1000,
                         "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20}

    for i in range(args.warmup):
        row = rows[i % len(rows)]
        for optimized in ((False, True) if i % 2 == 0 else (True, False)):
            run(row, optimized)
    measurements = {"baseline": [], "reuse": []}
    matched = 0
    max_error = 0.0
    features_exact = True
    finite = True
    replay_matches = 0
    tail_matches = 0
    for i in range(args.trials):
        outputs = {}
        row = rows[i % len(rows)]
        for optimized in ((False, True) if i % 2 == 0 else (True, False)):
            key = "reuse" if optimized else "baseline"
            outputs[key], timing = run(row, optimized)
            measurements[key].append(timing)
        parity = compare_outputs(outputs["baseline"], outputs["reuse"])
        finite = finite and parity["finite"]
        features_exact = features_exact and parity["features_exact"]
        max_error = max(max_error, parity["max_error"]) if finite else None
        matched += parity["matched_sprites"]
        if args.verify_seed_replay:
            h, u, expected = outputs["baseline"]
            repeated = sampler.sample(h, u, seeds=torch.full_like(seeds, args.seed), cond=cond)
            changed_tail = torch.full_like(seeds, (args.seed + 1) % 2**31)
            changed_tail[0] = args.seed
            tail = sampler.sample(h, u, seeds=changed_tail, cond=cond)
            replay_matches += int((expected == repeated).all(dim=1).sum().item())
            tail_matches += int((expected == tail).all(dim=1).sum().item())
    report = {
        "scope": "CLIP encoding plus FastSampler only; excludes reward, PNG, polling and HTTP",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "batch": args.batch, "text_len": config.text_len, "cfg": 2.0,
        "temperature": 0.8, "top_k": 32, "graphs": not args.no_graph,
        "warmup_pairs": args.warmup, "trial_pairs": args.trials, "prompt_count": len(rows),
        "conditioned": cond is not None, "seed_start": args.seed,
        "mode": "full candidate-batch blank cache; original full prompt batch",
        "finite_features": finite, "features_exact": features_exact,
        "feature_max_abs_error": max_error,
        "exact_sprite_match_fraction": matched / (args.trials * args.batch),
        "measurements": measurements,
        "summary": {key: {metric: summary([row[metric] for row in values])
                          for metric in values[0]} for key, values in measurements.items()},
    }
    if args.verify_seed_replay:
        report["seed_replay"] = {
            "scheme": "torch-gumbel-batch-v1", "job_seed": args.seed,
            "scope": "same full batch and runtime; no chunk/reorder invariance promised",
            "sprites_checked": args.trials * args.batch,
            "exact_replay_matches": replay_matches, "ignored_tail_matches": tail_matches,
        }
    return write_report(report, args.out)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("benchmark failed: " + type(error).__name__)
        raise SystemExit(1)
