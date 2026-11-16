"""Offline complete generation benchmark: CLIP, sampler, reward, RGBA and PNG.

Uses an explicitly supplied local worker config. Never polls or posts to an app.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import importlib
import io
import json
import os
from pathlib import Path
import statistics
import time

from benchmark_blank_cache import require_idle_gpu
from latency_identity import json_hash, workload_identity


def stats(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def signature(worker, result):
    out, all_rows, _, _, scorer = result
    return {"results": out, "all": all_rows, "candidates": worker.last_candidates,
            "sampling": worker.last_sampling, "scorer": scorer}


def benchmark(module, generator, *, prompts, trials, warmup, batch, show, candidates):
    def call(profile, prompt, seed):
        generator.profile_phases = profile
        module.torch.cuda.synchronize(generator.device)
        began = time.perf_counter()
        result = generator.run(prompt, batch, show, candidates, seed=seed)
        module.torch.cuda.synchronize(generator.device)
        elapsed = (time.perf_counter() - began) * 1000
        return elapsed, signature(generator, result), generator.last_profile

    for index in range(warmup):
        call(False, prompts[index % len(prompts)], 1000 + index * batch)
    module.torch.cuda.reset_peak_memory_stats(generator.device)
    rows = []
    for trial in range(trials):
        for prompt_index, prompt in enumerate(prompts):
            pairs = {}
            for profiled in ((False, True) if trial % 2 == 0 else (True, False)):
                pairs[profiled] = call(profiled, prompt, 2000 + (trial * len(prompts) + prompt_index) * batch)
            exact = pairs[False][1] == pairs[True][1]
            if pairs[False][1]["scorer"] != module.SCORER_CFG or pairs[True][1]["scorer"] != module.SCORER_CFG:
                raise ValueError("configured scorer fell back; benchmark is invalid")
            profile = pairs[True][2]
            if profile is None:
                raise ValueError("worker did not produce a phase profile")
            rows.append({"trial": trial, "prompt_index": prompt_index, "exact_outputs": exact,
                         "unprofiled_ms": pairs[False][0], "profiled_ms": pairs[True][0], "profile": profile})
    phase_names = sorted({key for row in rows for key in row["profile"]["phase_ms"]})
    return {"comparison_valid": all(row["exact_outputs"] for row in rows),
            "scope": "complete offline generation with real configured scorer, shown and stored PNGs; excludes queue, polling, JSON serialization and HTTP",
            "timing_status": "measured on selected local hardware when this command runs",
            "profile_note": "synchronization perturbs execution; compare clean total with instrumented total, never sum phase medians",
            "batch": batch, "show": show, "stored_candidates": candidates,
            "warmup_calls": warmup, "trials": trials, "prompt_count": len(prompts),
            "scorer": module.SCORER_CFG, "blank_cache": module.CACHE_BLANK, "graph": module.USE_GRAPH,
            "unprofiled_ms": stats([row["unprofiled_ms"] for row in rows]),
            "profiled_ms": stats([row["profiled_ms"] for row in rows]),
            "phase_ms": {key: stats([row["profile"]["phase_ms"][key] for row in rows]) for key in phase_names},
            "peak_allocated_bytes": module.torch.cuda.max_memory_allocated(generator.device),
            "peak_reserved_bytes": module.torch.cuda.max_memory_reserved(generator.device),
            "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True, help="explicit local worker config; paths and URLs are never copied to the report")
    parser.add_argument("--device", required=True, help="one physical GPU, e.g. cuda:0; GPU must be idle")
    parser.add_argument("--out", type=Path, required=True, help="new output folder")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--show", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--prompt", action="append")
    args = parser.parse_args(argv)
    if args.out.exists() or not args.config.is_file():
        parser.error("supply a local config and a new output folder")
    if not 5 <= args.trials <= 10 or not 2 <= args.warmup <= 10:
        parser.error("trials must be 5..10 and warmup 2..10")
    if not 1 <= args.show <= args.candidates <= args.batch <= 64:
        parser.error("require 1 <= show <= candidates <= batch <= 64")
    import torch
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        parser.error("select one explicit physical CUDA index")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if config.get("device", "cuda") != "cuda" or config.get("v7_device", "cuda") not in ("cuda", "cuda:0"):
            raise ValueError("config must use CUDA on the one selected device")
        require_idle_gpu(device.index)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(f"cannot validate config or idle GPU ({type(error).__name__})")
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
    os.environ["PIXEL_WORKER_CONFIG"] = str(args.config.resolve())
    try:
        # Worker startup logs may contain asset paths. Keep them local in memory;
        # no worker key, lock port, poll loop or HTTP endpoint is used.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            module = importlib.import_module("worker_v3")
            torch.cuda.set_device(0)
            generator = module.Gen()
            if generator.scorer_name != module.SCORER_CFG:
                raise ValueError("configured scorer did not load")
            prompts = args.prompt or ["copper golem", "azure jelly", "rusted cutlass"]
            result = benchmark(module, generator, prompts=prompts,
                               trials=args.trials, warmup=args.warmup, batch=args.batch,
                               show=args.show, candidates=args.candidates)
            result["checkpoint_sha256"] = generator.info["sha"]
            result["sampling_settings"] = generator.last_sampling
            result["clip_weights_sha256"] = module.sha256(module.CFG["clip_pt"])
            result["tokenizer_sha256"] = module.sha256(module.CFG["clip_tokenizer"])
            result["workload_identity"] = workload_identity(module, generator, prompts, batch=args.batch, show=args.show,
                                                            candidates=args.candidates, trials=args.trials, warmup=args.warmup)
            result["workload_sha256"] = json_hash(result["workload_identity"])
            result["gpu_model"] = torch.cuda.get_device_name(0)
    except (Exception, SystemExit) as error:
        result = {"comparison_valid": False, "error_type": type(error).__name__,
                  "scope": "offline worker benchmark failed; no valid speed claim"}
    (args.out / "worker-benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))
    return 0 if result["comparison_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
