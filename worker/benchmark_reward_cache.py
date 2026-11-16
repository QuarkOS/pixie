"""Compare uncached, cold-cache and warm-cache full worker calls offline."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import importlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_blank_cache import require_idle_gpu
from benchmark_worker import signature, stats
from fast_v7 import FastRewardV7


class TimedReward:
    def __init__(self, scorer):
        self.scorer = scorer

    def score_indices(self, indices, prompt):
        self.indices = np.stack(indices).copy()
        began = time.perf_counter()
        self.scores = self.scorer.score_indices(indices, prompt)
        self.milliseconds = (time.perf_counter() - began) * 1000
        return self.scores


def compare(module, generator, *, prompts, trials, warmup, batch, show, candidates):
    if generator.scorer_name != "reward_v7" or generator.v7 is None:
        raise ValueError("real reward-v7 scorer required")
    uncached = generator.v7
    uncached.text_cache_size = 0
    cached = FastRewardV7(uncached.v7, generator.palette, text_cache_size=64)
    sides = {"uncached": TimedReward(uncached), "cold": TimedReward(cached), "warm": TimedReward(cached)}
    generator.profile_phases = False

    def call(side, prompt, seed):
        scorer = sides[side]
        generator.v7 = scorer
        if side == "cold":
            cached.clear_text_cache()
        elif side == "warm":
            for key in cached.v7.models:
                cached._encode_text(key, prompt)
        before_hits, before_misses = cached.text_cache_hits, cached.text_cache_misses
        torch.cuda.synchronize(generator.device)
        began = time.perf_counter()
        result = generator.run(prompt, batch, show, candidates, seed=seed)
        torch.cuda.synchronize(generator.device)
        elapsed = (time.perf_counter() - began) * 1000
        if result[-1] != "reward_v7":
            raise ValueError("reward scorer fell back; no valid comparison")
        return {"worker_ms": elapsed, "reward_ms": scorer.milliseconds,
                "hits": cached.text_cache_hits - before_hits, "misses": cached.text_cache_misses - before_misses,
                "signature": signature(generator, result), "scores": scorer.scores.copy(), "indices": scorer.indices}

    try:
        for index in range(warmup):
            for side in ("uncached", "cold", "warm"):
                call(side, prompts[index % len(prompts)], 1000 + index * batch)
        torch.cuda.reset_peak_memory_stats(generator.device)
        rows = []
        profiles = []
        names = ("uncached", "cold", "warm")
        for trial in range(trials):
            for prompt_index, prompt in enumerate(prompts):
                order = names[trial % 3:] + names[:trial % 3]
                values = {side: call(side, prompt, 2000 + (trial * len(prompts) + prompt_index) * batch) for side in order}
                baseline = values["uncached"]
                exact = all(value["signature"] == baseline["signature"] and
                            np.array_equal(value["scores"], baseline["scores"]) and np.isfinite(value["scores"]).all()
                            for value in values.values())
                models = len(cached.v7.models)
                if values["warm"]["hits"] != models or values["warm"]["misses"] or values["cold"]["misses"] != models:
                    raise ValueError("cache workload did not execute as declared")
                rows.append({"trial": trial, "prompt_index": prompt_index, "exact_scores_rankings_pngs": bool(exact),
                             "sides": {side: {key: value[key] for key in ("worker_ms", "reward_ms", "hits", "misses")}
                                       for side, value in values.items()}})
                if trial == 0:
                    for side, scorer in (("uncached", uncached), ("warm", cached)):
                        scorer.profile_phases = True
                        scores = scorer.score_indices(baseline["indices"], prompt)
                        if not np.array_equal(scores, baseline["scores"]):
                            raise ValueError("profiled scorer output mismatch")
                        profiles.append({"side": side, "prompt_index": prompt_index, "profile": scorer.last_profile})
                        scorer.profile_phases = False
        return {"comparison_valid": all(row["exact_scores_rankings_pngs"] for row in rows),
                "scope": "full offline worker generation and completed reward scoring; excludes queue, polling, JSON and HTTP",
                "trials": trials, "batch": batch, "show": show, "stored_candidates": candidates,
                "warmup_calls_per_side": warmup, "prompt_count": len(prompts),
                "cache_scope": "warm is a repeated-prompt best case; cold includes token/model cache-key lookup and both text encodes",
                "cache_info": cached.text_cache_info(),
                "summary": {side: {metric: stats([row["sides"][side][metric] for row in rows])
                                    for metric in ("worker_ms", "reward_ms")} for side in names},
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(generator.device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(generator.device),
                "rows": rows, "separate_reward_profiles": profiles}
    finally:
        generator.v7 = uncached
        uncached.profile_phases = cached.profile_phases = False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--show", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--prompt", action="append")
    args = parser.parse_args(argv)
    if args.out.exists() or not args.config.is_file():
        parser.error("supply an explicit local config and a new output folder")
    if not 5 <= args.trials <= 10 or not 2 <= args.warmup <= 10:
        parser.error("trials must be 5..10 and warmup 2..10")
    if not 1 <= args.show <= args.candidates <= args.batch <= 64:
        parser.error("require 1 <= show <= candidates <= batch <= 64")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        parser.error("select one explicit physical CUDA index")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if config.get("device", "cuda") != "cuda" or config.get("v7_device", "cuda") not in ("cuda", "cuda:0"):
            raise ValueError("config must use CUDA on the selected device")
        if config.get("scorer") != "reward_v7":
            raise ValueError("config must select reward_v7")
        require_idle_gpu(device.index)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(f"cannot validate inputs ({type(error).__name__})")
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
    os.environ["PIXEL_WORKER_CONFIG"] = str(args.config.resolve())
    os.environ["PIXEL_REWARD_TEXT_CACHE"] = "0"
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            module = importlib.import_module("worker_v3")
            torch.cuda.set_device(0)
            generator = module.Gen()
            result = compare(module, generator, prompts=args.prompt or ["copper golem", "azure jelly", "rusted cutlass"],
                             trials=args.trials, warmup=args.warmup, batch=args.batch,
                             show=args.show, candidates=args.candidates)
            result["checkpoint_sha256"] = generator.info["sha"]
            result["reward_head_sha256"] = module.sha256(Path(config.get("v7_head") or (Path(config["v7_root"]) / "final_head.json")))
            result["clip_weights_sha256"] = module.sha256(config["clip_pt"])
            result["tokenizer_sha256"] = module.sha256(config["clip_tokenizer"])
            result["gpu_model"] = torch.cuda.get_device_name(0)
            result["sampling_settings"] = generator.last_sampling
    except (Exception, SystemExit) as error:
        result = {"comparison_valid": False, "error_type": type(error).__name__, "scope": "failed offline reward-cache benchmark; no valid speed result"}
    (args.out / "reward-cache-benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))
    return 0 if result["comparison_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
