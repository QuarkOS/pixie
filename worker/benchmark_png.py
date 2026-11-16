"""Compare lossless PNG encoding alone or the real complete offline worker."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import importlib
import io
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image

from png_encoding import encode_png, pixels


def stats(values):
    return {"median": statistics.median(values), "range": [min(values), max(values)]}


def compare_signatures(left, right):
    """Compare actual delivered pixels and metadata, allowing PNG byte changes."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return False
        return all(np.array_equal(np.asarray(pixels(left[key])), np.asarray(pixels(right[key]))
                                  ) if key == "png" else compare_signatures(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(compare_signatures(a, b) for a, b in zip(left, right))
    return left == right


def payload_bytes(value):
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode())


def run_pairs(call, trials, prompts):
    rows = []
    for trial in range(trials):
        for prompt_index, prompt in enumerate(prompts):
            outputs = {}
            for mode in (("legacy", "fast") if trial % 2 == 0 else ("fast", "legacy")):
                outputs[mode] = call(mode, prompt, 101 + trial * len(prompts) + prompt_index)
            rows.append({"trial": trial, "prompt_index": prompt_index,
                         "pixels_and_metadata_exact": compare_signatures(outputs["legacy"][1], outputs["fast"][1]),
                         "legacy_ms": outputs["legacy"][0], "fast_ms": outputs["fast"][0],
                         "legacy_payload_bytes": payload_bytes(outputs["legacy"][1]),
                         "fast_payload_bytes": payload_bytes(outputs["fast"][1])})
    return {"comparison_valid": all(row["pixels_and_metadata_exact"] for row in rows), "rows": rows,
            "summary": {key: stats([row[key] for row in rows]) for key in
                        ("legacy_ms", "fast_ms", "legacy_payload_bytes", "fast_payload_bytes")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sprites", type=Path, help="folder of local PNG sprites; encoding-only CPU measurement")
    source.add_argument("--worker-config", type=Path, help="local config for full offline GPU worker comparison")
    parser.add_argument("--device", help="explicit idle physical GPU for --worker-config")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--count", type=int, default=20, help="PNG count per encoding-only batch")
    parser.add_argument("--scale", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or not 5 <= args.trials <= 10 or not 1 <= args.count <= 64 or not 1 <= args.scale <= 32:
        parser.error("choose a new folder, 5..10 trials, 1..64 count and 1..32 scale")
    if args.worker_config:
        if not args.worker_config.is_file() or not args.device:
            parser.error("worker mode requires local --worker-config and explicit --device")
    elif not args.sprites.is_dir():
        parser.error("--sprites must be a folder of PNGs")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        if args.sprites:
            files = sorted(args.sprites.rglob("*.png"))[:args.count]
            if len(files) != args.count:
                raise ValueError("not enough PNG sprites for requested count")
            sprites = []
            for path in files:
                with Image.open(path) as image:
                    sprites.append(np.asarray(image.convert("RGBA").resize((16, 16), Image.Resampling.NEAREST)).copy())
            def call(mode, prompt, seed):
                began = time.perf_counter()
                result = [{"png": encode_png(sprite, args.scale, mode)} for sprite in sprites]
                return (time.perf_counter() - began) * 1000, result
            for mode in ("legacy", "fast"):
                call(mode, None, None)
            result = run_pairs(call, args.trials, [None])
            result.update(scope="CPU resize, PNG and base64 only; input sprites resized to native16 before timing; excludes model, reward, JSON, queue and HTTP",
                          count=args.count, output_size=[16 * args.scale] * 2, pillow_version=Image.__version__)
        else:
            import torch
            from benchmark_blank_cache import require_idle_gpu
            from benchmark_worker import signature
            device = torch.device(args.device)
            if device.type != "cuda" or device.index is None:
                raise ValueError("select an explicit physical CUDA device")
            config = json.loads(args.worker_config.read_text(encoding="utf-8"))
            if config.get("device", "cuda") != "cuda" or config.get("v7_device", "cuda") not in ("cuda", "cuda:0"):
                raise ValueError("worker config must use the selected GPU")
            require_idle_gpu(device.index)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
            os.environ["PIXEL_WORKER_CONFIG"] = str(args.worker_config.resolve())
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                module = importlib.import_module("worker_v3")
                torch.cuda.set_device(0)
                generator = module.Gen()
                generator.profile_phases = False
                if generator.scorer_name != module.SCORER_CFG:
                    raise ValueError("configured scorer failed to load")
                def call(mode, prompt, seed):
                    generator.png_mode = mode
                    torch.cuda.synchronize(generator.device)
                    began = time.perf_counter()
                    generated = generator.run(prompt, 64, 4, 20, seed=seed)
                    torch.cuda.synchronize(generator.device)
                    elapsed = (time.perf_counter() - began) * 1000
                    if generated[4] != module.SCORER_CFG:
                        raise ValueError("scorer fallback invalidates full-worker comparison")
                    return elapsed, signature(generator, generated)
                prompts = ["copper golem", "azure jelly", "rusted cutlass"]
                for mode in ("legacy", "fast"):
                    for prompt in prompts:
                        call(mode, prompt, 11)
                result = run_pairs(call, args.trials, prompts)
                result.update(scope="complete offline worker at64 candidates, show4/store20; excludes JSON serialization, queue and HTTP",
                              checkpoint_sha256=generator.info["sha"], sampling=generator.last_sampling,
                              output_size=[16 * module.SCALE] * 2,
                              clip_sha256=module.sha256(module.CFG["clip_pt"]),
                              tokenizer_sha256=module.sha256(module.CFG["clip_tokenizer"]))
    except (Exception, SystemExit) as error:
        result = {"comparison_valid": False, "error_type": type(error).__name__}
    (args.out / "png-benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))
    return 0 if result["comparison_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
