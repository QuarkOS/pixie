"""Path-free identity of the actual offline worker workload, outside timing."""
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys


PROTOCOL = "pixie-worker-latency-v2"


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_hash(path):
    path = Path(path)
    data = path.read_bytes()
    if path.suffix.lower() in (".py", ".pyw"):
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def state_hash(state):
    """Hash exact dense tensors one at a time, including names/shapes/dtypes."""
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":")).encode())
        data = tensor.reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(data).cast("B"))
    return digest.hexdigest()


def workload_identity(module, generator, prompts, *, batch, show, candidates, trials, warmup):
    import numpy as np
    import PIL
    torch = module.torch
    reward = None
    if generator.scorer_name == "reward_v7":
        import common
        from fast_v7 import TEMPLATES
        fast = generator.v7
        scorer = fast.v7
        head = {"mu": scorer.mu, "sd": scorer.sd, "w": scorer.w}
        for key, (mean, projection) in scorer.pca.items():
            head[f"pca:{key}:mu"] = mean
            head[f"pca:{key}:projection"] = projection
        reward = {"recipe_sha256": json_hash(scorer.recipe), "head_state_sha256": state_hash(head),
                  "actual_render_size": int(fast.size),
                  "scorer_implementation_sha256": source_hash(sys.modules[type(scorer).__module__].__file__),
                  "hand_feature_implementation_sha256": source_hash(common.__file__),
                  "models": {key: {"weights_sha256": state_hash(value["model"].state_dict()),
                                    "render_size": int(value["size"]), "mean": [float(x) for x in value["mean"]],
                                    "std": [float(x) for x in value["std"]],
                                    "ordered_text_token_sha256": [state_hash({name: value["tokenizer"]([template.format(prompt=prompt)])
                                                                            for name, template in TEMPLATES.items()})
                                                                  for prompt in prompts]}
                             for key, value in sorted(scorer.models.items())},
                  "text_cache_capacity": getattr(fast, "text_cache_size", 0)}
    else:
        reward = {"feature_head_sha256": state_hash({"mu": module.MU, "sd": module.SD, "w": module.W})}
    settings = dict(generator.last_sampling)
    settings.pop("job_seed", None)
    return {
        "protocol": PROTOCOL,
        "ordered_prompt_sha256": [hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts],
        "seed_policy": "warmup:1000+index*batch;trial:2000+(trial*prompt_count+prompt_index)*batch",
        "batch": batch, "show": show, "stored_candidates": candidates, "trials": trials, "warmup_calls": warmup,
        "checkpoint_sha256": generator.info["sha"],
        "clip_weights_sha256": module.sha256(module.CFG["clip_pt"]),
        "tokenizer_sha256": module.sha256(module.CFG["clip_tokenizer"]),
        "palette_state_sha256": state_hash({"palette": np.asarray(generator.palette)}),
        "implementation_sha256": {name: source_hash(Path(__file__).parent / name)
                                  if (Path(__file__).parent / name).is_file() else None
                                  for name in ("worker_v3.py", "v3_model.py", "fast_sample.py", "fast_v7.py",
                                               "blank_features.py", "phase_timing.py", "png_encoding.py",
                                               "benchmark_worker.py", "benchmark_blank_cache.py", "latency_identity.py")},
        "loaded_model_implementation_sha256": source_hash(module.M.__file__),
        "loaded_runtime_implementation_sha256": {
            name: source_hash(sys.modules[getattr(module, name).__module__].__file__)
            for name in ("FastSampler", "FastRewardV7", "BlankFeatures", "PhaseTimer")},
        "sampling_settings": settings, "png_scale": module.SCALE,
        "png_mode": getattr(generator, "png_mode", getattr(module, "PNG_MODE", "legacy")),
        "reward": reward,
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(), "pillow": PIL.__version__, "numpy": np.__version__,
                    "python": list(sys.version_info[:3]),
                    "packages": {name: package_version(name) for name in ("open_clip_torch", "timm", "tokenizers", "transformers")}},
        "math": {"matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
                 "fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                 "fp16_accumulation": torch.backends.cuda.matmul.allow_fp16_accumulation,
                 "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                 "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                 "deterministic": torch.are_deterministic_algorithms_enabled(),
                 "flash_sdpa": torch.backends.cuda.flash_sdp_enabled(),
                 "efficient_sdpa": torch.backends.cuda.mem_efficient_sdp_enabled(),
                 "math_sdpa": torch.backends.cuda.math_sdp_enabled(),
                 "cudnn_sdpa": torch.backends.cuda.cudnn_sdp_enabled()},
    }


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
