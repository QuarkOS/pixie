"""Native 32x32 training with exact resume or an explicit full-state branch."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.v3_model import ClipTextPenultimate
from train.pixie32.data import load_dataset
from train.pixie32.p32_model import P32, P32Config


class InputError(ValueError):
    """Operator-facing validation messages that contain no supplied paths."""


@dataclass(frozen=True)
class Recipe:
    batch: int
    seed: int
    dataset: str
    model: dict
    images: int
    warmup: int
    decay_images: int
    lr: float
    lr_min: float
    clip_sha256: str = ""
    tokenizer_sha256: str = ""
    device_type: str = "cpu"
    ce_chunks: int = 16
    gradient_checkpointing: bool = True
    save_every_images: int = 100_000
    eval_every_images: int = 100_000
    sample_every_images: int = 100_000
    schedule: str = "wsd"
    weight_decay: float = 0.05
    alpha_weight: float = 0.25
    caption_dropout: float = 0.1
    subject_caption: float = 0.15
    flip: float = 0.5
    pal_top1: float = 0.30
    pal_top3: float = 0.50
    style: float = 0.70
    compile: bool = False
    clip_cache_sha256: str = ""


def fingerprint(recipe: Recipe) -> dict:
    fields = asdict(recipe)
    if not recipe.compile:
        fields.pop("compile")
    if not recipe.clip_cache_sha256:
        fields.pop("clip_cache_sha256")
    return {"fields": fields, "sha256": hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def check_resume(saved: dict, expected: dict) -> None:
    if saved != expected:
        old, new = saved.get("fields", {}), expected["fields"]
        changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
        raise InputError("resume fingerprint mismatch in " + ", ".join(changed or ["digest"]))


BRANCH_CHANGES = frozenset({"batch", "clip_cache_sha256", "compile", "ce_chunks",
                            "gradient_checkpointing", "save_every_images", "eval_every_images",
                            "sample_every_images"})


def check_branch(saved: dict, expected: dict, optimizer, model) -> dict:
    """Validate full-state continuation before allowing a new batch/cache recipe."""
    if not isinstance(saved, dict) or saved.get("format") != "pixie32-full-v1":
        raise InputError("branch-from requires a full Pixie32 checkpoint")
    required = {"config", "model", "optimizer", "fingerprint", "rng", "step", "images_seen",
                "epoch", "offset", "best_iid_nll", "next_save", "next_eval", "next_sample"}
    if not required <= saved.keys():
        raise InputError("branch-from requires a complete full Pixie32 checkpoint")
    old = saved.get("fingerprint", {})
    try:
        source = Recipe(**old["fields"])
    except (KeyError, TypeError):
        raise InputError("invalid branch source recipe") from None
    if fingerprint(source) != old or saved.get("config") != source.model:
        raise InputError("invalid branch source fingerprint or config")
    new_fields = expected["fields"]
    changes = {key: {"from": old["fields"].get(key), "to": new_fields.get(key)}
               for key in old["fields"].keys() | new_fields.keys()
               if old["fields"].get(key) != new_fields.get(key)}
    forbidden = changes.keys() - BRANCH_CHANGES
    if forbidden:
        raise InputError("branch-from cannot change " + ", ".join(sorted(forbidden)))
    if saved["config"] != model.config.to_dict() or list(saved["model"]) != list(model.state_dict()):
        raise InputError("branch-from model config or tensor keys differ")
    for key, value in model.state_dict().items():
        parent_value = saved["model"][key]
        if not isinstance(parent_value, torch.Tensor) or parent_value.shape != value.shape:
            raise InputError("branch-from model tensor shapes differ")
    step, images = saved.get("step"), saved.get("images_seen")
    if (type(step) is not int or type(images) is not int or step < 0 or images < 0
            or (step == 0) != (images == 0) or step > images or images >= source.images):
        raise InputError("invalid or completed branch source progress")
    if not isinstance(saved.get("rng"), dict) or not {"torch", "numpy", "python"} <= saved["rng"].keys():
        raise InputError("branch-from requires full RNG state")
    best = saved["best_iid_nll"]
    if type(best) not in (int, float) or math.isnan(best) or best == -float("inf"):
        raise InputError("invalid branch source evaluation state")
    state = saved.get("optimizer", {})
    groups = state.get("param_groups", [])
    current = optimizer.state_dict()["param_groups"]
    if not isinstance(state.get("state"), dict) or len(groups) != len(current):
        raise InputError("branch-from requires matching AdamW state")
    parameters = list(model.parameters())
    parent_ids = []
    for old_group, new_group in zip(groups, current):
        if not isinstance(old_group, dict) or set(old_group) != set(new_group):
            raise InputError("branch-from AdamW parameter groups differ")
        if any(old_group[key] != value for key, value in new_group.items() if key not in {"params", "lr"}):
            raise InputError("branch-from AdamW hyperparameters differ")
        if old_group["params"] != new_group["params"]:
            raise InputError("branch-from AdamW parameter order differs")
        expected_lr = lr_at(step - 1, images, source) if step else source.lr
        if not math.isclose(old_group["lr"], expected_lr, rel_tol=1e-12, abs_tol=0):
            raise InputError("branch-from AdamW learning rate differs from saved progress")
        parent_ids.extend(old_group["params"])
    if len(parent_ids) != len(parameters) or len(set(parent_ids)) != len(parent_ids):
        raise InputError("branch-from AdamW parameter count differs")
    if set(state["state"]) != (set(parent_ids) if step else set()):
        raise InputError("branch-from AdamW parameter state is incomplete")
    for number, parameter in zip(parent_ids, parameters):
        if not step:
            break
        moments = state["state"][number]
        if set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
            raise InputError("branch-from AdamW moments are incomplete")
        if not isinstance(moments["step"], torch.Tensor) or moments["step"].numel() != 1 or moments["step"].item() != step:
            raise InputError("branch-from AdamW step differs from global step")
        for key in ("exp_avg", "exp_avg_sq"):
            value = moments[key]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or not torch.isfinite(value).all():
                raise InputError("branch-from AdamW moment shapes or values differ")
    return dict(sorted(changes.items()))


def lr_at(step: int, images_seen: int, recipe: Recipe) -> float:
    if step < recipe.warmup:
        return recipe.lr * (step + 1) / recipe.warmup
    decay_start = recipe.images - recipe.decay_images
    if recipe.decay_images and images_seen >= decay_start:
        fraction = min(1.0, (images_seen - decay_start) / recipe.decay_images)
        return recipe.lr + fraction * (recipe.lr_min - recipe.lr)
    return recipe.lr


def epoch_plan(indices: torch.Tensor, weights: torch.Tensor, batch: int,
               seed: int, epoch: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003 * epoch)
    count = math.ceil(len(indices) / batch) * batch
    choice = torch.multinomial(weights.double(), count, replacement=True, generator=generator)
    return {"rows": indices[choice], "random": torch.rand(count, 5, generator=generator)}


def capture_rng(device: torch.device) -> dict:
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
             "python": random.getstate()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state: dict, device: torch.device) -> None:
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if device.type == "cuda":
        if "cuda" not in state:
            raise InputError("resume device RNG differs from checkpoint")
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)
    elif "cuda" in state:
        raise InputError("resume device RNG differs from checkpoint")


def atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_json(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def empty_text(batch: int, length: int, device: torch.device) -> torch.Tensor:
    text = torch.full((batch, length), 49407, dtype=torch.long, device=device)
    text[:, 0] = 49406
    return text


def autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def resolve_inputs(a):
    if not a.resume and not a.branch_from and a.warmstart is None:
        value = os.environ.get("V3_CKPT")
        if not value:
            raise InputError("provide --v3-ckpt or V3_CKPT for a warm start")
        a.warmstart = Path(value)
    if a.resume or a.branch_from:
        a.warmstart = None
    if a.tokenizer is None:
        value = os.environ.get("TOKENIZER")
        if not value:
            raise InputError("provide --tokenizer or TOKENIZER")
        a.tokenizer = Path(value)
    if a.clip is None:
        directory = a.clip_dir or os.environ.get("CLIP_DIR")
        if not directory:
            raise InputError("provide --clip-dir or CLIP_DIR")
        a.clip = Path(directory) / "clip_l14_text_l11_f32.pt"
    return a


def load_clip(path: Path, device: torch.device):
    clip = ClipTextPenultimate().to(device)
    clip.load_hf(torch.load(path, map_location="cpu", weights_only=False))
    clip.requires_grad_(False).eval()
    if device.type == "cuda":
        clip.to(torch.bfloat16)
    return clip


def prepared_batch(arrays: dict, plan: dict, start: int, count: int,
                   recipe: Recipe, device: torch.device, *, text_on_cpu=False):
    rows = plan["rows"][start:start + count].numpy()
    draws = plan["random"][start:start + count]
    image = torch.as_tensor(np.asarray(arrays["tokens"][rows]).astype(np.int64), device=device)
    text_device = torch.device("cpu") if text_on_cpu else device
    text = torch.as_tensor(np.asarray(arrays["text"][rows]).astype(np.int64), device=text_device)
    subject = torch.as_tensor(np.asarray(arrays["subj_text"][rows]).astype(np.int64), device=text_device)
    blank = draws[:, 0] < recipe.caption_dropout
    use_subject = (draws[:, 0] >= recipe.caption_dropout) & (
        draws[:, 0] < recipe.caption_dropout + recipe.subject_caption)
    text[use_subject.to(text_device)] = subject[use_subject.to(text_device)]
    text[blank.to(text_device)] = empty_text(int(blank.sum()), text.shape[1], text_device)
    flip = ((draws[:, 1] < recipe.flip).numpy() & ~np.asarray(arrays["noflip"][rows], dtype=bool))
    flip = torch.as_tensor(flip, device=device)
    image[flip] = image[flip].reshape(-1, 32, 32).flip(-1).reshape(-1, 1024)
    cond = torch.as_tensor(np.asarray(arrays["cond"][rows]).astype(np.int64), device=device)
    palette_draw = draws[:, 2].to(device)
    cond[palette_draw >= recipe.pal_top1 + recipe.pal_top3, :3] = 4097
    cond[palette_draw < recipe.pal_top1, 1:3] = 4097
    cond[(draws[:, 3] >= recipe.style).to(device), 3:] = 0
    return text, image, cond


@torch.no_grad()
def evaluate(model, clip, arrays: dict, device: torch.device, batch: int,
             chunks: int, alpha_weight: float, feature_cache=None) -> dict:
    model.eval()
    result = {}
    for split, name in ((1, "iid"), (2, "probe")):
        rows = np.flatnonzero(arrays["split"] == split)
        total, conditioned_total, count = 0.0, 0.0, 0
        for begin in range(0, len(rows), batch):
            chosen = rows[begin:begin + batch]
            image = torch.as_tensor(np.asarray(arrays["tokens"][chosen]).astype(np.int64), device=device)
            text = torch.as_tensor(np.asarray(arrays["text"][chosen]).astype(np.int64),
                                   device="cpu" if feature_cache else device)
            cond = torch.as_tensor(np.asarray(arrays["cond"][chosen]).astype(np.int64), device=device)
            with autocast(device):
                features = feature_cache(text, device) if feature_cache else clip(text)
                _, nll = model.loss_fwd(features, image, alpha_weight, chunks=chunks)
                _, conditioned_nll = model.loss_fwd(features, image, alpha_weight, cond=cond, chunks=chunks)
            total += float(nll) * len(chosen)
            conditioned_total += float(conditioned_nll) * len(chosen)
            count += len(chosen)
        if count:
            result[name + "_nll"] = total / count
            result[name + "_conditioned_nll"] = conditioned_total / count
            result[name + "_count"] = count
    model.train()
    return result


def load_sample_prompts(path, tokenizer, holdouts):
    from train.pixie32.build_data_p32 import encode

    candidates = ["red apple", "wooden chair", "blue teapot", "green backpack",
                  "silver spoon", "clay bowl", "yellow raincoat", "small wooden crate",
                  "iron hammer", "leather boot", "purple mushroom", "stone tower"]
    if path is not None:
        candidates = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    safe = []
    excluded = 0
    for prompt in candidates:
        row, trimmed = encode(tokenizer, prompt)
        if holdouts.matches(prompt, trimmed):
            excluded += 1
        else:
            safe.append((prompt, row))
    if not safe:
        raise InputError("no safe sample prompts after frozen holdout exclusion")
    print(f"sample prompts: admitted={len(safe)} excluded={excluded}", flush=True)
    return safe


@torch.no_grad()
def sample_grid(model, clip, samples, palette, device, out: Path, step: int,
                seed: int, *, scale=4, columns=4, batch=4) -> None:
    from PIL import Image

    columns = min(columns, len(samples))
    grid = Image.new("RGBA", (columns * 32, math.ceil(len(samples) / columns) * 32), (0, 0, 0, 0))
    generator = torch.Generator(device=device).manual_seed(seed + step)
    was_training = model.training
    model.eval()
    try:
        for begin in range(0, len(samples), batch):
            chosen = samples[begin:begin + batch]
            text = torch.as_tensor(np.stack([row for _, row in chosen]).astype(np.int64), device=device)
            with autocast(device):
                result = model.sample(clip(text), clip(empty_text(len(chosen), 18, device)),
                                      cfg=2.0, generator=generator)
            for offset, indices in enumerate(result.cpu().numpy().reshape(-1, 32, 32)):
                rgba = np.concatenate([palette[indices], ((indices != 0) * 255).astype(np.uint8)[..., None]], axis=2)
                cell = begin + offset
                grid.paste(Image.fromarray(rgba), ((cell % columns) * 32, (cell // columns) * 32))
        grid.resize((grid.width * scale, grid.height * scale), Image.Resampling.NEAREST).save(
            out / f"samples_{step:08d}.png")
        write_json({"prompts": [prompt for prompt, _ in samples], "columns": columns,
                    "scale": scale, "batch": batch, "seed": seed + step,
                    "cell_order": "row-major", "native_sprite_size": 32},
                   out / f"samples_{step:08d}.json")
    finally:
        model.train(was_training)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for flag in ("data", "out"):
        p.add_argument("--" + flag, type=Path, required=True)
    holds = Path(os.environ.get("PIXIE_HOLDOUT_DIR") or Path(__file__).resolve().parents[2] / "holdout")
    p.add_argument("--blind50", type=Path, default=holds / "blind-50.json")
    p.add_argument("--eval50", type=Path, default=holds / "eval-50.json")
    p.add_argument("--clip-dir", type=Path)
    p.add_argument("--clip", type=Path, help="compatibility alias for the CLIP weight file")
    p.add_argument("--tokenizer", type=Path)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--v3-ckpt", "--warmstart", dest="warmstart", type=Path)
    source.add_argument("--resume", type=Path)
    source.add_argument("--branch-from", type=Path, help="carry full P32 optimizer/WSD progress into a new batch/cache recipe")
    p.add_argument("--device", required=True, help="cpu or one explicit cuda index, e.g. cuda:0")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--images", type=int, default=12_000_000)
    p.add_argument("--decay-images", type=int, default=2_000_000)
    p.add_argument("--warmup", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ce-chunks", type=int, default=16)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action="store_true", help="compile the training loss with torch.compile (Inductor)")
    p.add_argument("--clip-cache", type=Path, help="verified RAM-resident frozen text features")
    p.add_argument("--clip-cache-max-gb", type=float, default=4, help="maximum feature cache RAM, decimal GB")
    p.add_argument("--save-every-images", type=int, default=100_000)
    p.add_argument("--eval-every-images", type=int, default=100_000)
    p.add_argument("--sample-every-images", type=int, default=100_000)
    p.add_argument("--sample-prompts", type=Path, help="UTF-8 file with one prompt per line")
    p.add_argument("--sample-scale", type=int, default=4, help="nearest-neighbour grid enlargement")
    p.add_argument("--sample-columns", type=int, default=4)
    p.add_argument("--sample-batch", type=int, default=4, help="maximum prompts sampled together, before CFG doubling")
    p.add_argument("--max-hours", type=float)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-steps", type=int, default=200)
    p.add_argument("--profile-steps", type=int, default=0, help="record up to 200 update phase timings; zero disables instrumentation")
    p.add_argument("--profile-warmup", type=int, default=10, help="segment updates excluded before phase profiling")
    return p


def run(a) -> dict:
    a = resolve_inputs(a)
    if a.device != "cpu" and not (a.device.startswith("cuda:") and a.device[5:].isdigit()):
        raise InputError("--device must be cpu or one explicit cuda index")
    device = torch.device(a.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise InputError("requested CUDA device is unavailable")
        torch.cuda.set_device(device)
    positive = (a.batch, a.images, a.ce_chunks, a.save_every_images, a.eval_every_images, a.sample_every_images,
                a.sample_scale, a.sample_columns, a.sample_batch)
    if any(value <= 0 for value in positive) or a.warmup < 0 or not 0 <= a.decay_images <= a.images:
        raise InputError("invalid batch, image budget, warmup, chunk, sample grid or checkpoint interval")
    if a.lr <= 0 or not 0 <= a.lr_min <= a.lr or (a.max_hours is not None and a.max_hours <= 0):
        raise InputError("invalid learning rate or time budget")
    if not 1 <= a.smoke_steps <= 300:
        raise InputError("--smoke-steps must be between 1 and 300")
    if not 0 <= a.profile_steps <= 200 or not 0 <= a.profile_warmup <= 100:
        raise InputError("profile steps must be 0..200 and warmup 0..100")
    if a.out.exists():
        raise InputError("output directory already exists; select a new output folder")
    source_name = "warmstart" if a.warmstart else ("resume" if a.resume else "branch_from")
    for name in ("clip", "tokenizer", source_name):
        if not getattr(a, name).is_file():
            raise InputError("missing required " + name + " input")
    dataset = load_dataset(a.data, a.blind50, a.eval50)
    def file_sha(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    tokenizer_sha = file_sha(a.tokenizer)
    if tokenizer_sha != dataset.manifest["inputs"]["tokenizer"]:
        raise InputError("tokenizer hash differs from the dataset manifest")
    clip_sha = file_sha(a.clip)
    feature_cache = None
    if a.clip_cache:
        from train.pixie32.clip_cache import FeatureCache
        feature_cache = FeatureCache(a.clip_cache, dataset, clip_sha, a.clip_cache_max_gb)
        if feature_cache.compute != ("bf16" if device.type == "cuda" else "float32"):
            raise InputError("CLIP cache compute precision differs; rebuild on the training device type")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    samples = load_sample_prompts(a.sample_prompts, tokenizer, dataset.holdouts)
    arrays = dataset.arrays
    indices = torch.from_numpy(np.flatnonzero(arrays["split"] == 0).copy())
    if not len(indices):
        raise InputError("no training rows after holdout filtering")
    if not np.any(arrays["split"] == 1):
        raise InputError("iid evaluation split is empty")
    weights = torch.from_numpy(np.asarray(arrays["weight"])[indices.numpy()].astype(np.float64))
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise InputError("training weights must be positive and finite")
    print("cache manifest verified; train holdout matches=0", flush=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.random.default_generator.manual_seed(a.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(a.seed)
    config = P32Config(d_model=512, n_layer=16, n_head=8, dropout=0.1, text_len=18, n_pal=3, style=True)
    model = P32(config).to(device)
    model.gradient_checkpointing = a.gradient_checkpointing
    recipe = Recipe(a.batch, a.seed, dataset.fingerprint, config.to_dict(), a.images,
                    a.warmup, a.decay_images, a.lr, a.lr_min,
                    clip_sha256=clip_sha, tokenizer_sha256=tokenizer_sha,
                    device_type=device.type, ce_chunks=a.ce_chunks,
                    gradient_checkpointing=a.gradient_checkpointing,
                    save_every_images=a.save_every_images, eval_every_images=a.eval_every_images,
                    sample_every_images=a.sample_every_images, compile=a.compile,
                    clip_cache_sha256=feature_cache.fingerprint if feature_cache else "")
    expected = fingerprint(recipe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=recipe.weight_decay)
    step, images_seen, best, epoch, offset = 0, 0, float("inf"), 0, 0
    segment_best = float("inf")
    next_save, next_eval, next_sample = a.save_every_images, a.eval_every_images, a.sample_every_images
    resumed = None
    branch = None
    if a.resume:
        resumed = torch.load(a.resume, map_location="cpu", weights_only=False)
        if resumed.get("format") != "pixie32-full-v1":
            raise InputError("resume requires a full Pixie32 checkpoint")
        check_resume(resumed["fingerprint"], expected)
        model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        step, images_seen = resumed["step"], resumed["images_seen"]
        epoch, offset, best = resumed["epoch"], resumed["offset"], resumed["best_iid_nll"]
        next_save, next_eval, next_sample = resumed["next_save"], resumed["next_eval"], resumed["next_sample"]
        branch = resumed.get("branch")
    elif a.branch_from:
        resumed = torch.load(a.branch_from, map_location="cpu", weights_only=False)
        changes = check_branch(resumed, expected, optimizer, model)
        model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        step, images_seen = resumed["step"], resumed["images_seen"]
        best = resumed["best_iid_nll"]
        next_save, next_eval, next_sample = tuple((images_seen // interval + 1) * interval
                                                 for interval in (a.save_every_images, a.eval_every_images, a.sample_every_images))
        branch = {"parent_sha256": file_sha(a.branch_from), "parent_step": step,
                  "parent_images_seen": images_seen, "source_fingerprint": resumed["fingerprint"]["sha256"],
                  "target_fingerprint": expected["sha256"], "changes": changes,
                  "sampler": {"epoch": 0, "offset": 0, "seed": a.seed},
                  "optimizer": "carried", "schedule": "global step and images preserved"}
        print("branch-from " + json.dumps(branch, sort_keys=True), flush=True)
    else:
        report = model.load_v3_warmstart(a.warmstart)
        print("warmstart " + json.dumps(report, sort_keys=True), flush=True)
    clip = load_clip(a.clip, device)
    train_loss = torch.compile(model.loss_fwd) if a.compile else model.loss_fwd
    if resumed is not None:
        restore_rng(resumed["rng"], device)
        del resumed
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "worker.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    (a.out / "launch.txt").write_text(
        "$PY -m train.pixie32.train_p32 --data $DATASET_OUT --blind50 $BLIND50 "
        "--eval50 $EVAL50 --clip-dir $CLIP_DIR --tokenizer $TOKENIZER --out $RUNS/NEW "
        + ("--resume $RESUME" if a.resume else ("--branch-from $P32_CKPT" if a.branch_from else "--v3-ckpt $V3_CKPT"))
        + f" --device {a.device} --batch {a.batch} --images {a.images}"
        + f" --seed {a.seed} --warmup {a.warmup} --decay-images {a.decay_images}"
        + f" --lr {a.lr} --lr-min {a.lr_min} --ce-chunks {a.ce_chunks}"
        + (" --gradient-checkpointing" if a.gradient_checkpointing else " --no-gradient-checkpointing")
        + f" --save-every-images {a.save_every_images} --eval-every-images {a.eval_every_images}"
        + f" --sample-every-images {a.sample_every_images}"
        + f" --sample-scale {a.sample_scale} --sample-columns {a.sample_columns} --sample-batch {a.sample_batch}"
        + (" --sample-prompts $SAMPLE_PROMPTS" if a.sample_prompts else "")
        + (" --compile" if a.compile else "")
        + (f" --clip-cache $CLIP_CACHE --clip-cache-max-gb {a.clip_cache_max_gb}" if feature_cache else "")
        + (f" --smoke --smoke-steps {a.smoke_steps}" if a.smoke else "")
        + (f" --profile-steps {a.profile_steps} --profile-warmup {a.profile_warmup}" if a.profile_steps else "")
        + (f" --max-hours {a.max_hours}" if a.max_hours else "") + "\n", encoding="utf-8")
    write_json(expected, a.out / "recipe.json")
    if branch is not None:
        write_json(branch, a.out / "branch.json")
    write_json({"prompts": [prompt for prompt, _ in samples], "scale": a.sample_scale,
                "columns": a.sample_columns, "batch": a.sample_batch}, a.out / "sampling.json")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    step_profile = None
    if a.profile_steps:
        from train.pixie32.step_profile import StepProfile
        step_profile = StepProfile(device, a.profile_steps, a.profile_warmup)
    started = time.perf_counter()
    timed_steps, timed_images, training_seconds, last_loss = 0, 0, 0.0, None
    phase_seconds = {name: 0.0 for name in ("training", "evaluation", "sampling", "checkpoint")}
    phase_counts = dict.fromkeys(phase_seconds, 0)
    last_evaluated_step = last_sampled_step = None
    reason = "complete"
    metrics = {}
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    previous_handler = signal.signal(signal.SIGINT, request_stop)

    @contextmanager
    def phase(name):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        began = time.perf_counter()
        try:
            yield
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            phase_seconds[name] += time.perf_counter() - began
            phase_counts[name] += 1

    def save(value, path):
        with phase("checkpoint"):
            atomic_save(value, path)

    def checkpoint():
        save({"format": "pixie32-full-v1", "model": model.state_dict(),
                     "config": config.to_dict(), "optimizer": optimizer.state_dict(),
                     "fingerprint": expected, "rng": capture_rng(device), "step": step,
                     "images_seen": images_seen, "epoch": epoch, "offset": offset,
                     "best_iid_nll": best, "segment_best_iid_nll": segment_best,
                     "next_save": next_save, "next_eval": next_eval,
                     "next_sample": next_sample, **({"branch": branch} if branch is not None else {})}, a.out / "last.pt")

    def report_evaluation():
        nonlocal best, segment_best, last_evaluated_step
        with phase("evaluation"):
            result = evaluate(model, clip, arrays, device, a.batch, a.ce_chunks, recipe.alpha_weight, feature_cache)
        last_evaluated_step = step
        best = min(best, result["iid_nll"])
        if result["iid_nll"] < segment_best:
            segment_best = result["iid_nll"]
            save({"model": model.state_dict(), "config": config.to_dict(),
                  "step": step, "images_seen": images_seen, "iid_nll": segment_best}, a.out / "best.pt")
        return result

    def report_sample():
        nonlocal last_sampled_step
        with phase("sampling"):
            sample_grid(model, clip, samples, dataset.palette, device, a.out, step, a.seed,
                        scale=a.sample_scale, columns=a.sample_columns, batch=a.sample_batch)
        last_sampled_step = step

    try:
        plan = epoch_plan(indices, weights, a.batch, a.seed, epoch)
        model.train()
        while images_seen < a.images:
            if stop_requested:
                reason = "interrupt"
                break
            if a.smoke and timed_steps >= a.smoke_steps:
                reason = "smoke"
                break
            if a.max_hours and time.perf_counter() - started >= a.max_hours * 3600:
                reason = "max_hours"
                break
            if offset == len(plan["rows"]):
                epoch += 1
                offset = 0
                plan = epoch_plan(indices, weights, a.batch, a.seed, epoch)
            count = min(a.batch, a.images - images_seen)
            began = time.perf_counter()
            if step_profile:
                step_profile.begin(timed_steps, count)
            text, image, cond = prepared_batch(arrays, plan, offset, count, recipe, device,
                                               text_on_cpu=feature_cache is not None)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, images_seen + count, recipe)
            optimizer.zero_grad(set_to_none=True)
            if step_profile:
                step_profile.mark()
            with torch.no_grad(), autocast(device):
                features = feature_cache(text, device) if feature_cache else clip(text)
            if step_profile:
                step_profile.mark()
            with autocast(device):
                loss, _ = train_loss(features, image, recipe.alpha_weight, cond=cond, chunks=a.ce_chunks)
            if not torch.isfinite(loss):
                raise InputError("non-finite training loss")
            if step_profile:
                step_profile.mark()
            loss.backward()
            if step_profile:
                step_profile.mark()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step_profile:
                step_profile.mark()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if step_profile:
                step_profile.finish()
            training_seconds += time.perf_counter() - began
            phase_seconds["training"] = training_seconds
            phase_counts["training"] += 1
            timed_steps += 1
            timed_images += count
            last_loss = float(loss.detach())
            step += 1
            offset += count
            images_seen += count
            if timed_steps == 1 or timed_steps % 10 == 0:
                print(f"step={step} images={images_seen} loss={last_loss:.6f} step/s={timed_steps / training_seconds:.3f}", flush=True)
            if images_seen >= next_eval:
                metrics = report_evaluation()
                while next_eval <= images_seen:
                    next_eval += a.eval_every_images
                print("evaluation " + json.dumps(metrics, sort_keys=True), flush=True)
            if images_seen >= next_sample:
                report_sample()
                while next_sample <= images_seen:
                    next_sample += a.sample_every_images
            if images_seen >= next_save:
                save({"model": model.state_dict(), "config": config.to_dict(), "step": step,
                             "images_seen": images_seen}, a.out / f"img_{images_seen:09d}.pt")
                while next_save <= images_seen:
                    next_save += a.save_every_images
                checkpoint()
        checkpoint()
        if reason in ("complete", "smoke"):
            if last_evaluated_step != step:
                metrics = report_evaluation()
            if last_sampled_step != step:
                report_sample()
            checkpoint()
        if stop_requested:
            reason = "interrupt"
        wall_seconds = time.perf_counter() - started
        if step_profile:
            write_json(step_profile.report(), a.out / "step-profile.json")
        result = {"status": reason, "step": step, "images_seen": images_seen, "measured_steps": timed_steps,
                  "loss": last_loss, "training_seconds": training_seconds,
                  "segment_images": timed_images, "segment_wall_seconds": wall_seconds,
                  "wall_images_per_second": timed_images / wall_seconds,
                  "phase_seconds": phase_seconds, "phase_counts": phase_counts,
                  "unclassified_seconds": wall_seconds - sum(phase_seconds.values()),
                  "wall_scope": "segment after asset loading; includes first-step compile, epoch plans, reporting and checkpoint writes; excludes summary and step-profile report writes",
                  "global_best_iid_nll": best if math.isfinite(best) else None,
                  "segment_best_iid_nll": segment_best if math.isfinite(segment_best) else None,
                  "steps_per_second": timed_steps / training_seconds if training_seconds else None,
                  "images_per_second": timed_images / training_seconds if training_seconds else None,
                  "throughput_scope": ("cached CLIP transfer" if feature_cache else "online CLIP")
                                      + ", augmentation and completed training updates",
                  "peak_vram_scope": "training, evaluation and sampling",
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
                  "device_type": device.type, **metrics}
        write_json(result, a.out / ("train_summary.json" if reason == "complete" else "paused.json"))
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        (a.out / "worker.pid").unlink(missing_ok=True)


def main() -> int:
    a = parser().parse_args()
    try:
        run(a)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
