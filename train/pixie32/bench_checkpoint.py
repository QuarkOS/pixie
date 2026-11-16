"""Measure a periodic model snapshot plus full recovery write, without training."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import statistics
import time

import torch

from train.pixie32.clip_cache import idle_device
from train.pixie32.data import sha256
from train.pixie32.train_p32 import atomic_save


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(map(tensor_bytes, value))
    return 0


def on_device(saved, device):
    """Move the state that the current non-fused trainer keeps on the GPU."""
    state = copy.copy(saved)
    state["model"] = {key: value.to(device) for key, value in saved["model"].items()}
    state["optimizer"] = copy.deepcopy(saved["optimizer"])
    for value in state["optimizer"]["state"].values():
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in value:
                value[key] = value[key].to(device)
    return state


def durable_file(path):
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def same_tensors(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_tensors(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same_tensors(a, b) for a, b in zip(left, right))
    return left == right


def benchmark(saved, out, *, trials, device):
    if saved.get("format") != "pixie32-full-v1" or not {"model", "optimizer", "config", "step", "images_seen", "rng"} <= saved.keys():
        raise ValueError("supply a full P32 recovery checkpoint")
    if not saved["optimizer"].get("state"):
        raise ValueError("checkpoint must contain initialized optimizer moments")
    for group in saved["optimizer"]["param_groups"]:
        if group.get("fused") or group.get("capturable"):
            raise ValueError("this measurement mirrors the current non-fused, non-capturable trainer")
    snapshot = {key: saved[key] for key in ("model", "config", "step", "images_seen")}
    required = int((tensor_bytes(saved) + tensor_bytes(snapshot)) * trials * 1.05) + 8_000_000
    if shutil.disk_usage(out).free < required:
        raise ValueError("insufficient free space for all fresh checkpoint pairs")
    rows = []
    for trial in range(trials):
        folder = out / f"trial-{trial:02d}"
        folder.mkdir(exist_ok=False)
        row = {"trial": trial}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        began = time.perf_counter()
        for name, value in (("model", snapshot), ("full", saved)):
            start = time.perf_counter()
            atomic_save(value, folder / f"{name}.pt")
            row[name + "_write_seconds"] = time.perf_counter() - start
        row["cycle_write_seconds"] = time.perf_counter() - began
        start = time.perf_counter()
        for name in ("model", "full"):
            durable_file(folder / f"{name}.pt")
        row["post_write_file_fsync_seconds"] = time.perf_counter() - start
        row["cycle_write_plus_file_fsync_seconds"] = row["cycle_write_seconds"] + row["post_write_file_fsync_seconds"]
        row["bytes"] = {name: (folder / f"{name}.pt").stat().st_size for name in ("model", "full")}
        full = torch.load(folder / "full.pt", map_location="cpu", weights_only=False)
        model = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
        row["roundtrip_exact"] = same_tensors(saved["model"], full["model"]) and same_tensors(saved["optimizer"], full["optimizer"]) and same_tensors(snapshot["model"], model["model"]) and full["step"] == saved["step"] and full["images_seen"] == saved["images_seen"]
        del full, model
        rows.append(row)
    keys = ("model_write_seconds", "full_write_seconds", "cycle_write_seconds", "post_write_file_fsync_seconds", "cycle_write_plus_file_fsync_seconds")
    return {"comparison_valid": all(row["roundtrip_exact"] for row in rows), "trials": rows,
            "summary": {key: {"median": statistics.median(row[key] for row in rows),
                              "range": [min(row[key] for row in rows), max(row[key] for row in rows)]} for key in keys},
            "minimum_output_space_bytes": required, "device_type": device.type,
            "scope": "resident state -> torch.save temporary file -> os.replace, as current trainer; source load and verification excluded. CUDA serialization includes device-to-host copies. CPU mode measures CPU serialization only.",
            "durability": "Trainer write return can use OS page cache. Separate post-write file fsync flushes both files; directory metadata is not fsynced. This is not a power-loss recovery guarantee. Later trials see storage/page-cache pressure from earlier writes."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True, help="cpu, or an explicit idle GPU such as cuda:0")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or not args.checkpoint.is_file() or not 5 <= args.trials <= 10:
        parser.error("supply a full checkpoint, 5..10 trials, and a new output folder")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and device.index is None):
        parser.error("select cpu or exactly one explicit CUDA device")
    idle_device(args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        source_digest = sha256(args.checkpoint)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        saved = on_device(saved, device)
        result = benchmark(saved, args.out, trials=args.trials, device=device)
        if sha256(args.checkpoint) != source_digest:
            raise ValueError("source checkpoint changed during measurement")
        result["checkpoint_sha256"] = source_digest
    except Exception as error:
        result = {"comparison_valid": False, "error_type": type(error).__name__}
    (args.out / "checkpoint-benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))
    return 0 if result["comparison_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
