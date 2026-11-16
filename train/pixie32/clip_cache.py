"""Verified, RAM-resident frozen CLIP features, keyed by complete token rows."""
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess

import numpy as np
import torch

from train.pixie32.data import sha256


def idle_device(value):
    """Refuse a busy explicitly selected GPU before initializing CUDA."""
    if value == "cpu":
        return torch.device(value)
    if not value.startswith("cuda:") or not value[5:].isdigit():
        raise ValueError("device must be cpu or one explicit cuda index")
    # nvidia-smi indexes physical GPUs; refuse remapping to avoid checking the wrong GPU.
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("benchmark/cache builder requires an unmapped explicit GPU index")
    try:
        output = subprocess.check_output(["nvidia-smi", "-i", value[5:],
                                          "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                         text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("could not verify that selected GPU is idle") from error
    if any(line.strip() and line.strip() != str(os.getpid()) for line in output.splitlines()):
        raise ValueError("selected GPU is busy; benchmark/cache build refused")
    return torch.device(value)


def token_rows(dataset):
    blank = np.full((1, 18), 49407, dtype=np.uint16)
    blank[:, 0] = 49406
    rows = np.concatenate((dataset.arrays["text"], dataset.arrays["subj_text"], blank))
    if rows.shape[1:] != (18,) or rows.dtype != np.uint16 or np.any(rows >= 49408):
        raise ValueError("CLIP cache requires uint16 token rows of length 18 in vocabulary")
    return np.unique(rows, axis=0)


def identity(dataset, clip_sha256):
    return {"dataset": dataset.fingerprint,
            "ids_sha256": hashlib.sha256("\n".join(dataset.ids).encode()).hexdigest(),
            "tokenizer": dataset.manifest["inputs"]["tokenizer"], "clip": clip_sha256}


def check_budget(count, max_gb):
    # Include the temporary gathered batch/token index overhead conservatively.
    required = count * (18 * 768 * 2 + 18 * 2 + 256)
    if not math.isfinite(max_gb) or max_gb <= 0 or required > max_gb * 1e9:
        raise ValueError("CLIP cache exceeds --clip-cache-max-gb RAM budget")
    return required


@torch.no_grad()
def build_cache(dataset, clip, clip_sha256, out, device, *, batch=128, max_gb=4):
    from train.pixie32.train_p32 import autocast

    if out.exists():
        raise ValueError("CLIP cache output already exists; select a new folder")
    if batch <= 0:
        raise ValueError("CLIP cache batch must be positive")
    rows = token_rows(dataset)
    check_budget(len(rows), max_gb)
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "tokens.npy", rows, allow_pickle=False)
    features = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=np.float16,
                                        shape=(len(rows), 18, 768))
    storage_error = 0.0
    for start in range(0, len(rows), batch):
        text = torch.as_tensor(rows[start:start + batch].astype(np.int64), device=device)
        with autocast(device):
            encoded = clip(text)
        if encoded.shape != (len(text), 18, 768) or not torch.isfinite(encoded).all() or encoded.abs().max() > 65504:
            raise ValueError("CLIP produced invalid or out-of-range features")
        original = encoded.float().cpu().numpy()
        stored = original.astype(np.float16)
        storage_error = max(storage_error, float(np.abs(original - stored.astype(np.float32)).max()))
        features[start:start + len(text)] = stored
    features.flush()
    del features
    manifest = {"schema": "pixie32-clip-v1", "identity": identity(dataset, clip_sha256),
                "rows": len(rows), "compute": "bf16" if device.type == "cuda" else "float32",
                "storage_max_abs_error": storage_error,
                "files": {name: sha256(out / name) for name in ("tokens.npy", "features.npy")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class FeatureCache:
    def __init__(self, root, dataset, clip_sha256, max_gb=4):
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != "pixie32-clip-v1" or manifest.get("identity") != identity(dataset, clip_sha256):
            raise ValueError("CLIP cache identity differs from dataset, tokenizer or CLIP weights")
        count = manifest.get("rows")
        if type(count) is not int or count <= 0:
            raise ValueError("invalid CLIP cache row count")
        self.resident_bytes = check_budget(count, max_gb)
        if set(manifest.get("files", {})) != {"tokens.npy", "features.npy"}:
            raise ValueError("invalid CLIP cache file manifest")
        for name, expected in manifest["files"].items():
            if sha256(root / name) != expected:
                raise ValueError("CLIP cache file digest mismatch")
        rows = np.load(root / "tokens.npy", mmap_mode="r", allow_pickle=False)
        features = np.load(root / "features.npy", mmap_mode="r", allow_pickle=False)
        if rows.dtype != np.uint16 or rows.shape != (count, 18) or features.dtype != np.float16 or features.shape != (count, 18, 768):
            raise ValueError("invalid CLIP cache array shape or dtype")
        expected = token_rows(dataset)
        # Mapping is by tokens: a reordered cache remains valid if its keys/features agree.
        keys = [row.tobytes() for row in rows]
        if len(set(keys)) != count or set(keys) != {row.tobytes() for row in expected}:
            raise ValueError("CLIP cache token keys do not cover dataset captions, subjects and blank")
        self.index = {key: i for i, key in enumerate(keys)}
        self.features = np.array(features)  # One sequential disk read; no random I/O in training.
        for start in range(0, count, 128):
            if not np.isfinite(self.features[start:start + 128]).all():
                raise ValueError("non-finite CLIP cache features")
        self.compute = manifest.get("compute")
        if self.compute not in ("bf16", "float32"):
            raise ValueError("unsupported CLIP cache compute precision")
        self.fingerprint = sha256(root / "manifest.json")

    def __call__(self, text, device):
        if text.device.type != "cpu":
            raise ValueError("CLIP cache lookup expects CPU token rows")
        rows = text.numpy()
        if rows.ndim != 2 or rows.shape[1] != 18 or np.any(rows < 0) or np.any(rows >= 49408):
            raise ValueError("invalid CLIP cache lookup token row")
        try:
            indices = [self.index[row.tobytes()] for row in rows.astype(np.uint16)]
        except KeyError as error:
            raise ValueError("caption token row is missing from CLIP cache") from error
        return torch.from_numpy(self.features[indices]).to(
            device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
