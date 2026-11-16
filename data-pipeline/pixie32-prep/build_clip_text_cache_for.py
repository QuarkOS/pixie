"""Build CLIP-L penultimate text cache for data32 in the v3 s7 format.

Writes a t18 CLIP text cache under --data/cache/ without touching img_idx.npy, runs/,
or any v3 (model) cache. Pass --data $PIXIE_ROOT/data32/out_v3 for the recaption set. Uses the same encoder, weights, penultimate layer, token
length, and bin layout the v3 trainer loads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

HOME = Path.home()
V3_CODE = Path(os.path.expandvars("$PIXIE_ROOT")) / "v3/code"
sys.path.insert(0, str(V3_CODE))
import s7_data as D  # noqa: E402
import v3_model as M  # noqa: E402

BOS, EOS = 49406, 49407
L26, TEXT_IDS, TEXT_LEN = 26, 24, 18
CLIP_PT = Path(os.path.expandvars("$PIXIE_ROOT")) / "v3/clip/clip_l14_text_l11_f32.pt"
TOK = Path(os.path.expandvars("$PIXIE_ROOT")) / "v3/tok/tokenizer.json"
META = Path(os.path.expandvars("$PIXIE_ROOT")) / "data32/out_v2/meta.jsonl"
IMG_IDX = Path(os.path.expandvars("$PIXIE_ROOT")) / "data32/out_v2/cache/img_idx.npy"
OUT = Path(os.path.expandvars("$PIXIE_ROOT")) / "data32/out_v2/cache"
# Keep total GPU use well under 11 GB so pixie-chat stays up.
MAX_TOTAL_MIB = 9000
EST_CLIP_MIB = 1500


def gpu_mem():
    used, total, free = [
        int(x.strip())
        for x in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
             "--format=csv,nounits,noheader"],
            text=True,
        ).strip().split(",")
    ]
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"],
        text=True,
    ).strip()
    return used, total, free, apps


def encode_t26(tok, strings):
    out = np.full((len(strings), L26), EOS, np.uint16)
    raw_len = np.zeros(len(strings), np.int32)
    for k, e in enumerate(tok.encode_batch(list(strings), add_special_tokens=False)):
        ids = e.ids[:TEXT_IDS]
        out[k, 0] = BOS
        out[k, 1:1 + len(ids)] = ids
        out[k, 1 + len(ids)] = EOS
        raw_len[k] = len(e.ids)
    return out, raw_len


def load_captions(path):
    caps = []
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            cap = r.get("caption") or ""
            if not isinstance(cap, str):
                cap = str(cap)
            if int(r.get("idx", i)) != i:
                raise SystemExit(f"meta idx mismatch at line {i}: {r.get('idx')}")
            caps.append(cap)
    return caps


def pick_device(force_cpu: bool):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu"), False
    used, total, free, apps = gpu_mem()
    print(f"gpu before: used={used} MiB total={total} free={free}\napps:\n{apps}", flush=True)
    if "worker_v3" not in apps:
        print("WARNING: pixie-chat worker not visible in nvidia-smi", flush=True)
    if used + EST_CLIP_MIB < MAX_TOTAL_MIB and free > 2500:
        return torch.device("cuda"), True
    print("GPU headroom too tight for CLIP alongside chat; using CPU", flush=True)
    return torch.device("cpu"), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--data", type=Path, default=None, help="dataset dir (default: data32/out_v2)")
    a = ap.parse_args()
    data = a.data or (Path(os.path.expandvars("$PIXIE_ROOT")) / "data32/out_v2")
    meta_path = data / "meta.jsonl"
    img_idx_path = data / "cache" / "img_idx.npy"
    out_dir = data / "cache"
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    img_idx = np.load(img_idx_path, mmap_mode="r")
    caps = load_captions(meta_path)
    N = len(caps)
    if img_idx.shape[0] != N:
        raise SystemExit(f"img_idx {img_idx.shape} vs meta {N}")
    print(f"rows {N} img_idx {img_idx.shape} {img_idx.dtype}", flush=True)

    tok = Tokenizer.from_file(str(TOK))
    t26, raw_len = encode_t26(tok, caps)
    t18 = D.trim(t26, TEXT_LEN)
    blank = D.blank_caption(TEXT_LEN)
    allc = np.concatenate([t18, blank])
    v = allc.view(np.dtype((np.void, 2 * TEXT_LEN))).ravel()
    uniq, inv = np.unique(v, return_inverse=True)
    table = uniq.view(np.uint16).reshape(-1, TEXT_LEN)
    inv = inv.astype(np.int32).ravel()
    row_main = inv[:N]
    blank_i = int(inv[N])
    sha = hashlib.sha256(table.tobytes()).hexdigest()
    fit = float((D.content_len(t26) <= TEXT_LEN - 2).mean())
    over24 = float((raw_len > TEXT_IDS).mean())
    n_empty = int(sum(1 for c in caps if not c.strip()))
    C = len(table)
    L = TEXT_LEN - 1
    need = C * L * 768 * 2
    free = shutil.disk_usage(out_dir).free
    print(
        f"unique captions {C} (blank={blank_i}) need {need / 1e9:.2f} GB "
        f"free {free / 1e9:.1f} GB fit_16ids {fit:.4f} truncated_>24 {over24:.4f} empty {n_empty}",
        flush=True,
    )
    if need >= free - 8e9:
        raise SystemExit("not enough free disk (keeps 8 GB headroom)")

    np.save(out_dir / "captions_t18.npy", table)
    np.save(out_dir / "row_main.npy", row_main)

    fpath = out_dir / "feats_t18_f16.bin"
    tmp = out_dir / "feats_t18_f16.bin.tmp"
    if tmp.exists():
        tmp.unlink()

    dev, use_cuda = pick_device(a.force_cpu)
    print(f"device {dev} batch {a.batch}", flush=True)

    clip = M.ClipTextPenultimate()
    sd = torch.load(CLIP_PT, map_location="cpu", weights_only=False)
    clip.load_hf(sd)
    del sd
    clip.to(dev)
    if use_cuda:
        clip.to(torch.bfloat16)
    clip.eval()
    if use_cuda:
        torch.cuda.synchronize()
        used, total, free_m, apps = gpu_mem()
        print(f"gpu after clip load: used={used} MiB free={free_m}\napps:\n{apps}", flush=True)
        if used > MAX_TOTAL_MIB:
            print("VRAM too high after load, falling back to CPU", flush=True)
            clip.to(torch.device("cpu")).to(torch.float32)
            dev = torch.device("cpu")
            use_cuda = False

    q = queue.Queue(maxsize=4)

    def writer():
        with open(tmp, "wb", buffering=0) as f:
            while True:
                x = q.get()
                if x is None:
                    break
                f.write(memoryview(x).cast("B"))
            os.fsync(f.fileno())

    th = threading.Thread(target=writer, daemon=True)
    th.start()

    amax = 0.0
    bos = None
    t1 = time.time()
    ac = torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else torch.autocast("cpu", enabled=False)
    with torch.no_grad(), ac:
        for s in range(0, C, a.batch):
            ids = torch.from_numpy(table[s:s + a.batch].astype(np.int64)).to(dev)
            h = clip(ids)
            if bos is None:
                bos = h[0, 0].float().cpu().numpy()
            if not torch.equal(h[:, 0].float(), h[:1, 0].float().expand_as(h[:, 0])):
                raise SystemExit("BOS position not constant")
            amax = max(amax, float(h.abs().max()))
            hf = h[:, 1:].to(torch.float16)
            if not torch.isfinite(hf).all():
                raise SystemExit("non-finite features")
            q.put(np.ascontiguousarray(hf.cpu().numpy()))
            n = s + ids.shape[0]
            if (s // a.batch) % 20 == 0 or n == C:
                el = time.time() - t1
                print(f"{n}/{C} {el:.0f}s ({n / max(el, 1e-9):.0f} cap/s) absmax {amax:.1f}", flush=True)
                if use_cuda and s == 0:
                    used, total, free_m, apps = gpu_mem()
                    print(f"gpu after first batch: used={used} MiB free={free_m}", flush=True)
                    if used > MAX_TOTAL_MIB:
                        raise SystemExit(f"VRAM {used} exceeded cap {MAX_TOTAL_MIB}; aborting to protect chat")
    q.put(None)
    th.join()
    os.replace(tmp, fpath)
    np.save(out_dir / "bos_feat.npy", bos)

    meta = {
        "captions": int(C),
        "table_rows": int(C),
        "images": int(N),
        "text_len": TEXT_LEN,
        "stored_positions": L,
        "dim": 768,
        "dtype": "float16",
        "file": fpath.name,
        "bytes": fpath.stat().st_size,
        "sha256_table": sha,
        "abs_max": amax,
        "fit_16ids": {"caption": fit},
        "truncated_over_24ids": over24,
        "empty_captions": n_empty,
        "blank": blank_i,
        "row_main": "row_main.npy",
        "img_idx": str(img_idx_path),
        "img_idx_shape": list(img_idx.shape),
        "data": str(data),
        "clip": str(CLIP_PT),
        "tokenizer": str(TOK),
        "device": str(dev),
        "batch": a.batch,
        "seconds": round(time.time() - t0, 1),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if meta["bytes"] != need:
        raise SystemExit(f"bytes {meta['bytes']} != need {need}")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)

    mm = np.memmap(fpath, dtype=np.float16, mode="r", shape=(C, L, 768))
    if mm.shape[0] != table.shape[0] or row_main.shape[0] != img_idx.shape[0]:
        raise SystemExit("alignment failed")
    if int(row_main.min()) < 0 or int(row_main.max()) >= C:
        raise SystemExit("row_main out of range")
    rng = np.random.default_rng(0)
    chk = rng.choice(C, size=min(256, C), replace=False)
    with torch.no_grad(), (torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else torch.autocast("cpu", enabled=False)):
        on = clip(torch.from_numpy(table[chk].astype(np.int64)).to(dev)).float().cpu()
    cf = torch.from_numpy(np.ascontiguousarray(mm[chk])).float()
    bos_t = torch.from_numpy(bos).float()
    diff = float((on[:, 1:] - cf).abs().max())
    bdiff = float((on[:, 0] - bos_t).abs().max())
    print(
        f"verify: feats {mm.shape} align img_idx {tuple(img_idx.shape)} "
        f"row_main {row_main.shape} unique_used {len(np.unique(row_main))} "
        f"max|online-cached| {diff:.6g} bos {bdiff:.6g}",
        flush=True,
    )
    print(f"note: trainer-style fp32 vs fp16 max abs {diff} (bos {bdiff}); 1 ulp at absmax~856 is 0.125-0.5", flush=True)
    if diff > 1.0 or bdiff > 1.0:
        raise SystemExit(f"cache mismatch too large: {diff} bos {bdiff}")
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
