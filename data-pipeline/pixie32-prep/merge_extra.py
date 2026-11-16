#!/usr/bin/env python3
"""Merge kenney + oga extras into clean/extra/images.npy + meta.jsonl with hashes."""
import os  # env placeholders ($DATA_ROOT etc.)
import json
from pathlib import Path
import numpy as np
from PIL import Image

OUT = Path(os.path.expandvars("$PIXIE_ROOT")) / "data32/clean/extra"

def dhash64(arr, size=8):
    im = Image.fromarray(arr).convert("L").resize((size + 1, size), Image.Resampling.NEAREST)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    h = 0
    for i, v in enumerate(bits):
        if v:
            h |= 1 << i
    return h

def ahash64(arr, size=8):
    im = Image.fromarray(arr).convert("L").resize((size, size), Image.Resampling.NEAREST)
    a = np.asarray(im, dtype=np.int16)
    bits = (a.flatten() > a.mean())
    h = 0
    for i, v in enumerate(bits):
        if v:
            h |= 1 << i
    return h

def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

def main():
    parts = []
    for name in ("kenney", "oga"):
        ip, mp = OUT / f"{name}_images.npy", OUT / f"{name}_meta.jsonl"
        if ip.exists() and mp.exists():
            imgs = np.load(ip)
            recs = load_jsonl(mp)
            assert len(imgs) == len(recs), (name, len(imgs), len(recs))
            parts.append((name, imgs, recs))
            print(f"merge: {name} n={len(recs)}")
    if not parts:
        raise SystemExit("no extra parts")
    recs_all, imgs_all = [], []
    for name, imgs, recs in parts:
        for i, r in enumerate(recs):
            arr = imgs[i]
            r = dict(r)
            r["dhash"] = dhash64(arr)
            r["ahash"] = ahash64(arr)
            recs_all.append(r)
            imgs_all.append(arr)
    arr = np.stack(imgs_all, 0)
    tmp = OUT / "images.npy.tmp"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=arr.shape)
    mm[:] = arr; mm.flush(); del mm
    tmp.replace(OUT / "images.npy")
    with (OUT / "meta.jsonl").open("w", encoding="utf-8") as f:
        for j, r in enumerate(recs_all):
            r["idx"] = j
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("merge extra done", len(recs_all), "from", [n for n,_,_ in parts])

if __name__ == "__main__":
    main()
