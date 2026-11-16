#!/usr/bin/env python3
"""Second-pass magenta-fringe drop on already-filtered azyg images (8-connected, slightly looser key)."""
import os  # env placeholders ($DATA_ROOT etc.)
import json, shutil
from collections import Counter
from pathlib import Path
import numpy as np

AZYG = Path(os.path.expandvars("$PIXIE_ROOT")) / "data32/clean/azyg"

def fringe(arr, gmax=50, rmin=90, bmin=90, rbd=60, gmin_delta=70, max_out=12, max_in=1):
    r, g, b, a = arr[...,0].astype(np.int16), arr[...,1].astype(np.int16), arr[...,2].astype(np.int16), arr[...,3]
    opaque = a >= 128
    key = opaque & (g <= gmax) & (r >= rmin) & (b >= bmin) & (np.abs(r - b) <= rbd) & ((np.maximum(r, b) - g) >= gmin_delta)
    trans = a <= 16
    pad = np.pad(trans, 1, constant_values=True)
    n8 = (pad[:-2,1:-1]|pad[2:,1:-1]|pad[1:-1,:-2]|pad[1:-1,2:]|
          pad[:-2,:-2]|pad[:-2,2:]|pad[2:,:-2]|pad[2:,2:])
    outline = opaque & n8
    n_out = int((key & outline).sum()); n_in = int((key & ~outline).sum())
    return (n_out > 0) and (n_in <= max_in) and (n_out <= max_out), n_out, n_in

def main():
    recs = [json.loads(l) for l in (AZYG / "meta.jsonl").open()]
    imgs = np.load(AZYG / "images.npy", mmap_mode="r")
    assert len(recs) == len(imgs)
    keep, dropped = [], []
    for i, r in enumerate(recs):
        # only re-check lora; sheet already had zero hits under the first detector
        if r.get("backend") == "lora":
            has, no, ni = fringe(imgs[i])
            if has:
                dropped.append({"idx": i, "caption": r.get("caption"), "n_out": no, "n_in": ni, "split": r.get("split")})
                continue
        keep.append(i)
    print(f"pass2: in={len(recs)} drop={len(dropped)} keep={len(keep)}", flush=True)
    (AZYG / "pass2_dropped.json").write_text(json.dumps({"n": len(dropped), "by_n_out": dict(Counter(d["n_out"] for d in dropped)), "examples": dropped[:20]}, indent=2))
    if not dropped:
        print("nothing to do")
        return
    out = np.asarray(imgs[keep])
    tmp = AZYG / "images.npy.tmp"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=out.shape)
    mm[:] = out; mm.flush(); del mm
    tmp.replace(AZYG / "images.npy")
    new_recs = []
    for j, i in enumerate(keep):
        r = recs[i]
        r["idx"] = j
        new_recs.append(r)
    with (AZYG / "meta.jsonl").open("w", encoding="utf-8") as f:
        for r in new_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    st = json.loads((AZYG / "filter_stats.json").read_text())
    st["pass2_dropped"] = len(dropped)
    st["kept"] = len(keep)
    st["kept_by_backend"]["lora"] = st["kept_by_backend"].get("lora", 0) - len(dropped)
    st["dropped_magenta_by_backend"]["lora"] = st["dropped_magenta_by_backend"].get("lora", 0) + len(dropped)
    st["counters"]["drop_magenta_fringe_pass2"] = len(dropped)
    st["counters"]["kept"] = len(keep)
    (AZYG / "filter_stats.json").write_text(json.dumps(st, indent=2))
    print("pass2 done kept", len(keep), "stats updated")

if __name__ == "__main__":
    main()
