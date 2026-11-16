"""Precompute frozen CLIP-L penultimate features for every distinct training caption (s7 caption table).

Output (new folder, nothing existing is touched):
  feats_t{TL}_f16.bin  raw fp16 [C, TL-1, 768]  (position 0 = BOS is identical for every caption -> bos_feat.npy)
  bos_feat.npy, captions_t{TL}.npy, meta.json
Features are computed exactly like the online trainer (bf16 weights, bf16 autocast) and stored as fp16
(bf16 -> fp16 is exact inside fp16 range; range is checked).
"""
import argparse, json, os, queue, shutil, sys, threading, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3_model as M  # noqa: E402
import s7_data as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--text-len", type=int, default=18)
    ap.add_argument("--batch", type=int, default=2048); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = a.out; out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    tr = np.load(a.data / "v3_train.npz")
    tab = D.build_caption_table(tr, a.text_len)
    T = tab["table"]; C = len(T) if not a.limit else min(a.limit, len(T)); L = a.text_len - 1
    need = C * L * 768 * 2
    free = shutil.disk_usage(out).free
    print(f"captions {len(T)} (writing {C}), {need / 1e9:.1f} GB, free {free / 1e9:.1f} GB, table {time.time() - t0:.0f}s", flush=True)
    assert need < free - 40e9, "not enough free space (keeps 40 GB headroom)"
    np.save(out / f"captions_t{a.text_len}.npy", T)
    clip = M.ClipTextPenultimate().to(dev)
    clip.load_hf(torch.load(a.clip, map_location="cpu", weights_only=False))
    clip.to(torch.bfloat16).eval()
    q = queue.Queue(maxsize=6); fpath = out / f"feats_t{a.text_len}_f16.bin"

    def writer():
        with open(fpath, "wb", buffering=0) as f:
            while True:
                x = q.get()
                if x is None:
                    break
                f.write(memoryview(x).cast("B"))
            os.fsync(f.fileno())
    th = threading.Thread(target=writer); th.start()
    amax = 0.0; bos = None; t1 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, C, a.batch):
            ids = torch.from_numpy(T[s:s + a.batch].astype(np.int64)).to(dev)
            h = clip(ids)                                        # bf16 [b,TL,768]
            if bos is None:
                bos = h[0, 0].float().cpu().numpy()
            assert torch.equal(h[:, 0], h[:1, 0].expand_as(h[:, 0])), "BOS position not constant"
            hm = float(h.abs().max()); amax = max(amax, hm)
            hf = h[:, 1:].to(torch.float16)
            assert torch.isfinite(hf).all()
            q.put(hf.cpu().numpy())
            if (s // a.batch) % 100 == 0:
                el = time.time() - t1
                print(f"{s + len(ids)}/{C} {el:.0f}s ({(s + len(ids)) / max(el, 1e-9):.0f} cap/s) absmax {amax:.1f}", flush=True)
    q.put(None); th.join()
    np.save(out / "bos_feat.npy", bos)
    meta = {"captions": int(C), "table_rows": int(len(T)), "text_len": a.text_len, "stored_positions": L, "dim": 768,
            "dtype": "float16", "file": fpath.name, "bytes": fpath.stat().st_size, "sha256_table": tab["sha256"],
            "abs_max": amax, "fit_16ids": tab["fit"], "data": str(a.data), "clip": str(a.clip),
            "seconds": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    assert meta["bytes"] == need
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
