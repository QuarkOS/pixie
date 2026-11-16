"""Gather v3 image arrays from the shared token cache (read-only). CPU only.

tokens.npy rows: [BOS] 96 caption bytes(+3) [SEP] 256 palette ids(+259). Writes into OUT:
train_img.npy [Nt,256] uint16, probe_img.npy, iid_img.npy, probe_tok.npy / iid_tok.npy (full v1 rows)."""
import argparse, glob, hashlib, json, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", required=True, help="tokens.npy, or a glob of shard *.tokens.npy (concatenated in order)")
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--limit-train", type=int, default=0)
a = ap.parse_args()
t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
if "*" in a.tokens:
    parts = [np.load(p, mmap_mode="r") for p in sorted(glob.glob(a.tokens))]
    offs = np.cumsum([0] + [p.shape[0] for p in parts])
    def take(rows):
        rows = np.asarray(rows, np.int64); res = np.empty((len(rows), 354), np.uint16)
        sh = np.searchsorted(offs, rows, side="right") - 1
        for s in np.unique(sh):
            m = sh == s; res[m] = parts[s][rows[m] - offs[s]]
        return res
else:
    tok = np.load(a.tokens, mmap_mode="r"); assert tok.shape[1] == 354
    def take(rows):
        return np.asarray(tok[np.asarray(rows, np.int64)])
tr = np.load(Path(a.data) / "v3_train.npz"); ev = np.load(Path(a.data) / "v3_eval.npz")
rows = tr["rows"][:a.limit_train] if a.limit_train else tr["rows"]
img = np.lib.format.open_memmap(out / "train_img.tmp.npy", mode="w+", dtype=np.uint16, shape=(len(rows), 256))
CH = 100_000
order = np.argsort(rows)  # sequential reads on the HDD
for s in range(0, len(rows), CH):
    idx = np.sort(order[s:s + CH])
    blk = take(rows[idx])[:, 98:].astype(np.int32) - 259
    assert blk.min() >= 0 and blk.max() < 4097
    img[idx] = blk.astype(np.uint16)
    print("train", s + len(idx), f"{time.time() - t0:.0f}s", flush=True)
img.flush(); del img
(out / "train_img.tmp.npy").replace(out / "train_img.npy")
man = {"train_rows": int(len(rows)), "limit_train": a.limit_train}
for name in ("probe", "iid"):
    t = take(ev[f"{name}_rows"]); np.save(out / f"{name}_tok.npy", t)
    np.save(out / f"{name}_img.npy", (t[:, 98:].astype(np.int32) - 259).astype(np.uint16))
for f in ("train_img.npy", "probe_img.npy", "iid_img.npy", "probe_tok.npy", "iid_tok.npy"):
    h = hashlib.sha256()
    with open(out / f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    man["sha256_" + f] = h.hexdigest()
man["data_manifest_sha256"] = hashlib.sha256((Path(a.data) / "manifest_v3.json").read_bytes()).hexdigest()
man["seconds"] = round(time.time() - t0, 1)
(out / "cache_manifest.json").write_text(json.dumps(man, indent=1))
print(json.dumps(man), flush=True)
