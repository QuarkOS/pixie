#!/usr/bin/env python3
"""Tighter perceptual dedup for sparse 32x32 sprites.

The first pass used 8x8 dhash/ahash on the full canvas with Hamming <= 6.
Transparent padding makes those hashes collide, so distinct icons got merged.
This pass:
  - always drops exact MD5 duplicates (prefer real over synthetic)
  - hashes the opaque bounding box, 16x16 dhash + 16x16 ahash
  - merges only if dhash<=4 AND ahash<=4 AND mean opaque RGB L1 <= 24
    (keeps colour variants of the same silhouette)
  - then applies Blind-50 / eval-50 holdout filters
  - writes a new out/ set; does not delete clean/ or raw/
"""
from __future__ import annotations

import csv, json, os, re, shutil, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(os.path.expandvars("$PIXIE_ROOT/data32"))
CLEAN = ROOT / "clean"
OUT = ROOT / "out"
HOLD = Path(__file__).resolve().parent / "holdout"
BACKUP = ROOT / "out_thresh6"  # preserve the first (too-aggressive) pass


def log(*a):
    print(*a, flush=True)


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_holdouts():
    def read(name):
        p = HOLD / name
        if not p.exists():
            p = HOLD.parent / name
        return [l.strip().lower() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    blind, ev = read("blind50.txt"), read("eval50.txt")
    assert len(blind) == 50 and len(ev) == 50
    return blind, ev


def holdout_re(phrases):
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b", re.I)


def is_holdout(rec, blind_re, eval_re, eval_set):
    for s in (rec.get("caption") or "", rec.get("text") or "", rec.get("caption_alt") or "", rec.get("caption_original") or ""):
        c = " ".join(s.lower().split())
        for pref in ("pixel art icon of a ", "pixel art icon of ", "pixel art of a ", "pixel art of ", "monochrome "):
            if c.startswith(pref):
                c = c[len(pref):]
                break
        if c in eval_set or s.lower().strip() in eval_set:
            return "eval50_exact"
        if blind_re.search(s) or blind_re.search(c):
            return "blind50_phrase"
        if eval_re.search(s) or eval_re.search(c):
            return "eval50_phrase"
    return None


def opaque_crop(arr: np.ndarray) -> np.ndarray:
    a = arr[..., 3] >= 128
    if not a.any():
        return arr
    ys, xs = np.where(a)
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def hash16(arr: np.ndarray):
    crop = opaque_crop(arr)
    # 16x16 on the opaque box: each bit is a ~2px region for a full 32x32 sprite
    g = Image.fromarray(crop).convert("L")
    d_im = np.asarray(g.resize((17, 16), Image.Resampling.NEAREST), dtype=np.int16)
    a_im = np.asarray(g.resize((16, 16), Image.Resampling.NEAREST), dtype=np.int16)
    dbits = (d_im[:, 1:] > d_im[:, :-1]).flatten()
    abits = (a_im.flatten() > a_im.mean()).flatten()
    dh = ah = 0
    for i, v in enumerate(dbits):
        if v:
            dh |= 1 << i
    for i, v in enumerate(abits):
        if v:
            ah |= 1 << i
    op = arr[..., 3] >= 128
    if op.any():
        mean = arr[op][:, :3].mean(axis=0)
    else:
        mean = np.zeros(3, dtype=np.float64)
    return int(dh), int(ah), mean.astype(np.float32)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def keep_key(rec):
    synth = 1 if rec.get("synthetic") else 0
    src = rec.get("source") or ""
    # prefer real packs, then pixl, then kenney/oga, then azyg
    if src.startswith("ocdbytes"):
        sr = 0
    elif src in ("OpenGameArt", "kenney.nl") or src.startswith("kenney"):
        sr = 1
    else:
        sr = 2
    cap = 0 if (rec.get("caption") or "").strip() else 1
    return (synth, sr, cap, -len(rec.get("caption") or ""))


def save_memmap_u8(path: Path, arr: np.ndarray):
    tmp = path.with_suffix(path.suffix + ".tmp")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=arr.shape)
    mm[:] = arr
    mm.flush()
    del mm
    tmp.replace(path)


def maybe_backup_old_out():
    """Move the thresh-6 result aside once, don't overwrite a backup."""
    if not (OUT / "images.npy").exists():
        return
    if (BACKUP / "images.npy").exists():
        log("backup already exists at", BACKUP, "— leaving it")
        return
    BACKUP.mkdir(parents=True, exist_ok=True)
    for name in ("images.npy", "manifest.csv", "meta.jsonl", "meta_pre_recaption.jsonl",
                 "stats.json", "dedup_stats.json", "ATTRIBUTION.md",
                 "data32-sample.png", "data32-sample.json", "recaption_stats.json"):
        src = OUT / name
        if src.exists():
            shutil.move(str(src), str(BACKUP / name))
    log("moved thresh-6 out/ ->", BACKUP)


def main():
    t0 = time.time()
    maybe_backup_old_out()
    parts = []
    mms = {}
    for name in ("azyg", "pixl", "extra"):
        img_p = CLEAN / name / "images.npy"
        meta_p = CLEAN / name / "meta.jsonl"
        if not img_p.exists():
            continue
        imgs = np.load(img_p, mmap_mode="r")
        recs = load_jsonl(meta_p)
        assert len(recs) == len(imgs), (name, len(recs), len(imgs))
        mms[name] = imgs
        for i, r in enumerate(recs):
            r = dict(r)
            r["_part"] = name
            r["_local_idx"] = i
            parts.append(r)
        log(f"loaded {name} n={len(recs)}")
    n = len(parts)
    log("hashing", n)
    dhs = np.empty(n, dtype=object)
    ahs = np.empty(n, dtype=object)
    means = np.zeros((n, 3), dtype=np.float32)
    for i, r in enumerate(parts):
        arr = mms[r["_part"]][r["_local_idx"]]
        dh, ah, mu = hash16(arr)
        dhs[i] = dh
        ahs[i] = ah
        means[i] = mu
        r["dhash16"] = dh
        r["ahash16"] = ah
        if (i + 1) % 25000 == 0:
            log(f"  hashed {i+1}/{n}")
    log("hashed in", round(time.time() - t0, 1), "s")

    # exact md5
    by_md5 = defaultdict(list)
    for i, r in enumerate(parts):
        by_md5[r["md5"]].append(i)
    drop = set()
    exact_groups = 0
    for ids in by_md5.values():
        if len(ids) < 2:
            continue
        exact_groups += 1
        ids_sorted = sorted(ids, key=lambda i: keep_key(parts[i]))
        for j in ids_sorted[1:]:
            drop.add(j)
    log("exact md5 groups", exact_groups, "extra rows dropped", len(drop))

    # perceptual among survivors — fast bands, not 16 growing prefix lists
    # (those became O(n^2) in practice). Two steps:
    #   1) exact same 16x16 dhash+ahash, merge if colour is also close
    #   2) Hamming <= 4 on both hashes: compare only within 32-bit LSH bands
    survivors = [i for i in range(n) if i not in drop]
    TH_H, TH_A, TH_COL = 4, 4, 24.0
    n_phash = 0
    parent = {i: i for i in survivors}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri == rj:
            return False
        # keep the better rank as root
        if keep_key(parts[ri]) <= keep_key(parts[rj]):
            parent[rj] = ri
        else:
            parent[ri] = rj
        return True

    def close(i, j):
        if hamming(dhs[i], dhs[j]) <= TH_H and hamming(ahs[i], ahs[j]) <= TH_A:
            if float(np.abs(means[i] - means[j]).sum()) <= TH_COL:
                return True
        return False

    # exact perceptual hash groups
    exact_h = defaultdict(list)
    for i in survivors:
        exact_h[(dhs[i], ahs[i])].append(i)
    for ids in exact_h.values():
        if len(ids) < 2:
            continue
        # split by colour so a red/blue pair with the same silhouette stay distinct
        used = [False] * len(ids)
        for a in range(len(ids)):
            if used[a]:
                continue
            for b in range(a + 1, len(ids)):
                if used[b]:
                    continue
                if float(np.abs(means[ids[a]] - means[ids[b]]).sum()) <= TH_COL:
                    if union(ids[a], ids[b]):
                        n_phash += 1
                    used[b] = True

    # near-duplicates: 8 x 32-bit bands of the 256-bit dhash
    bands = defaultdict(list)
    for i in survivors:
        h = dhs[i]
        for b in range(8):
            bands[((h >> (b * 32)) & 0xFFFFFFFF, b)].append(i)
    # only inspect bands that are small enough to be selective
    seen_pairs = set()
    for ids in bands.values():
        if 2 <= len(ids) <= 128:
            for a in range(len(ids)):
                ia = ids[a]
                for b in range(a + 1, len(ids)):
                    ib = ids[b]
                    if find(ia) == find(ib):
                        continue
                    key = (ia, ib) if ia < ib else (ib, ia)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    if close(ia, ib):
                        if union(ia, ib):
                            n_phash += 1

    for i in survivors:
        r = find(i)
        if r != i:
            drop.add(i)

    keep_idx = [i for i in range(n) if i not in drop]
    log(f"after phash: drop_total={len(drop)} phash_merges={n_phash} keep={len(keep_idx)}")

    blind, ev = load_holdouts()
    blind_re, eval_re, eval_set = holdout_re(blind), holdout_re(ev), set(ev)
    hold_drop = []
    final = []
    for i in keep_idx:
        reason = is_holdout(parts[i], blind_re, eval_re, eval_set)
        if reason:
            hold_drop.append({"reason": reason, "caption": parts[i].get("caption"), "text": parts[i].get("text"), "source": parts[i].get("source")})
        else:
            final.append(i)
    log(f"holdout dropped {len(hold_drop)} keep={len(final)}")

    OUT.mkdir(parents=True, exist_ok=True)
    arr = np.empty((len(final), 32, 32, 4), dtype=np.uint8)
    out_recs = []
    for j, i in enumerate(final):
        r = parts[i]
        arr[j] = mms[r["_part"]][r["_local_idx"]]
        rr = {k: v for k, v in r.items() if not k.startswith("_")}
        rr["idx"] = j
        out_recs.append(rr)
    save_memmap_u8(OUT / "images.npy", arr)
    with (OUT / "meta_pre_recaption.jsonl").open("w", encoding="utf-8") as f:
        for r in out_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # recaption was already "kept existing labels" — copy through
    with (OUT / "meta.jsonl").open("w", encoding="utf-8") as f:
        for r in out_recs:
            if not r.get("synthetic") and "caption_original" not in r:
                r["caption_original"] = r.get("caption") or ""
                r["caption_source"] = r.get("label_source") or r.get("caption_source") or "existing"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    (OUT / "recaption_stats.json").write_text(json.dumps({
        "real": sum(1 for r in out_recs if not r.get("synthetic")),
        "method": "kept_existing_no_key",
        "updated": 0,
        "failed": 0,
        "note": "GEMINI_API_KEY was not on the 3080 Ti; pixl claude-visual/filename/atlas labels kept; Kenney/OGA use filename/index captions.",
    }, indent=2))

    dedup_stats = {
        "in": n,
        "dropped_dup": len(drop),
        "dropped_holdout": len(hold_drop),
        "kept": len(final),
        "dedup": {
            "exact_dup_groups": exact_groups,
            "phash_dups": n_phash,
            "dhash_bits": 256,
            "ahash_bits": 256,
            "threshold_dhash": TH_H,
            "threshold_ahash": TH_A,
            "mean_rgb_l1": TH_COL,
            "hash": "opaque-bbox 16x16 NEAREST",
        },
        "holdout_reasons": dict(Counter(x["reason"] for x in hold_drop)),
        "holdout_examples": hold_drop[:30],
        "kept_by_source": dict(Counter(parts[i].get("source") for i in final)),
        "kept_synthetic": int(sum(1 for i in final if parts[i].get("synthetic"))),
        "kept_real": int(sum(1 for i in final if not parts[i].get("synthetic"))),
        "in_by_source": dict(Counter(r.get("source") for r in parts)),
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "dedup_stats.json").write_text(json.dumps(dedup_stats, indent=2, ensure_ascii=False))
    log("dedup_v2 done", json.dumps({k: dedup_stats[k] for k in ("in", "dropped_dup", "dropped_holdout", "kept", "kept_by_source")}))


if __name__ == "__main__":
    main()
