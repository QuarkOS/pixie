#!/usr/bin/env python3
"""Pixie-32 out_v2 data fixes (review). CPU only.

Writes a new tree at $PIXIE_ROOT/data32/out_v2 and work files at
$PIXIE_ROOT/data32/work_v2. Never deletes or overwrites out/, clean/, runs/,
or v3 caches. Chat must stay off the GPU path used here (this process is CPU).
"""
from __future__ import annotations

import argparse, csv, hashlib, io, json, os, queue, re, shutil, sys, threading, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ.get("DATA32_ROOT", os.path.expandvars("$PIXIE_ROOT/data32")))
CLEAN = ROOT / "clean"
RAW = ROOT / "raw"
WORK = ROOT / "work_v2"
OUT = ROOT / "out_v2"
PAL_PATH = Path(os.path.expandvars("$PIXIE_ROOT/data/subset150k/palette.npy"))
HOLD = Path(__file__).resolve().parent / "holdout"
V3_CODE = Path(os.path.expandvars("$PIXIE_ROOT/v3/code"))
CLIP_PT = Path(os.path.expandvars("$PIXIE_ROOT/v3/clip/clip_l14_text_l11_f32.pt"))
TOK_PATH = Path(os.path.expandvars("$PIXIE_ROOT/v3/tok/tokenizer.json"))

BOS, EOS, L26, TEXT_IDS, TEXT_LEN = 49406, 49407, 26, 24, 18


def log(*a):
    print(*a, flush=True)


def ensure():
    WORK.mkdir(parents=True, exist_ok=True)
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- colour / alpha
def srgb_u8_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """rgb [..., 3] uint8 -> Oklab [..., 3] float64."""
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.2158617610 * m_ - 0.2417665976 * s_  # noqa: N806
    return np.stack([L, A, B], axis=-1)


def snap_canonical(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hard-threshold alpha at 128; set fully transparent pixels to (0,0,0,0).

    Returns (snapped uint8 [...,4], semi_ratio float [...] of original alpha in 1..254).
    """
    a = arr[..., 3]
    semi = ((a >= 1) & (a <= 254)).mean(axis=(-1, -2)) if arr.ndim == 4 else float(((a >= 1) & (a <= 254)).mean())
    out = arr.copy()
    opaque = a >= 128
    out[..., 3] = np.where(opaque, 255, 0)
    trans = out[..., 3] == 0
    out[trans, 0] = 0
    out[trans, 1] = 0
    out[trans, 2] = 0
    return out, semi


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2].astype(np.uint32)


def palette_oklab_tables(pal: np.ndarray):
    lab = srgb_u8_to_oklab(pal)  # [4097, 3]
    return lab


def nn_oklab(colors_u8: np.ndarray, pal_lab: np.ndarray, skip0: bool = False, chunk: int = 4096) -> np.ndarray:
    """colors_u8 [M,3] -> palette indices [M] uint16, nearest in Oklab."""
    if len(colors_u8) == 0:
        return np.zeros((0,), dtype=np.uint16)
    lab = srgb_u8_to_oklab(colors_u8)
    out = np.empty(len(colors_u8), dtype=np.uint16)
    for s in range(0, len(colors_u8), chunk):
        e = min(len(colors_u8), s + chunk)
        d = lab[s:e, None, :] - pal_lab[None, :, :]
        ss = np.einsum("mcd,mcd->mc", d, d)
        if skip0:
            ss[:, 0] = np.inf
        out[s:e] = np.argmin(ss, axis=1).astype(np.uint16)
    return out


def unique_opaque_and_mse(imgs: np.ndarray, pal_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """imgs [N,32,32,4] already snapped. Returns K_img, mse (Oklab), n_opaque."""
    n = len(imgs)
    K = np.zeros(n, np.int32)
    mse = np.zeros(n, np.float64)
    n_op = np.zeros(n, np.int32)
    # collect unique colours in this block, map once
    op = imgs[..., 3] >= 128
    packed = pack_rgb(imgs[..., :3])
    uniq = np.unique(packed[op]) if op.any() else np.zeros((0,), np.uint32)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
    ids = nn_oklab(cols, pal_lab, skip0=False)
    pal_of = pal_lab[ids]
    src_lab = srgb_u8_to_oklab(cols)
    err = np.sum((src_lab - pal_of) ** 2, axis=1)  # per unique colour
    order = np.argsort(uniq)
    keys, ev, plab = uniq[order], err[order], pal_of[order]
    for i in range(n):
        m = op[i]
        n_op[i] = int(m.sum())
        if n_op[i] == 0:
            continue
        u = packed[i][m]
        K[i] = int(np.unique(u).size)
        ix = np.searchsorted(keys, u)
        ix = np.clip(ix, 0, len(keys) - 1)
        # keys should always hit
        mse[i] = float(ev[ix].mean()) if len(keys) else 0.0
    return K, mse, n_op


# ---------------------------------------------------------------- pHash (DCT, bbox, hflip)
def _dct1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[-1]
    v = np.empty_like(x)
    v[..., : N // 2] = x[..., ::2]
    v[..., N // 2 :] = x[..., ::-2]
    X = np.fft.fft(v, axis=-1).real
    k = np.arange(N, dtype=np.float64)
    X *= 2.0 * np.cos(np.pi * k / (2 * N))
    return X


def dct2(a: np.ndarray) -> np.ndarray:
    return _dct1d(_dct1d(a.T).T)


def bbox_gray(arr: np.ndarray) -> np.ndarray | None:
    m = arr[..., 3] >= 128
    if not m.any():
        return None
    ys, xs = np.where(m)
    crop = arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    rgb = crop[..., :3].astype(np.float64)
    g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    g = np.where(crop[..., 3] >= 128, g, np.nan)
    mu = np.nanmean(g)
    sd = np.nanstd(g)
    g = np.where(np.isnan(g), mu if np.isfinite(mu) else 0.0, g)
    if np.isfinite(sd) and sd > 1e-6:
        g = (g - mu) / sd  # contrast-normalise so recolours with luma shift still match
    else:
        g = g - (mu if np.isfinite(mu) else 0.0)
    return g.astype(np.float64)


def phash256(gray: np.ndarray, freq: int = 16, size: int = 32) -> np.ndarray:
    """Return 256-bit hash as uint64[4]."""
    im = Image.fromarray(np.clip(gray * 32 + 128, 0, 255).astype(np.uint8), mode="L")
    a = np.asarray(im.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float64)
    c = dct2(a)[:freq, :freq]
    # skip DC
    flat = c.flatten()[1:]
    med = np.median(flat)
    bits = (c.flatten() > med)
    bits[0] = False
    packed = np.packbits(bits.astype(np.uint8), bitorder="big")  # 32 bytes
    return np.frombuffer(packed.tobytes().ljust(32, b"\0")[:32], dtype=np.uint64).copy()


def image_phashes(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = bbox_gray(arr)
    if g is None:
        z = np.zeros(4, dtype=np.uint64)
        return z, z
    h = phash256(g)
    hf = phash256(np.fliplr(g))
    return h, hf


def ham256(a: np.ndarray, b: np.ndarray) -> int:
    x = a.astype(np.uint64) ^ b.astype(np.uint64)
    return int(sum(int(v).bit_count() for v in x))


def ham_hflip(h1, f1, h2, f2) -> int:
    return min(ham256(h1, h2), ham256(h1, f2), ham256(f1, h2), ham256(f1, f2))


# ---------------------------------------------------------------- I/O helpers
def load_jsonl(path: Path) -> list[dict]:
    recs = []
    with path.open() as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def write_jsonl(path: Path, recs):
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_u8(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=arr.shape)
    mm[:] = arr
    mm.flush()
    del mm
    tmp.replace(path)


def pil_from_cell(cell):
    if cell is None:
        return None
    if hasattr(cell, "as_py"):
        cell = cell.as_py()
    if isinstance(cell, Image.Image):
        return cell.convert("RGBA")
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b:
            return Image.open(io.BytesIO(b)).convert("RGBA")
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(cell))).convert("RGBA")
    return None


def is_gameicons(pack: str, url: str) -> bool:
    p = (pack or "").lower()
    u = (url or "").lower()
    # Drop game-icons.net only (pack gameicons_game-icons). Do not drop Kenney Game Icons.
    return ("game-icons.net" in u) or p.startswith("game-icons") or p.startswith("gameicons_") or p.startswith("gameicons/")


def font(size=11):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def contact_sheet(imgs, labels, path: Path, scale=6, cols=8, title=""):
    if not imgs:
        return
    tw, th = 32 * scale, 32 * scale + 28
    rows = (len(imgs) + cols - 1) // cols
    pad_top = 28 if title else 4
    canvas = Image.new("RGB", (cols * (tw + 2) + 2, pad_top + rows * (th + 2) + 2), (40, 40, 40))
    dr = ImageDraw.Draw(canvas)
    fn = font(14 if title else 11)
    fns = font(10)
    if title:
        dr.text((6, 4), title, fill=(240, 240, 240), font=fn)
    for i, (im, lab) in enumerate(zip(imgs, labels)):
        r, c = divmod(i, cols)
        x, y = 2 + c * (tw + 2), pad_top + r * (th + 2)
        cell = Image.fromarray(im, "RGBA")
        bg = Image.new("RGBA", (32, 32), (48, 48, 48, 255))
        bg.alpha_composite(cell)
        tile = bg.convert("RGB").resize((32 * scale, 32 * scale), Image.Resampling.NEAREST)
        canvas.paste(tile, (x, y))
        dr.text((x + 2, y + 32 * scale + 2), str(lab)[:40], fill=(220, 220, 220), font=fns)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    log("wrote", path, canvas.size)


# ---------------------------------------------------------------- stages
def stage_profile_azyg(pal_lab):
    ensure()
    dest = WORK / "azyg_profile.npz"
    if dest.exists():
        log("profile_azyg: already done")
        return dict(np.load(dest, allow_pickle=False))
    imgs = np.load(CLEAN / "azyg" / "images.npy", mmap_mode="r")
    n = len(imgs)
    log(f"profile_azyg: {n} images")
    K = np.zeros(n, np.int32)
    mse = np.zeros(n, np.float64)
    semi = np.zeros(n, np.float64)
    n_op = np.zeros(n, np.int32)
    t0 = time.time()
    CH = 2048
    # we need snapped copies for K/MSE; do not rewrite clean/
    for s in range(0, n, CH):
        e = min(n, s + CH)
        blk, semi[s:e] = snap_canonical(np.asarray(imgs[s:e]))
        k, m, no = unique_opaque_and_mse(blk, pal_lab)
        K[s:e], mse[s:e], n_op[s:e] = k, m, no
        if e % 16384 == 0 or e == n:
            log(f"  profile {e}/{n} {time.time()-t0:.0f}s  K50={np.median(K[:e]):.0f} mse50={np.median(mse[:e]):.5f}")
    np.savez(dest, K=K, mse=mse, semi=semi, n_op=n_op)
    log(f"profile_azyg done {time.time()-t0:.1f}s")
    return dict(K=K, mse=mse, semi=semi, n_op=n_op)


def choose_thresholds(prof) -> dict:
    K, mse, semi = prof["K"], prof["mse"], prof["semi"]
    crisp = K <= 32
    # MSE of already-crisp sprites: drop high-K images whose error exceeds almost all of these
    t_mse = float(np.quantile(mse[crisp], 0.99)) if crisp.any() else 0.01
    t_k_hard = 96  # 32x32 with this many unique colours is almost never hand-pixel art
    # drop high-K / high-MSE smooth renders; keep low-MSE high-K (already palettized) unless K is extreme
    drop = ((K > 32) & (mse > t_mse)) | (K >= t_k_hard)
    # never drop empty? keep them tagged uncrisp
    stats = {
        "t_mse_oklab": t_mse,
        "t_k_hard": t_k_hard,
        "rule": "drop if (K_img>32 and oklab_mse>P99_of_K<=32) or K_img>=96; tag crisp if K_img<=32",
        "n": int(len(K)),
        "n_crisp": int(crisp.sum()),
        "n_drop": int(drop.sum()),
        "n_keep": int((~drop).sum()),
        "K_pct": {str(p): float(np.percentile(K, p)) for p in (25, 50, 75, 90, 95, 99)},
        "mse_pct": {str(p): float(np.percentile(mse, p)) for p in (25, 50, 75, 90, 95, 99)},
        "mse_crisp_pct": {str(p): float(np.percentile(mse[crisp], p)) for p in (50, 90, 95, 99)} if crisp.any() else {},
        "semi_pct": {str(p): float(np.percentile(semi, p)) for p in (50, 90, 95, 99)},
        "n_semi_any": int((semi > 0).sum()),
    }
    (WORK / "azyg_thresholds.json").write_text(json.dumps(stats, indent=2))
    log("thresholds", json.dumps(stats))
    return {**stats, "drop_mask": drop, "crisp_mask": crisp}


def stage_boundary_sheet(thr):
    ensure()
    path = OUT / "sheets" / "boundary_kept_vs_dropped.png"
    imgs = np.load(CLEAN / "azyg" / "images.npy", mmap_mode="r")
    recs = load_jsonl(CLEAN / "azyg" / "meta.jsonl")
    K = np.load(WORK / "azyg_profile.npz")["K"]
    mse = np.load(WORK / "azyg_profile.npz")["mse"]
    drop = thr["drop_mask"]
    t_mse = thr["t_mse_oklab"]
    # boundary kept: not dropped, but close (K just over 32 with mse just under T, or K 28-32 high mse)
    kept_idx = np.where(~drop)[0]
    dropped_idx = np.where(drop)[0]
    # score closeness to boundary
    if len(kept_idx):
        # prefer K in 24-48 and mse near T
        score_k = np.abs(K[kept_idx].astype(np.float64) - 33)
        score_m = np.abs(mse[kept_idx] - t_mse)
        keep_order = kept_idx[np.argsort(score_k + 200 * score_m)]
    else:
        keep_order = kept_idx
    if len(dropped_idx):
        score_k = np.abs(K[dropped_idx].astype(np.float64) - 33)
        score_m = np.abs(mse[dropped_idx] - t_mse)
        drop_order = dropped_idx[np.argsort(score_k + 200 * score_m)]
    else:
        drop_order = dropped_idx
    nk, nd = 32, 32
    cells, labs = [], []
    for i in list(keep_order[:nk]):
        im, _ = snap_canonical(np.asarray(imgs[i]))
        cap = (recs[i].get("caption") or "")[:28]
        cells.append(im)
        labs.append(f"KEEP K={int(K[i])} mse={mse[i]:.4f}")
    for i in list(drop_order[:nd]):
        im, _ = snap_canonical(np.asarray(imgs[i]))
        cells.append(im)
        labs.append(f"DROP K={int(K[i])} mse={mse[i]:.4f}")
    contact_sheet(cells, labs, path, scale=5, cols=8,
                  title=f"azyg boundary  KEEP (top 4 rows) vs DROP (bottom 4)  T_mse={t_mse:.5f}  T_Khard=96")
    # also dump a few captions for the report
    ex = {
        "kept_boundary": [{"idx": int(i), "K": int(K[i]), "mse": float(mse[i]), "caption": recs[i].get("caption")} for i in keep_order[:12]],
        "dropped_boundary": [{"idx": int(i), "K": int(K[i]), "mse": float(mse[i]), "caption": recs[i].get("caption")} for i in drop_order[:12]],
    }
    (WORK / "boundary_examples.json").write_text(json.dumps(ex, indent=2))
    return path


def magenta_fringe(arr: np.ndarray) -> bool:
    r, g, b, a = arr[..., 0].astype(np.int16), arr[..., 1].astype(np.int16), arr[..., 2].astype(np.int16), arr[..., 3]
    opaque = a >= 128
    key = opaque & (g <= 40) & (r >= 100) & (b >= 100) & (np.abs(r - b) <= 50) & ((np.maximum(r, b) - g) >= 80)
    if not key.any():
        return False
    trans = a <= 16
    pad = np.pad(trans, 1, constant_values=True)
    outline = opaque & (pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:])
    n_out = int((key & outline).sum())
    n_in = int((key & ~outline).sum())
    return (n_out > 0) and (n_in <= 2) and (n_out <= 12)


def stage_refilter_pixl():
    ensure()
    dest_i, dest_m, dest_s = WORK / "pixl_images.npy", WORK / "pixl_meta.jsonl", WORK / "pixl_filter.json"
    if dest_i.exists() and dest_m.exists() and dest_s.exists():
        log("refilter_pixl: already done")
        return json.loads(dest_s.read_text())
    files = []
    for split, folder in (("train", RAW / "pixl-data" / "sprites" / "train-00000-of-00001.parquet"),
                          ("test", RAW / "pixl-data" / "sprites" / "test-00000-of-00001.parquet")):
        if folder.exists():
            files.append((split, folder))
    stats = Counter()
    kept_pack = Counter()
    recs, imgs = [], []
    for split, fp in files:
        rows = pq.read_table(fp).to_pylist()
        log(f"  pixl {split} {len(rows)}")
        for row in rows:
            stats["seen"] += 1
            pack = row.get("pack") or ""
            url = row.get("source_url") or ""
            if is_gameicons(pack, url):
                stats["drop_gameicons"] += 1
                continue
            try:
                im = pil_from_cell(row.get("image"))
            except Exception:
                stats["drop_decode"] += 1
                continue
            if im is None or im.size != (32, 32):
                stats["drop_not_32"] += 1
                continue
            arr = np.asarray(im.convert("RGBA"), dtype=np.uint8)
            arr, semi = snap_canonical(arr)
            if magenta_fringe(arr):
                stats["drop_magenta"] += 1
                continue
            recs.append({
                "idx": len(recs),
                "split": split,
                "source": "ocdbytes/pixl-data",
                "licence": (row.get("license") or "").strip() or "unknown",
                "synthetic": False,
                "pack": pack,
                "creator": row.get("creator") or "",
                "source_url": url,
                "path": row.get("path") or "",
                "caption": row.get("label") or "",
                "label_source": row.get("label_source") or "",
                "semi_ratio": float(semi),
            })
            imgs.append(arr)
            kept_pack[pack] += 1
            stats["kept"] += 1
            if semi > 0:
                stats["snapped_soft_alpha"] += 1
    if not imgs:
        raise SystemExit("pixl refilter kept 0")
    save_u8(dest_i, np.stack(imgs, 0))
    write_jsonl(dest_m, recs)
    out = {"counters": dict(stats), "kept_by_pack": dict(kept_pack)}
    dest_s.write_text(json.dumps(out, indent=2))
    log("refilter_pixl", out)
    return out


def load_extra():
    """Kenney + OGA already extracted 32x32; snap + canonicalize."""
    recs, imgs = [], []
    for name, ip, mp, src, lic in (
        ("kenney", CLEAN / "extra" / "kenney_images.npy", CLEAN / "extra" / "kenney_meta.jsonl", "kenney.nl", "CC0 1.0"),
        ("oga", CLEAN / "extra" / "oga_images.npy", CLEAN / "extra" / "oga_meta.jsonl", "OpenGameArt", "CC0 1.0"),
    ):
        if not ip.exists():
            log("missing", ip)
            continue
        im = np.load(ip, mmap_mode="r")
        meta = load_jsonl(mp)
        assert len(im) == len(meta), (name, len(im), len(meta))
        blk, semi = snap_canonical(np.asarray(im))
        for i, r in enumerate(meta):
            rec = dict(r)
            rec["source"] = rec.get("source") or src
            rec["licence"] = rec.get("licence") or lic
            rec["synthetic"] = False
            rec["semi_ratio"] = float(semi[i] if np.ndim(semi) else semi)
            recs.append(rec)
            imgs.append(blk[i])
        log(f"extra {name}: {len(meta)}")
    return imgs, recs


def stage_gather(thr, pal_lab):
    """Assemble pre-dedup snapped images + meta into work_v2/pool_*."""
    ensure()
    dest_i, dest_m = WORK / "pool_images.npy", WORK / "pool_meta.jsonl"
    if dest_i.exists() and dest_m.exists() and (WORK / "pool_stats.json").exists():
        log("gather: already done")
        return json.loads((WORK / "pool_stats.json").read_text())
    recs_all, chunks = [], []
    # azyg kept
    a_imgs = np.load(CLEAN / "azyg" / "images.npy", mmap_mode="r")
    a_meta = load_jsonl(CLEAN / "azyg" / "meta.jsonl")
    drop = thr["drop_mask"]
    crisp = thr["crisp_mask"]
    K = np.load(WORK / "azyg_profile.npz")["K"]
    mse = np.load(WORK / "azyg_profile.npz")["mse"]
    semi = np.load(WORK / "azyg_profile.npz")["semi"]
    keep_a = np.where(~drop)[0]
    log(f"gather azyg keep {len(keep_a)}/{len(a_meta)}")
    # snap in chunks and append
    kept_imgs = []
    CH = 4096
    for s in range(0, len(keep_a), CH):
        ix = keep_a[s:s + CH]
        blk, _ = snap_canonical(np.asarray(a_imgs[ix]))
        kept_imgs.append(blk)
        for j, i in enumerate(ix):
            r = dict(a_meta[int(i)])
            r["source"] = r.get("source") or "azyg0s/pixelart-172k"
            r["licence"] = r.get("licence") or "MIT"
            r["synthetic"] = True
            r["K_img"] = int(K[i])
            r["oklab_mse"] = float(mse[i])
            r["semi_ratio"] = float(semi[i])
            r["crisp"] = bool(crisp[i])
            r["azyg_src_idx"] = int(i)
            recs_all.append(r)
    # pixl
    p_imgs = np.load(WORK / "pixl_images.npy")
    p_meta = load_jsonl(WORK / "pixl_meta.jsonl")
    p_K, p_mse, _ = unique_opaque_and_mse(p_imgs, pal_lab)
    for i, r in enumerate(p_meta):
        rr = dict(r)
        rr["K_img"] = int(p_K[i])
        rr["oklab_mse"] = float(p_mse[i])
        rr["crisp"] = bool(p_K[i] <= 32)
        rr["synthetic"] = False
        recs_all.append(rr)
    kept_imgs.append(p_imgs)
    # extra
    e_imgs, e_meta = load_extra()
    if e_imgs:
        e_arr = np.stack(e_imgs, 0)
        e_K, e_mse, _ = unique_opaque_and_mse(e_arr, pal_lab)
        for i, r in enumerate(e_meta):
            rr = dict(r)
            rr["K_img"] = int(e_K[i])
            rr["oklab_mse"] = float(e_mse[i])
            rr["crisp"] = bool(e_K[i] <= 32)
            rr["synthetic"] = False
            recs_all.append(rr)
        kept_imgs.append(e_arr)
    pool = np.concatenate(kept_imgs, 0)
    assert len(pool) == len(recs_all)
    for i, r in enumerate(recs_all):
        r["idx"] = i
        r["md5"] = hashlib.md5(np.ascontiguousarray(pool[i]).tobytes()).hexdigest()
    save_u8(dest_i, pool)
    write_jsonl(dest_m, recs_all)
    by_src = Counter(r["source"] for r in recs_all)
    stats = {
        "n": len(recs_all),
        "by_source": dict(by_src),
        "synthetic": int(sum(1 for r in recs_all if r.get("synthetic"))),
        "real": int(sum(1 for r in recs_all if not r.get("synthetic"))),
        "crisp": int(sum(1 for r in recs_all if r.get("crisp"))),
    }
    (WORK / "pool_stats.json").write_text(json.dumps(stats, indent=2))
    log("gather", stats)
    return stats


def stage_dedup_holdout():
    ensure()
    if (WORK / "keep_idx.npy").exists() and (WORK / "dedup_holdout.json").exists():
        log("dedup_holdout: already done")
        return json.loads((WORK / "dedup_holdout.json").read_text())
    imgs = np.load(WORK / "pool_images.npy", mmap_mode="r")
    recs = load_jsonl(WORK / "pool_meta.jsonl")
    n = len(recs)
    t0 = time.time()
    # hashes
    hashes_p = WORK / "phash.npy"
    hashes_f = WORK / "phash_flip.npy"
    if hashes_p.exists() and hashes_f.exists() and np.load(hashes_p).shape[0] == n:
        H = np.load(hashes_p)
        F = np.load(hashes_f)
        log("loaded cached phashes")
    else:
        H = np.zeros((n, 4), np.uint64)
        F = np.zeros((n, 4), np.uint64)
        for i in range(n):
            H[i], F[i] = image_phashes(imgs[i])
            if (i + 1) % 10000 == 0 or i + 1 == n:
                log(f"  phash {i+1}/{n} {time.time()-t0:.0f}s")
        np.save(hashes_p, H)
        np.save(hashes_f, F)

    # exact MD5 groups (already canonical)
    by_md5 = defaultdict(list)
    for i, r in enumerate(recs):
        by_md5[r["md5"]].append(i)
    exact_groups = [v for v in by_md5.values() if len(v) > 1]

    # pHash clusters, no colour check, hflip-invariant
    # LSH: 8 bands of 32 bits from the 256-bit hash (and from the flip hash)
    parent = np.arange(n, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    T = 16  # 256-bit Hamming; looser than old T=4 because recolours must collide for split purposes
    # band keys from both orientations
    def bands_of(h):
        # h uint64[4] -> 8 x uint32
        out = []
        for v in h:
            v = int(v)
            out.append(v & 0xFFFFFFFF)
            out.append((v >> 32) & 0xFFFFFFFF)
        return out

    buckets = defaultdict(list)
    for i in range(n):
        for bi, key in enumerate(bands_of(H[i]) + bands_of(F[i])):
            buckets[(bi, int(key))].append(i)
    compared = 0
    phash_links = 0
    seen_pairs = set()
    for ids in buckets.values():
        if len(ids) < 2 or len(ids) > 400:
            continue
        uniq = list(dict.fromkeys(ids))
        for a in range(len(uniq)):
            ia = uniq[a]
            for b in range(a + 1, len(uniq)):
                ib = uniq[b]
                if ia > ib:
                    ia, ib = ib, ia
                if (ia, ib) in seen_pairs:
                    continue
                seen_pairs.add((ia, ib))
                compared += 1
                if ham_hflip(H[ia], F[ia], H[ib], F[ib]) <= T:
                    union(ia, ib)
                    phash_links += 1
    clusters = defaultdict(list)
    for i in range(n):
        clusters[int(find(i))].append(i)
    cl_list = list(clusters.values())
    log(f"dedup: exact_groups={len(exact_groups)} phash_links={phash_links} compared={compared} clusters={len(cl_list)} T={T} {time.time()-t0:.0f}s")

    # ---- holdout: caption phrase + CLIP text similarity
    def read_hold(name):
        p = HOLD / name
        if not p.exists():
            p = HOLD.parent / name
        return [l.strip().lower() for l in p.read_text().splitlines() if l.strip()]
    blind = read_hold("blind50.txt")
    ev = read_hold("eval50.txt")
    assert len(blind) == 50 and len(ev) == 50
    blind_re = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in blind) + r")\b", re.I)
    eval_re = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in ev) + r")\b", re.I)
    eval_set = set(ev)
    prefs = ("pixel art icon of a ", "pixel art icon of ", "pixel art of a ", "pixel art of ", "monochrome ")

    def strip_pref(s):
        c = " ".join((s or "").lower().split())
        for pref in prefs:
            if c.startswith(pref):
                c = c[len(pref):]
                break
        return c

    phrase_reason = [None] * n
    for i, r in enumerate(recs):
        for s in (r.get("caption") or "", r.get("text") or "", r.get("caption_original") or ""):
            c = strip_pref(s)
            if not c and not s:
                continue
            if c in eval_set or (s or "").lower().strip() in eval_set:
                phrase_reason[i] = "eval50_exact"
                break
            if blind_re.search(s or "") or blind_re.search(c):
                phrase_reason[i] = "blind50_phrase"
                break
            if eval_re.search(s or "") or eval_re.search(c):
                phrase_reason[i] = "eval50_phrase"
                break

    # CLIP text embeddings on CPU
    emb_p = WORK / "clip_text_emb.npy"
    sim_p = WORK / "clip_holdout_sim.npy"
    if emb_p.exists() and sim_p.exists() and np.load(sim_p).shape[0] == n:
        maxsim = np.load(sim_p)
        log("loaded cached clip holdout sims")
    else:
        sys.path.insert(0, str(V3_CODE))
        import torch
        from tokenizers import Tokenizer
        import v3_model as M
        import s7_data as D
        tok = Tokenizer.from_file(str(TOK_PATH))

        def encode_t18(strings):
            out = np.full((len(strings), L26), EOS, np.uint16)
            for k, e in enumerate(tok.encode_batch(list(strings), add_special_tokens=False)):
                ids = e.ids[:TEXT_IDS]
                out[k, 0] = BOS
                out[k, 1:1 + len(ids)] = ids
                out[k, 1 + len(ids)] = EOS
            return D.trim(out, TEXT_LEN)

        caps = [(r.get("caption") or r.get("text") or "") for r in recs]
        probes = []
        probe_tag = []
        for p in blind:
            for t in (p, f"pixel art icon of a {p}", f"pixel art of {p}"):
                probes.append(t)
                probe_tag.append("blind")
        for p in ev:
            for t in (p, f"pixel art icon of a {p}", f"pixel art of {p}"):
                probes.append(t)
                probe_tag.append("eval")
        log(f"CLIP-L text CPU: {n} captions + {len(probes)} probes")
        clip = M.ClipTextPenultimate()
        sd = torch.load(CLIP_PT, map_location="cpu", weights_only=False)
        clip.load_hf(sd)
        del sd
        clip.to(torch.device("cpu")).eval()

        def embed_ids(t18, batch=64):
            E = np.zeros((len(t18), 768), np.float32)
            with torch.no_grad():
                for s in range(0, len(t18), batch):
                    ids = torch.from_numpy(t18[s:s + batch].astype(np.int64))
                    h = clip(ids).float()  # [b,18,768]
                    # EOS-position feature (first EOS at or after pos 1)
                    rows = t18[s:s + batch]
                    eos_pos = np.argmax(rows == EOS, axis=1)
                    eos_pos = np.where(rows[:, 0] == EOS, 0, eos_pos)
                    idx = torch.from_numpy(eos_pos.astype(np.int64))
                    feat = h[torch.arange(h.shape[0]), idx]
                    feat = torch.nn.functional.normalize(feat, dim=-1)
                    E[s:s + h.shape[0]] = feat.numpy()
                    if (s // batch) % 50 == 0 or s + h.shape[0] == len(t18):
                        log(f"  embed {s + h.shape[0]}/{len(t18)}")
            return E

        cap_ids = encode_t18(caps)
        pr_ids = encode_t18(probes)
        E = embed_ids(cap_ids)
        P = embed_ids(pr_ids, batch=32)
        maxsim = (E @ P.T).max(axis=1).astype(np.float32)
        np.save(emb_p, E)
        np.save(sim_p, maxsim)
        np.save(WORK / "clip_probe_emb.npy", P)
        (WORK / "clip_probe_tags.json").write_text(json.dumps(probe_tag))
        del clip, E, P
        log("clip sims done")

    # threshold from phrase-positive vs the rest
    phrase_pos = np.array([r is not None for r in phrase_reason])
    if phrase_pos.any() and (~phrase_pos).any():
        # aim to catch near-paraphrases: mid-point between p10 of known hits and p99 of non-hits, clamped
        p_hit = float(np.quantile(maxsim[phrase_pos], 0.10))
        p_rest = float(np.quantile(maxsim[~phrase_pos], 0.999))
        t_sim = float(max(0.88, min(0.96, 0.5 * (p_hit + p_rest))))
    else:
        t_sim = 0.92
    clip_hit = maxsim >= t_sim
    # reasons
    hold_reason = phrase_reason[:]
    for i in range(n):
        if hold_reason[i] is None and clip_hit[i]:
            hold_reason[i] = "clip_text_sim"
    # visual clusters: if any member is holdout, all are holdout (never across train/holdout), including recolours
    cluster_hold = 0
    for members in cl_list:
        if any(hold_reason[i] is not None for i in members):
            why = next(hold_reason[i] for i in members if hold_reason[i] is not None)
            for i in members:
                if hold_reason[i] is None:
                    hold_reason[i] = "visual_cluster_of_" + why
                    cluster_hold += 1
    n_hold = sum(1 for r in hold_reason if r is not None)

    # keep: drop exact-MD5 duplicates (prefer real, then crisp, then lower mse, then longer caption)
    # but do NOT collapse pHash recolours inside the kept (non-holdout) train set
    def pref(i):
        r = recs[i]
        return (0 if r.get("synthetic") else 1,
                1 if r.get("crisp") else 0,
                -(r.get("oklab_mse") or 0.0),
                len(r.get("caption") or ""))

    drop_exact = np.zeros(n, bool)
    for g in exact_groups:
        # if the group mixes holdout and non-holdout, all non-holdout already marked visual? MD5 exact is a cluster too
        # drop extras, never keep a holdout row
        usable = [i for i in g if hold_reason[i] is None]
        if not usable:
            for i in g:
                drop_exact[i] = True  # all holdout
            continue
        usable.sort(key=pref, reverse=True)
        for i in usable[1:]:
            drop_exact[i] = True
        for i in g:
            if hold_reason[i] is not None:
                drop_exact[i] = True  # will be dropped as holdout anyway

    keep = [i for i in range(n) if hold_reason[i] is None and not drop_exact[i]]
    np.save(WORK / "keep_idx.npy", np.array(keep, np.int32))
    reasons = Counter(r or "keep" for r in hold_reason)
    # also count exact dropped among non-holdout
    n_exact = int(sum(1 for i in range(n) if drop_exact[i] and hold_reason[i] is None))
    report = {
        "in": n,
        "kept": len(keep),
        "dropped_holdout": int(n_hold),
        "dropped_exact_md5": n_exact,
        "phash_T": T,
        "phash_links": phash_links,
        "phash_clusters": len(cl_list),
        "clip_t_sim": t_sim,
        "clip_sim_phrase_hit_p10": float(np.quantile(maxsim[phrase_pos], 0.10)) if phrase_pos.any() else None,
        "clip_sim_nonphrase_p99": float(np.quantile(maxsim[~phrase_pos], 0.99)) if (~phrase_pos).any() else None,
        "clip_sim_nonphrase_p999": float(np.quantile(maxsim[~phrase_pos], 0.999)) if (~phrase_pos).any() else None,
        "holdout_reasons": dict(reasons),
        "visual_cluster_extra": cluster_hold,
        "kept_by_source": dict(Counter(recs[i]["source"] for i in keep)),
        "kept_synthetic": int(sum(1 for i in keep if recs[i].get("synthetic"))),
        "kept_real": int(sum(1 for i in keep if not recs[i].get("synthetic"))),
        "in_by_source": dict(Counter(r["source"] for r in recs)),
        "seconds": round(time.time() - t0, 1),
    }
    # store reasons for audit
    audit = []
    for i, why in enumerate(hold_reason):
        if why:
            audit.append({"idx": i, "reason": why, "sim": float(maxsim[i]), "caption": recs[i].get("caption"), "source": recs[i].get("source")})
    write_jsonl(WORK / "holdout_dropped.jsonl", audit)
    (WORK / "dedup_holdout.json").write_text(json.dumps(report, indent=2))
    log("dedup_holdout", json.dumps(report))
    return report


def stage_write_out():
    ensure()
    keep = np.load(WORK / "keep_idx.npy")
    imgs = np.load(WORK / "pool_images.npy", mmap_mode="r")
    recs = load_jsonl(WORK / "pool_meta.jsonl")
    kept_imgs = np.asarray(imgs[keep])
    kept_recs = []
    for new_i, old in enumerate(keep.tolist()):
        r = dict(recs[int(old)])
        r["idx"] = new_i
        r["pool_idx"] = int(old)
        kept_recs.append(r)
    # sampling weights: real CC0 upweighted 3-5x aiming for 25-35% of steps
    is_real_cc0 = np.array([
        (not r.get("synthetic")) and ("cc0" in (r.get("licence") or "").lower().replace(" ", ""))
        for r in kept_recs
    ], dtype=bool)
    n = len(kept_recs)
    n_r = int(is_real_cc0.sum())
    n_o = n - n_r
    # w*n_r / (w*n_r + n_o) = 0.30
    target = 0.30
    if n_r == 0:
        w = 1.0
        share = 0.0
    else:
        w = (target * n_o) / ((1 - target) * n_r)
        w = float(min(5.0, max(3.0, w)))
        share = (w * n_r) / (w * n_r + n_o)
    for r, flag in zip(kept_recs, is_real_cc0):
        r["sample_weight"] = w if flag else 1.0
        r["crisp"] = bool(r.get("crisp"))
    save_u8(OUT / "images.npy", kept_imgs)
    write_jsonl(OUT / "meta.jsonl", kept_recs)
    fields = ["path", "source", "licence", "caption", "synthetic", "split", "backend", "pack",
              "creator", "source_url", "caption_source", "category", "md5", "sample_weight",
              "crisp", "K_img", "oklab_mse"]
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in kept_recs:
            row = dict(r)
            row["path"] = f"out_v2/images.npy:{r['idx']:06d}"
            row["synthetic"] = "1" if r.get("synthetic") else "0"
            row["crisp"] = "1" if r.get("crisp") else "0"
            wr.writerow(row)
    by_src = Counter(r["source"] for r in kept_recs)
    by_lic = Counter(r.get("licence") or "unknown" for r in kept_recs)
    stats = {
        "n": n,
        "synthetic": int(sum(1 for r in kept_recs if r.get("synthetic"))),
        "real": int(sum(1 for r in kept_recs if not r.get("synthetic"))),
        "real_cc0": n_r,
        "crisp": int(sum(1 for r in kept_recs if r.get("crisp"))),
        "by_source": dict(by_src),
        "by_licence": dict(by_lic),
        "sample_weight_real_cc0": w,
        "expected_real_cc0_step_share": share,
        "target_step_share": target,
        "weight_clamped_to_3_5x": True,
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    (OUT / "weights.json").write_text(json.dumps({
        "real_cc0_weight": w, "other_weight": 1.0,
        "n_real_cc0": n_r, "n_other": n_o, "expected_step_share_real_cc0": share,
    }, indent=2))
    log("wrote out_v2", stats)
    return stats, kept_imgs, kept_recs


def quant_report_real(kept_recs):
    reals = [r for r in kept_recs if not r.get("synthetic")]
    if not reals:
        return {}
    mse = np.array([r.get("oklab_mse") or 0.0 for r in reals], np.float64)
    K = np.array([r.get("K_img") or 0 for r in reals], np.int32)
    rep = {
        "n_real": len(reals),
        "oklab_mse_mean": float(mse.mean()),
        "oklab_mse_median": float(np.median(mse)),
        "oklab_mse_p90": float(np.percentile(mse, 90)),
        "oklab_mse_p99": float(np.percentile(mse, 99)),
        "oklab_mse_max": float(mse.max()),
        "K_median": float(np.median(K)),
        "K_p90": float(np.percentile(K, 90)),
        "frac_crisp": float(np.mean([bool(r.get("crisp")) for r in reals])),
        "by_source": {},
    }
    by = defaultdict(list)
    for r in reals:
        by[r["source"]].append(r.get("oklab_mse") or 0.0)
    for s, vs in by.items():
        a = np.array(vs, np.float64)
        rep["by_source"][s] = {"n": len(vs), "mse_mean": float(a.mean()), "mse_median": float(np.median(a)), "mse_p99": float(np.percentile(a, 99))}
    (OUT / "quantization_real_cc0.json").write_text(json.dumps(rep, indent=2))
    log("quant real", json.dumps(rep))
    return rep


def stage_sample_sheet(kept_imgs, kept_recs):
    rng = np.random.default_rng(32)
    n = len(kept_recs)
    # 32 real + 32 synthetic if possible, else random 64
    real = [i for i, r in enumerate(kept_recs) if not r.get("synthetic")]
    syn = [i for i, r in enumerate(kept_recs) if r.get("synthetic")]
    take = []
    if real and syn:
        take += list(rng.choice(real, size=min(32, len(real)), replace=False))
        take += list(rng.choice(syn, size=min(32, len(syn)), replace=False))
    else:
        take = list(rng.choice(n, size=min(64, n), replace=False))
    take = take[:64]
    cells = [kept_imgs[i] for i in take]
    labs = []
    for i in take:
        r = kept_recs[i]
        tag = "R" if not r.get("synthetic") else "S"
        labs.append(f"{tag} K={r.get('K_img')} {('crisp' if r.get('crisp') else '')}")
    path = OUT / "sheets" / "sample64.png"
    contact_sheet(cells, labs, path, scale=6, cols=8, title="data32 out_v2 sample 64  (R=real S=synthetic)")
    (OUT / "sheets" / "sample64.json").write_text(json.dumps([
        {"idx": int(i), "source": kept_recs[i].get("source"), "caption": kept_recs[i].get("caption"),
         "K_img": kept_recs[i].get("K_img"), "crisp": kept_recs[i].get("crisp"), "synthetic": kept_recs[i].get("synthetic")}
        for i in take
    ], indent=2))
    return path


def stage_img_idx(kept_imgs, pal, pal_lab):
    ensure()
    dest = OUT / "cache" / "img_idx.npy"
    if dest.exists() and np.load(dest, mmap_mode="r").shape[0] == len(kept_imgs):
        log("img_idx already done")
        return
    n = len(kept_imgs)
    op = kept_imgs[..., 3] >= 128
    packed = pack_rgb(kept_imgs[..., :3])
    uniq = np.unique(packed[op]) if op.any() else np.zeros((0,), np.uint32)
    cols = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1).astype(np.uint8)
    log(f"img_idx unique opaque colours {len(uniq)}")
    ids = nn_oklab(cols, pal_lab, skip0=True, chunk=2048)
    order = np.argsort(uniq)
    keys, vals = uniq[order], ids[order]
    tmp = dest.with_suffix(".tmp.npy")
    tok = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16, shape=(n, 1024))
    CH = 4096
    for s in range(0, n, CH):
        e = min(n, s + CH)
        p = packed[s:e]
        a = kept_imgs[s:e, ..., 3]
        idx = np.searchsorted(keys, p) if len(keys) else np.zeros_like(p)
        if len(keys):
            idx = np.clip(idx, 0, len(keys) - 1)
            hit = keys[idx] == p
            out = np.zeros(p.shape, np.uint16)
            out[hit] = vals[idx[hit]]
        else:
            out = np.zeros(p.shape, np.uint16)
            hit = np.zeros(p.shape, bool)
        out[a < 128] = 0
        miss = (a >= 128) & ~hit
        if miss.any():
            # shouldn't happen
            coords = np.argwhere(miss)
            for cy, cx in coords:
                # cy is (n,y,x) wait argwhere on [b,32,32]
                pass
            # vectorised fallback
            mcol = np.stack([(p[miss] >> 16) & 255, (p[miss] >> 8) & 255, p[miss] & 255], 1).astype(np.uint8)
            out[miss] = nn_oklab(mcol, pal_lab, skip0=True)
        tok[s:e] = out.reshape(e - s, 1024)
        if e % 20000 == 0 or e == n:
            log(f"  img_idx {e}/{n}")
    tok.flush()
    del tok
    tmp.replace(dest)
    log("img_idx written", dest, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma stages, empty=all")
    args = ap.parse_args()
    t0 = time.time()
    ensure()
    pal = np.load(PAL_PATH)
    assert pal.shape == (4097, 3), pal.shape
    pal_lab = palette_oklab_tables(pal)
    prof = stage_profile_azyg(pal_lab)
    # npz values are arrays; after reload from disk via dict(np.load) ok
    if not isinstance(prof["K"], np.ndarray):
        prof = dict(np.load(WORK / "azyg_profile.npz"))
    thr = choose_thresholds(prof)
    stage_boundary_sheet(thr)
    stage_refilter_pixl()
    stage_gather(thr, pal_lab)
    stage_dedup_holdout()
    stats, kept_imgs, kept_recs = stage_write_out()
    qrep = quant_report_real(kept_recs)
    stage_sample_sheet(kept_imgs, kept_recs)
    stage_img_idx(kept_imgs, pal, pal_lab)
    # copy sheets also under out_v2 root for easy scp
    shutil.copy2(OUT / "sheets" / "boundary_kept_vs_dropped.png", OUT / "boundary_kept_vs_dropped.png")
    shutil.copy2(OUT / "sheets" / "sample64.png", OUT / "sample64.png")
    summary = {
        "seconds": round(time.time() - t0, 1),
        "thresholds": {k: thr[k] for k in thr if k not in ("drop_mask", "crisp_mask")},
        "out": stats,
        "quant_real": qrep,
        "dedup": json.loads((WORK / "dedup_holdout.json").read_text()),
        "pixl_filter": json.loads((WORK / "pixl_filter.json").read_text()),
        "pool": json.loads((WORK / "pool_stats.json").read_text()),
        "before_out_v1": {
            "n": 187954,
            "by_source": {
                "azyg0s/pixelart-172k": 171171,
                "ocdbytes/pixl-data": 6641,
                "kenney.nl": 219,
                "OpenGameArt": 9923,
            },
            "note": "previous out/ left untouched",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log("DONE", json.dumps({k: summary[k] for k in ("seconds", "out")}, default=str))


if __name__ == "__main__":
    main()
