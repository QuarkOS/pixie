#!/usr/bin/env python3
"""Pixie-32 data prep. CPU-only, resume-friendly stages.

Stages: filter_azyg, filter_pixl, scrape_extra, dedup, recaption, manifest, cache, sheet
Never writes outside $PIXIE_ROOT/data32 (and a copy of the contact sheet).
Never touches runs/, v3/, cache/, or chat.
"""
from __future__ import annotations

import argparse, csv, hashlib, io, json, os, re, sys, time, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ.get("DATA32_ROOT", os.path.expandvars("$PIXIE_ROOT/data32")))
RAW = ROOT / "raw"
CLEAN = ROOT / "clean"
OUT = ROOT / "out"
HOLD = Path(__file__).resolve().parent / "holdout"
AZYG = RAW / "pixelart-172k"
PIXL = RAW / "pixl-data"

# Magenta-key leftovers as documented by azyg0s: e.g. (150,0,140) on outline corners,
# plus classic #FF00FF. Interior purple objects must NOT be dropped.
MAG_MAX_FRINGE = 12  # isolated outline key-pixels above this are still fringe if no interior mass


def log(*a):
    print(*a, flush=True)


def ensure_dirs():
    for p in (CLEAN / "azyg", CLEAN / "pixl", CLEAN / "extra", OUT, OUT / "cache", ROOT / "work"):
        p.mkdir(parents=True, exist_ok=True)


def load_holdouts():
    def read(name):
        p = HOLD / name
        if not p.exists():
            p = HOLD.parent / name
        lines = [l.strip().lower() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        return lines
    blind = read("blind50.txt")
    ev = read("eval50.txt")
    assert len(blind) == 50 and len(ev) == 50, (len(blind), len(ev))
    return blind, ev


def holdout_regex(phrases):
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b", re.I)


def caption_is_holdout(caption: str, text: str, blind_re, eval_set) -> str | None:
    for s in (caption or "", text or ""):
        c = " ".join(s.lower().split())
        if not c:
            continue
        # strip common templates
        for pref in ("pixel art icon of a ", "pixel art icon of ", "pixel art of a ", "pixel art of ",
                     "monochrome "):
            if c.startswith(pref):
                c = c[len(pref):]
        if c in eval_set or s.lower().strip() in eval_set:
            return "eval50_exact"
        if blind_re.search(s) or blind_re.search(c):
            return "blind50_phrase"
        # also treat eval50 as a phrase, so "pixel art icon of a red potion" is held out
        # (v3 used exact-only for eval50; we are stricter because these must never be trained on)
        if holdout_regex.__wrapped__ if False else False:
            pass
    return None


def eval_phrase_hit(s: str, eval_re) -> bool:
    return bool(eval_re.search(s or ""))


# ---------- images ----------

def pil_from_cell(cell) -> Image.Image | None:
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
        p = cell.get("path")
        if p and Path(p).exists():
            return Image.open(p).convert("RGBA")
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(cell))).convert("RGBA")
    return None


def to_rgba32(im: Image.Image) -> np.ndarray | None:
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    if im.size != (32, 32):
        return None
    return np.asarray(im, dtype=np.uint8)


def binary_alpha(arr: np.ndarray) -> bool:
    a = arr[..., 3]
    return bool(np.isin(a, (0, 255)).all())


def magenta_fringe(arr: np.ndarray) -> dict:
    """Return {n_key, n_outline_key, n_interior_key, has_fringe}."""
    r, g, b, a = arr[..., 0].astype(np.int16), arr[..., 1].astype(np.int16), arr[..., 2].astype(np.int16), arr[..., 3]
    opaque = a >= 128
    # Key-magenta: strong R+B, little G, R and B close. Covers (150,0,140) and (255,0,255).
    key = opaque & (g <= 40) & (r >= 100) & (b >= 100) & (np.abs(r - b) <= 50) & ((np.maximum(r, b) - g) >= 80)
    if not key.any():
        return {"n_key": 0, "n_outline_key": 0, "n_interior_key": 0, "has_fringe": False}
    # Transparent or nearly-transparent neighbor (4-connected), with edge pixels counting as outline.
    trans = a <= 16
    pad = np.pad(trans, 1, constant_values=True)
    outline = opaque & (pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:])
    n_key = int(key.sum())
    n_out = int((key & outline).sum())
    n_in = int((key & ~outline).sum())
    # Fringe: a few leftover key pixels on the outline, and no real magenta interior mass.
    has = (n_out > 0) and (n_in <= 2) and (n_out <= MAG_MAX_FRINGE)
    # Also drop if the ONLY opaque magenta is a handful of outline specks even if n_in is 0.
    return {"n_key": n_key, "n_outline_key": n_out, "n_interior_key": n_in, "has_fringe": bool(has)}


def dhash64(arr: np.ndarray, size: int = 8) -> int:
    """Perceptual difference hash; NEAREST resize so pixel art stays crisp."""
    im = Image.fromarray(arr, "RGBA").convert("L").resize((size + 1, size), Image.Resampling.NEAREST)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    h = 0
    for i, v in enumerate(bits):
        if v:
            h |= 1 << i
    return h


def ahash64(arr: np.ndarray, size: int = 8) -> int:
    im = Image.fromarray(arr, "RGBA").convert("L").resize((size, size), Image.Resampling.NEAREST)
    a = np.asarray(im, dtype=np.int16)
    bits = (a.flatten() > a.mean())
    h = 0
    for i, v in enumerate(bits):
        if v:
            h |= 1 << i
    return h


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def rgba_md5(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def save_memmap_u8(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=arr.shape)
    mm[:] = arr
    mm.flush()
    del mm
    tmp.replace(path)


# ---------- stages ----------

def iter_parquet_rows(files, columns):
    for fp in files:
        pf = pq.ParquetFile(fp)
        log(f"  reading {fp.name} rows={pf.metadata.num_rows} groups={pf.num_row_groups}")
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=columns)
            yield fp, table


def stage_filter_azyg():
    ensure_dirs()
    out_dir = CLEAN / "azyg"
    stats_p = out_dir / "filter_stats.json"
    img_p = out_dir / "images.npy"
    meta_p = out_dir / "meta.jsonl"
    if img_p.exists() and meta_p.exists() and stats_p.exists():
        log("filter_azyg: already done, skip")
        return json.loads(stats_p.read_text())

    files = sorted((AZYG / "data").glob("*.parquet"))
    if not files:
        raise SystemExit(f"no azyg parquets in {AZYG/'data'}")
    # First pass: we don't know keep count; allocate for all rows then shrink.
    n_total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    log(f"filter_azyg: {n_total} rows in {len(files)} files")
    tmp = out_dir / "images.npy.partial"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n_total, 32, 32, 4))
    stats = Counter()
    by_split = Counter()
    by_backend = Counter()
    drop_backend_mag = Counter()
    keep_backend = Counter()
    fringe_examples = []
    k = 0
    meta_f = open(out_dir / "meta.jsonl.partial", "w", encoding="utf-8")
    cols = None
    t0 = time.time()
    done = 0
    for fp, table in iter_parquet_rows(files, None):
        # infer split from filename: train-0000, val-0000, val_unseen-...
        name = fp.name
        if name.startswith("train"):
            split = "train"
        elif name.startswith("val_"):
            split = name.split("-")[0]  # val_unseen / val_binding / ...
        elif name.startswith("val"):
            split = "val"
        else:
            split = "unknown"
        n = table.num_rows
        # convert to pylist in reasonably small tables (row groups)
        rows = table.to_pylist()
        for i, row in enumerate(rows):
            done += 1
            stats["seen"] += 1
            by_split[split] += 1
            backend = row.get("backend") or ""
            by_backend[backend] += 1
            try:
                im = pil_from_cell(row.get("image"))
            except Exception:
                stats["drop_decode"] += 1
                continue
            if im is None:
                stats["drop_noimage"] += 1
                continue
            arr = to_rgba32(im)
            if arr is None:
                stats["drop_not_32"] += 1
                continue
            if not binary_alpha(arr):
                stats["drop_soft_alpha"] += 1
                continue
            fr = magenta_fringe(arr)
            if fr["has_fringe"]:
                stats["drop_magenta_fringe"] += 1
                drop_backend_mag[backend] += 1
                if len(fringe_examples) < 12:
                    fringe_examples.append({
                        "split": split, "backend": backend,
                        "caption": row.get("caption"), "file": row.get("file"),
                        **fr,
                    })
                continue
            rec = {
                "idx": k,
                "split": split,
                "source": "azyg0s/pixelart-172k",
                "licence": "MIT",
                "synthetic": True,
                "backend": backend,
                "caption": row.get("caption") or "",
                "caption_alt": row.get("caption_alt") or "",
                "text": row.get("text") or "",
                "category": row.get("category") or "",
                "hint": row.get("hint") or "",
                "sheet_id": row.get("sheet_id"),
                "file": row.get("file"),
                "native_width": row.get("native_width"),
                "native_height": row.get("native_height"),
                "purity": row.get("purity"),
                "n_colors": row.get("n_colors"),
                "flags": row.get("flags") or [],
                "md5": rgba_md5(arr),
                "dhash": dhash64(arr),
                "ahash": ahash64(arr),
            }
            mm[k] = arr
            k += 1
            keep_backend[backend] += 1
            stats["kept"] += 1
            meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if done % 20000 < n:
            log(f"  progress {done}/{n_total} kept={k} elapsed={time.time()-t0:.0f}s")
    meta_f.close()
    mm.flush()
    del mm
    # shrink
    if k == 0:
        raise SystemExit("filter_azyg kept 0 rows")
    src = np.load(tmp, mmap_mode="r")
    save_memmap_u8(img_p, np.asarray(src[:k]))
    del src
    tmp.unlink(missing_ok=True)
    (out_dir / "meta.jsonl.partial").replace(meta_p)
    out_stats = {
        "rows_in": n_total,
        "kept": k,
        "counters": dict(stats),
        "by_split_in": dict(by_split),
        "by_backend_in": dict(by_backend),
        "kept_by_backend": dict(keep_backend),
        "dropped_magenta_by_backend": dict(drop_backend_mag),
        "fringe_examples": fringe_examples,
        "seconds": round(time.time() - t0, 1),
        "images": str(img_p),
    }
    stats_p.write_text(json.dumps(out_stats, indent=2, ensure_ascii=False))
    log("filter_azyg done", json.dumps({x: out_stats[x] for x in ("rows_in", "kept", "counters", "seconds")}))
    return out_stats


def pixl_licence_ok(license_s: str, source_url: str, pack: str) -> tuple[bool, str]:
    lic = (license_s or "").strip()
    url = (source_url or "").lower()
    pack_l = (pack or "").lower()
    low = lic.lower().replace(" ", "")
    is_cc0 = ("cc0" in low) or ("cc-0" in low) or low.startswith("cc01.0")
    is_ccby3 = ("ccby3" in low.replace(".", "")) or ("cc-by-3" in low) or ("cc by 3" in lic.lower())
    is_gameicons = ("game-icons.net" in url) or pack_l.startswith("game-icons") or "game-icons.net" in pack_l
    if is_cc0:
        return True, "CC0 1.0"
    if is_ccby3 and is_gameicons:
        return True, "CC BY 3.0"
    return False, lic or "unknown"


def stage_filter_pixl():
    ensure_dirs()
    out_dir = CLEAN / "pixl"
    stats_p = out_dir / "filter_stats.json"
    img_p = out_dir / "images.npy"
    meta_p = out_dir / "meta.jsonl"
    if img_p.exists() and meta_p.exists() and stats_p.exists():
        log("filter_pixl: already done, skip")
        return json.loads(stats_p.read_text())

    files = []
    for split, folder in (("train", PIXL / "sprites" / "train-00000-of-00001.parquet"),
                          ("test", PIXL / "sprites" / "test-00000-of-00001.parquet")):
        if folder.exists():
            files.append((split, folder))
    if not files:
        raise SystemExit(f"no pixl sprite parquets in {PIXL/'sprites'}")
    n_total = sum(pq.ParquetFile(f).metadata.num_rows for _, f in files)
    log(f"filter_pixl: {n_total} sprite rows (scenes excluded)")
    tmp = out_dir / "images.npy.partial"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n_total, 32, 32, 4))
    stats = Counter()
    by_pack = Counter()
    by_lic = Counter()
    by_size = Counter()
    kept_pack = Counter()
    k = 0
    meta_f = open(out_dir / "meta.jsonl.partial", "w", encoding="utf-8")
    t0 = time.time()
    for split, fp in files:
        table = pq.read_table(fp)
        rows = table.to_pylist()
        log(f"  {split} {fp.name} n={len(rows)}")
        for row in rows:
            stats["seen"] += 1
            pack = row.get("pack") or ""
            lic_raw = row.get("license") or ""
            url = row.get("source_url") or ""
            w, h = str(row.get("width") or ""), str(row.get("height") or "")
            by_size[f"{w}x{h}"] += 1
            by_pack[pack] += 1
            by_lic[lic_raw] += 1
            ok, lic_norm = pixl_licence_ok(lic_raw, url, pack)
            if not ok:
                stats["drop_licence"] += 1
                continue
            try:
                im = pil_from_cell(row.get("image"))
            except Exception:
                stats["drop_decode"] += 1
                continue
            if im is None:
                stats["drop_noimage"] += 1
                continue
            # native 32x32, or game-icons 30x30 on a 32x32 canvas already stored as 32x32
            if im.size != (32, 32):
                stats["drop_not_32"] += 1
                continue
            # If metadata says another size but canvas is 32x32 (game-icons), keep.
            arr = to_rgba32(im)
            if arr is None:
                stats["drop_not_32"] += 1
                continue
            if not binary_alpha(arr):
                # snap almost-binary, else drop
                a = arr[..., 3]
                if ((a <= 2) | (a >= 253)).all():
                    arr = arr.copy()
                    arr[..., 3] = np.where(arr[..., 3] >= 128, 255, 0)
                    stats["snapped_alpha"] += 1
                else:
                    stats["drop_soft_alpha"] += 1
                    continue
            fr = magenta_fringe(arr)
            if fr["has_fringe"]:
                stats["drop_magenta_fringe"] += 1
                continue
            rec = {
                "idx": k,
                "split": split,
                "source": "ocdbytes/pixl-data",
                "licence": lic_norm,
                "licence_raw": lic_raw,
                "synthetic": False,
                "pack": pack,
                "creator": row.get("creator") or "",
                "source_url": url,
                "path": row.get("path") or "",
                "width_meta": w,
                "height_meta": h,
                "caption": row.get("label") or "",
                "label_source": row.get("label_source") or "",
                "md5": rgba_md5(arr),
                "dhash": dhash64(arr),
                "ahash": ahash64(arr),
            }
            mm[k] = arr
            k += 1
            kept_pack[pack] += 1
            stats["kept"] += 1
            meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta_f.close()
    mm.flush()
    del mm
    if k == 0:
        log("filter_pixl: kept 0 native 32x32 rows")
        # still write empty artefacts
        save_memmap_u8(img_p, np.zeros((0, 32, 32, 4), dtype=np.uint8))
        (out_dir / "meta.jsonl.partial").replace(meta_p)
    else:
        src = np.load(tmp, mmap_mode="r")
        save_memmap_u8(img_p, np.asarray(src[:k]))
        del src
        (out_dir / "meta.jsonl.partial").replace(meta_p)
    tmp.unlink(missing_ok=True)
    out_stats = {
        "rows_in": n_total,
        "kept": k,
        "counters": dict(stats),
        "by_size_in": dict(by_size.most_common()),
        "by_licence_in": dict(by_lic),
        "kept_by_pack": dict(kept_pack),
        "seconds": round(time.time() - t0, 1),
    }
    stats_p.write_text(json.dumps(out_stats, indent=2, ensure_ascii=False))
    log("filter_pixl done", json.dumps({"kept": k, "counters": dict(stats), "seconds": out_stats["seconds"]}))
    return out_stats


def hamming_dedup(records, imgs_list, threshold=6):
    """Greedy perceptual dedup. Prefer non-synthetic, then longer caption, then first source."""
    # records aligned with concatenated images
    buckets = defaultdict(list)
    for i, rec in enumerate(records):
        buckets[rec["dhash"] & ((1 << 16) - 1)].append(i)  # 16-bit bucket, compare within + nearby later
    # exact md5 first
    by_md5 = defaultdict(list)
    for i, rec in enumerate(records):
        by_md5[rec["md5"]].append(i)
    drop = set()
    exact_groups = 0
    for ids in by_md5.values():
        if len(ids) < 2:
            continue
        exact_groups += 1
        ids_sorted = sorted(ids, key=_keep_rank)
        for j in ids_sorted[1:]:
            drop.add(j)
    # perceptual among survivors
    survivors = [i for i in range(len(records)) if i not in drop]
    # sort by dhash so similar are near
    survivors.sort(key=lambda i: records[i]["dhash"])
    kept_idx = []
    # compare each to a window of previous kept with close dhash, plus ahash
    # For 180k this can be slow if naive O(n^2). Use 16-bit prefix buckets of full 64-bit dhash.
    kept_by_pref = defaultdict(list)

    def prefs(h):
        # 4 overlapping 16-bit slices
        return [(h >> s) & 0xFFFF for s in (0, 16, 32, 48)]

    n_phash = 0
    for i in survivors:
        h = records[i]["dhash"]
        ah = records[i]["ahash"]
        hit = None
        candidates = set()
        for pr in prefs(h):
            candidates.update(kept_by_pref[pr])
        for j in candidates:
            if hamming(h, records[j]["dhash"]) <= threshold and hamming(ah, records[j]["ahash"]) <= threshold + 2:
                hit = j
                break
        if hit is not None:
            n_phash += 1
            # keep the better of i and hit
            if _keep_rank(i) < _keep_rank(hit):
                # replace hit with i
                drop.add(hit)
                drop.discard(i)
                # leave hit in buckets (harmless); add i
                kept_idx.append(i)
                for pr in prefs(h):
                    kept_by_pref[pr].append(i)
            else:
                drop.add(i)
        else:
            kept_idx.append(i)
            for pr in prefs(h):
                kept_by_pref[pr].append(i)
    return drop, {"exact_dup_groups": exact_groups, "phash_dups": n_phash, "threshold": threshold}


def _keep_rank_key(rec):
    # lower is better
    synth = 1 if rec.get("synthetic") else 0
    cap = 0 if (rec.get("caption") or "").strip() else 1
    # prefer real, captioned, pixl/extra over azyg
    src_rank = 0 if rec.get("source", "").startswith("ocdbytes") or rec.get("source", "").startswith("kenney") or rec.get("source", "").startswith("opengameart") or rec.get("source", "") in ("kenney.nl", "OpenGameArt") else 1
    return (synth, src_rank, cap, -len(rec.get("caption") or ""))


# set later after records exist; use closure via function with recs global-ish
_RECS_FOR_RANK = []


def _keep_rank(i):
    return _keep_rank_key(_RECS_FOR_RANK[i])


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stage_dedup():
    ensure_dirs()
    out_p = OUT / "dedup_stats.json"
    # We'll write combined after holdout too; this stage only computes keep mask.
    parts = []
    for name in ("azyg", "pixl", "extra"):
        img_p = CLEAN / name / "images.npy"
        meta_p = CLEAN / name / "meta.jsonl"
        if not img_p.exists():
            continue
        imgs = np.load(img_p, mmap_mode="r")
        recs = load_jsonl(meta_p)
        assert len(recs) == len(imgs), (name, len(recs), len(imgs))
        for i, r in enumerate(recs):
            r = dict(r)
            r["_part"] = name
            r["_local_idx"] = i
            parts.append((r, name, i))
        log(f"dedup: loaded {name} n={len(recs)}")
    if not parts:
        raise SystemExit("dedup: nothing to merge")
    recs = [p[0] for p in parts]
    global _RECS_FOR_RANK
    _RECS_FOR_RANK = recs
    drop, dstats = hamming_dedup(recs, None, threshold=6)
    keep = [i for i in range(len(recs)) if i not in drop]
    # holdout filter
    blind, ev = load_holdouts()
    blind_re = holdout_regex(blind)
    eval_re = holdout_regex(ev)
    eval_set = set(ev)
    hold_drop = []
    final = []
    for i in keep:
        r = recs[i]
        cap = r.get("caption") or ""
        text = r.get("text") or ""
        reason = None
        caps_to_check = [cap, text, r.get("caption_alt") or ""]
        for s in caps_to_check:
            c = " ".join(s.lower().split())
            for pref in ("pixel art icon of a ", "pixel art icon of ", "pixel art of a ", "pixel art of ", "monochrome "):
                if c.startswith(pref):
                    c = c[len(pref):]
                    break
            if c in eval_set or s.lower().strip() in eval_set:
                reason = "eval50_exact"
                break
            if blind_re.search(s) or blind_re.search(c):
                reason = "blind50_phrase"
                break
            if eval_re.search(s) or eval_re.search(c):
                reason = "eval50_phrase"
                break
        if reason:
            hold_drop.append({"reason": reason, "caption": cap, "text": text, "source": r.get("source")})
        else:
            final.append(i)
    log(f"dedup: in={len(recs)} exact+phash_drop={len(drop)} holdout_drop={len(hold_drop)} keep={len(final)}")
    # write combined images + meta
    if final:
        arr = np.empty((len(final), 32, 32, 4), dtype=np.uint8)
        out_recs = []
        # load parts into memmaps
        mms = {}
        for name in ("azyg", "pixl", "extra"):
            p = CLEAN / name / "images.npy"
            if p.exists():
                mms[name] = np.load(p, mmap_mode="r")
        for j, i in enumerate(final):
            r = recs[i]
            arr[j] = mms[r["_part"]][r["_local_idx"]]
            rr = {k: v for k, v in r.items() if not k.startswith("_")}
            rr["idx"] = j
            out_recs.append(rr)
        save_memmap_u8(OUT / "images.npy", arr)
        with (OUT / "meta_pre_recaption.jsonl").open("w", encoding="utf-8") as f:
            for r in out_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        save_memmap_u8(OUT / "images.npy", np.zeros((0, 32, 32, 4), np.uint8))
        (OUT / "meta_pre_recaption.jsonl").write_text("")
    stats = {
        "in": len(recs),
        "dropped_dup": len(drop),
        "dropped_holdout": len(hold_drop),
        "kept": len(final),
        "dedup": dstats,
        "holdout_examples": hold_drop[:30],
        "holdout_reasons": dict(Counter(x["reason"] for x in hold_drop)),
        "kept_by_source": dict(Counter(recs[i].get("source") for i in final)),
        "kept_synthetic": int(sum(1 for i in final if recs[i].get("synthetic"))),
        "kept_real": int(sum(1 for i in final if not recs[i].get("synthetic"))),
    }
    out_p.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    log("dedup done", json.dumps({k: stats[k] for k in ("in", "dropped_dup", "dropped_holdout", "kept", "kept_by_source")}))
    return stats


def load_gemini_key() -> str | None:
    """Return the key or None. Never prints it."""
    env = os.environ.get("GEMINI_API_KEY")
    if env:
        return env.strip()
    for p in (
        Path(os.path.expandvars("$PIXIE_ROOT/secrets/GEMINI_API_KEY")),
        Path(os.path.expandvars("$PIXIE_ROOT/secrets/gemini_api_key")),
        ROOT / "secrets" / "GEMINI_API_KEY",
    ):
        if p.is_file():
            return p.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    return None


def stage_recaption():
    """Recaption real (non-synthetic) sprites only. Uses Gemini if a key is present; otherwise
    keeps existing labels and writes a note so we can re-run later."""
    ensure_dirs()
    meta_p = OUT / "meta_pre_recaption.jsonl"
    recs = load_jsonl(meta_p)
    real_idx = [i for i, r in enumerate(recs) if not r.get("synthetic")]
    key = load_gemini_key()
    stats = {"real": len(real_idx), "method": None, "updated": 0, "failed": 0}
    out_meta = OUT / "meta.jsonl"
    if not real_idx:
        log("recaption: no real sprites")
        meta_p.replace(out_meta) if False else (out_meta.write_text(meta_p.read_text(encoding="utf-8")))
        stats["method"] = "none"
        (OUT / "recaption_stats.json").write_text(json.dumps(stats, indent=2))
        return stats
    if not key:
        log("recaption: no GEMINI_API_KEY found; keeping existing labels (claude-visual/filename/atlas)")
        for r in recs:
            if not r.get("synthetic"):
                r["caption_original"] = r.get("caption") or ""
                r["caption_source"] = r.get("label_source") or "existing"
        with out_meta.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats["method"] = "kept_existing_no_key"
        (OUT / "recaption_stats.json").write_text(json.dumps(stats, indent=2))
        return stats
    # Gemini recaption — short sprite captions, batch if possible
    import base64, time as _t, urllib.request
    imgs = np.load(OUT / "images.npy", mmap_mode="r")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    prompt = (
        "You caption a 32x32 pixel-art game sprite with a transparent background. "
        "Reply with one short lowercase English caption only, 3 to 8 words, no quotes, "
        "no 'pixel art' prefix, naming the object and its main colour or material. "
        "Examples: red health potion, iron battle axe, green slime, oak tree, gold coin."
    )
    # skip holdout leakage: we already dropped holdout captions; still check new ones later in manifest
    for n, i in enumerate(real_idx):
        arr = imgs[i]
        im = Image.fromarray(arr, "RGBA")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        body = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": b64}},
                ]
            }],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 40},
        }
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip().strip('"').lower()
            text = " ".join(text.split())
            recs[i]["caption_original"] = recs[i].get("caption") or ""
            recs[i]["caption"] = text
            recs[i]["caption_source"] = "gemini"
            recs[i]["caption_model"] = model
            stats["updated"] += 1
        except Exception as e:
            recs[i]["caption_original"] = recs[i].get("caption") or ""
            recs[i]["caption_source"] = recs[i].get("label_source") or "existing"
            recs[i]["recaption_error"] = type(e).__name__
            stats["failed"] += 1
        if (n + 1) % 25 == 0:
            log(f"  recaption {n+1}/{len(real_idx)} ok={stats['updated']} fail={stats['failed']}")
            # light rate limit
            _t.sleep(0.2)
        else:
            _t.sleep(0.05)
    with out_meta.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["method"] = "gemini"
    (OUT / "recaption_stats.json").write_text(json.dumps(stats, indent=2))
    log("recaption done", stats)
    return stats


def stage_manifest():
    ensure_dirs()
    recs = load_jsonl(OUT / "meta.jsonl" if (OUT / "meta.jsonl").exists() else OUT / "meta_pre_recaption.jsonl")
    # re-check holdout after recaption
    blind, ev = load_holdouts()
    blind_re = holdout_regex(blind)
    eval_re = holdout_regex(ev)
    eval_set = set(ev)
    keep = []
    dropped = []
    for r in recs:
        cap = r.get("caption") or ""
        text = r.get("text") or ""
        reason = None
        for s in (cap, text, r.get("caption_original") or ""):
            c = " ".join(s.lower().split())
            for pref in ("pixel art icon of a ", "pixel art icon of ", "pixel art of a ", "pixel art of ", "monochrome "):
                if c.startswith(pref):
                    c = c[len(pref):]
                    break
            if c in eval_set or s.lower().strip() in eval_set:
                reason = "eval50_exact"; break
            if blind_re.search(s) or blind_re.search(c):
                reason = "blind50_phrase"; break
            if eval_re.search(s) or eval_re.search(c):
                reason = "eval50_phrase"; break
        if reason:
            dropped.append((reason, cap))
        else:
            keep.append(r)
    if dropped:
        log(f"manifest: dropping {len(dropped)} holdout after recaption")
        # rewrite images if any dropped
        imgs = np.load(OUT / "images.npy", mmap_mode="r")
        idxs = [r["idx"] for r in keep]
        save_memmap_u8(OUT / "images.npy", np.asarray(imgs[idxs]))
        for j, r in enumerate(keep):
            r["idx"] = j
        with (OUT / "meta.jsonl").open("w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        recs = keep

    fields = ["path", "source", "licence", "caption", "synthetic", "split", "backend", "pack",
              "creator", "source_url", "caption_source", "category", "md5"]
    man_p = OUT / "manifest.csv"
    with man_p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            row = dict(r)
            # stable relative path into the packed images.npy
            row["path"] = f"out/images.npy:{r['idx']:06d}"
            row["synthetic"] = "1" if r.get("synthetic") else "0"
            w.writerow(row)

    # ATTRIBUTION.md
    by_lic = Counter((r.get("licence") or "unknown") for r in recs)
    real = [r for r in recs if not r.get("synthetic")]
    # group by pack/creator/url
    groups = defaultdict(list)
    for r in real:
        key = (r.get("licence"), r.get("pack") or r.get("source"), r.get("creator") or "", r.get("source_url") or "")
        groups[key].append(r)
    lines = [
        "# Attribution for Pixie-32 training data",
        "",
        "This file lists third-party art used in the 32×32 training set.",
        "AI-generated icons from `azyg0s/pixelart-172k` are MIT-licensed (synthetic) and listed only as a count.",
        "",
        f"- Synthetic MIT (azyg0s/pixelart-172k): {sum(1 for r in recs if r.get('synthetic'))} images",
        f"- Real sprites: {len(real)} images",
        "",
        "## Licence counts",
        "",
    ]
    for lic, n in by_lic.most_common():
        lines.append(f"- {lic}: {n}")
    lines += ["", "## Real packs (please credit CC BY 3.0 authors)", ""]
    for (lic, pack, creator, url), rs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0][1] or "")):
        lines.append(f"- **{pack}** — {lic} — {len(rs)} images — creator: {creator or '—'} — {url or '—'}")
    lines += [
        "",
        "## Notes",
        "",
        "- game-icons.net icons are CC BY 3.0; credit the listed creator (e.g. Lorc, Delapouite).",
        "- Kenney packs are CC0 1.0 (https://kenney.nl).",
        "- OpenGameArt packs were included only when the pack page listed CC0; each row records its source URL.",
        "- The pixl-data `scenes` config was downloaded with the snapshot but is not used in this set.",
    ]
    (OUT / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # stats
    by_src = Counter(r.get("source") for r in recs)
    by_split = Counter(r.get("split") for r in recs)
    stats = {
        "n": len(recs),
        "synthetic": sum(1 for r in recs if r.get("synthetic")),
        "real": sum(1 for r in recs if not r.get("synthetic")),
        "by_source": dict(by_src),
        "by_licence": dict(by_lic),
        "by_split": dict(by_split),
        "holdout_dropped_at_manifest": len(dropped),
        "disk": _disk_report(),
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    log("manifest written", stats["n"], "real", stats["real"], "synthetic", stats["synthetic"])
    return stats


def _disk_report():
    def du(p: Path) -> int:
        if not p.exists():
            return 0
        if p.is_file():
            return p.stat().st_size
        tot = 0
        for root, dirs, files in os.walk(p):
            for fn in files:
                try:
                    tot += (Path(root) / fn).stat().st_size
                except OSError:
                    pass
        return tot
    import shutil
    usage = shutil.disk_usage("/")
    return {
        "data32_bytes": du(ROOT),
        "raw_bytes": du(RAW),
        "clean_bytes": du(CLEAN),
        "out_bytes": du(OUT),
        "disk_free_bytes": usage.free,
        "disk_used_bytes": usage.used,
        "disk_total_bytes": usage.total,
    }


def stage_sheet():
    ensure_dirs()
    imgs = np.load(OUT / "images.npy", mmap_mode="r")
    recs = load_jsonl(OUT / "meta.jsonl" if (OUT / "meta.jsonl").exists() else OUT / "meta_pre_recaption.jsonl")
    n = min(64, len(imgs))
    if n == 0:
        log("sheet: no images")
        return
    # stratified: 32 synthetic + 32 real if possible, else random-but-deterministic
    rng = np.random.default_rng(32)
    real = [i for i, r in enumerate(recs) if not r.get("synthetic")]
    synth = [i for i, r in enumerate(recs) if r.get("synthetic")]
    pick = []
    if real and synth:
        k_r = min(32, len(real), n)
        k_s = min(n - k_r, len(synth))
        pick = list(rng.choice(real, size=k_r, replace=False)) + list(rng.choice(synth, size=k_s, replace=False))
    else:
        pick = list(rng.choice(len(imgs), size=n, replace=False))
    pick = pick[:64]
    while len(pick) < 64 and len(imgs) >= 64:
        # pad shouldn't happen
        break
    cols, rows = 8, 8
    scale = 4
    pad = 2
    cell = 32 * scale + pad
    canvas = Image.new("RGBA", (cols * cell + pad, rows * cell + pad), (24, 24, 28, 255))
    # checkerboard for transparency
    def paste(im, x, y):
        bg = Image.new("RGBA", (32 * scale, 32 * scale), (0, 0, 0, 0))
        for yy in range(0, 32 * scale, 8):
            for xx in range(0, 32 * scale, 8):
                c = (70, 70, 76, 255) if ((xx // 8) + (yy // 8)) % 2 == 0 else (40, 40, 46, 255)
                ImageDraw.Draw(bg).rectangle([xx, yy, xx + 7, yy + 7], fill=c)
        spr = im.resize((32 * scale, 32 * scale), Image.Resampling.NEAREST)
        bg.alpha_composite(spr)
        canvas.paste(bg, (x, y))
    for i, idx in enumerate(pick):
        r, c = divmod(i, cols)
        im = Image.fromarray(imgs[idx], "RGBA")
        paste(im, pad + c * cell, pad + r * cell)
    dest = OUT / "data32-sample.png"
    canvas.convert("RGB").save(dest, "PNG")
    # also store pick captions
    (OUT / "data32-sample.json").write_text(json.dumps([
        {"idx": int(i), "caption": recs[i].get("caption"), "source": recs[i].get("source"), "synthetic": recs[i].get("synthetic")}
        for i in pick
    ], indent=2, ensure_ascii=False))
    log("sheet written", dest, "n", len(pick))
    return str(dest)


def estimate_cache_and_maybe_build():
    """Build palette-index token cache if it fits. Report CLIP-feature cache size, do not
    copy the 70G v3 clipcache layout unless it clearly fits and can be done on CPU."""
    recs = load_jsonl(OUT / "meta.jsonl" if (OUT / "meta.jsonl").exists() else OUT / "meta_pre_recaption.jsonl")
    n = len(recs)
    # palette indices uint16 [N,1024] = 2*N*1024
    token_bytes = n * 1024 * 2
    # captions jsonl ~ 80-150 B each
    cap_bytes = n * 200
    # CLIP-L text teacher like v3 clipcache: feats_t18_f16.bin is 70G for ~1.36M rows
    # => ~51.5 KiB/row. For N rows:
    clip_est = int(n * (70 * (1024 ** 3) / 1_356_348))
    pal_path = Path(os.path.expandvars("$PIXIE_ROOT/data/subset150k/palette.npy"))
    report = {
        "n": n,
        "token_cache_bytes": token_bytes,
        "caption_bytes_est": cap_bytes,
        "clip_l14_text_t18_est_bytes": clip_est,
        "will_build_token_cache": True,
        "will_build_clip_cache": False,
        "clip_skip_reason": "Estimated from v3 clipcache_s7_t18 (70G / 1.36M rows). Needs CLIP-L forward passes; skipped to keep the GPU free for chat. Size reported only.",
    }
    free = __import__("shutil").disk_usage("/").free
    # keep a 15G safety margin under the 30G new-usage cap; token cache is small
    if token_bytes + cap_bytes + 50_000_000 > free - 8 * (1024 ** 3):
        report["will_build_token_cache"] = False
        report["token_skip_reason"] = "not enough free disk with safety margin"
        (OUT / "cache_estimate.json").write_text(json.dumps(report, indent=2))
        log("cache: skip build", report)
        return report
    if not pal_path.exists():
        report["will_build_token_cache"] = False
        report["token_skip_reason"] = "palette.npy not found"
        (OUT / "cache_estimate.json").write_text(json.dumps(report, indent=2))
        return report
    pal = np.load(pal_path, allow_pickle=False)  # (4097,3) uint8
    # index 0 is transparent (black in the table); we use 0 for alpha=0, 1..4096 for colours
    color_to_id = {}
    # palette[0] is (0,0,0) used as transparent sentinel in v3? prep said 0=transparent, max < 4097
    for i, rgb in enumerate(pal):
        color_to_id[tuple(int(x) for x in rgb)] = i
    imgs = np.load(OUT / "images.npy", mmap_mode="r")
    tok = np.lib.format.open_memmap(OUT / "cache" / "img_idx.tmp.npy", mode="w+", dtype=np.uint16, shape=(n, 1024))
    misses = 0
    miss_pixels = 0
    total_opaque = 0
    # nearest-color fallback counts
    pal_rgb = pal.astype(np.int16)
    for i in range(n):
        a = imgs[i]
        out = np.empty((1024,), np.uint16)
        px = a.reshape(-1, 4)
        for j, (r, g, b, al) in enumerate(px):
            if al < 128:
                out[j] = 0
                continue
            total_opaque += 1
            key = (int(r), int(g), int(b))
            pid = color_to_id.get(key)
            if pid is None or pid == 0:
                # nearest
                d = pal_rgb - np.array(key, dtype=np.int16)
                pid = int(np.argmin(np.sum(d * d, axis=1)))
                if pid == 0:
                    # don't map opaque to transparent; use next-best if possible
                    ss = np.sum(d * d, axis=1)
                    ss[0] = 1 << 30
                    pid = int(np.argmin(ss))
                misses += 1
                miss_pixels += 1
            out[j] = pid
        tok[i] = out
        if (i + 1) % 10000 == 0:
            log(f"  cache {i+1}/{n} palette_miss_sprites={misses}")
    tok.flush()
    del tok
    (OUT / "cache" / "img_idx.tmp.npy").replace(OUT / "cache" / "img_idx.npy")
    report["palette_sprites_with_any_unmapped"] = misses
    report["palette_unmapped_opaque_pixels"] = miss_pixels
    report["opaque_pixels"] = total_opaque
    report["token_cache_path"] = str(OUT / "cache" / "img_idx.npy")
    (OUT / "cache_estimate.json").write_text(json.dumps(report, indent=2))
    (OUT / "cache" / "cache_manifest.json").write_text(json.dumps(report, indent=2))
    log("cache built", report["token_cache_path"], "unmapped_px", miss_pixels)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    help="filter_azyg,filter_pixl,dedup,recaption,manifest,cache,sheet,all,filter")
    args = ap.parse_args()
    stages = args.stage
    if stages == "all":
        seq = ["filter_azyg", "filter_pixl", "dedup", "recaption", "manifest", "sheet", "cache"]
    elif stages == "filter":
        seq = ["filter_azyg", "filter_pixl"]
    else:
        seq = [s.strip() for s in stages.split(",") if s.strip()]
    log("ROOT", ROOT, "stages", seq)
    fns = {
        "filter_azyg": stage_filter_azyg,
        "filter_pixl": stage_filter_pixl,
        "dedup": stage_dedup,
        "recaption": stage_recaption,
        "manifest": stage_manifest,
        "sheet": stage_sheet,
        "cache": estimate_cache_and_maybe_build,
    }
    t0 = time.time()
    for s in seq:
        log("=" * 60, s)
        fns[s]()
    log("ALL DONE in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
