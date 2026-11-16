#!/usr/bin/env python3
"""Download OpenGameArt packs that offer CC0, extract native 32x32 sprites / 32-grid slices.
Verifies CC0 from the pack page record. CPU/network only. Polite delays.
"""
from __future__ import annotations
import hashlib, io, json, os, re, shutil, time, urllib.request, zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(os.path.expandvars("$PIXIE_ROOT")) / "data32"
RAW = ROOT / "raw" / "oga32"
OUT = ROOT / "clean" / "extra"
HOLD = Path(__file__).resolve().parent
CATALOG = HOLD / "oga-catalog" / "packs.json"
UA = "PixieDataPrep/1.0 (CC0 dataset prep)"
MAX_FILE = 25 * 1024 * 1024
MAX_TOTAL = 2 * 1024 * 1024 * 1024
# Already represented in ocdbytes/pixl-data
SKIP_SLUG = {
    "/content/dungeon-crawl-32x32-tiles",
    "/content/dungeon-crawl-32x32-tiles-supplemental",
    "/content/idylwilds-inventory",
    "/content/spaceship-set-32x32px",
    "/content/a-platformer-in-the-forest",
    "/content/sunny-land",
    "/content/sunny-land-pixel-game-art",
    "/content/roguelikerpg-pack-1700-tiles",  # Kenney, already in pixl as Tiny/roguelike family? keep if not sure
}
# Kenney OGA mirrors we already cover via kenney fetch / pixl
SKIP_TITLE_RE = re.compile(r"\bkenney\b", re.I)
SKIP_SLUG_EXTRA = re.compile(r"kenney|1-bit-pack|tiny-dungeon|tiny-town|tiny-farm|tiny-battle", re.I)

def fetch(url, dest: Path | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = r.read()
    if dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return data

def key_magenta(arr: np.ndarray) -> np.ndarray:
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    mag = (r >= 240) & (g <= 15) & (b >= 240) & (a >= 128)
    if mag.mean() > 0.04:
        arr = arr.copy()
        arr[mag, 3] = 0
    return arr

def binary_or_snap(arr: np.ndarray) -> np.ndarray | None:
    a = arr[..., 3]
    if np.isin(a, (0, 255)).all():
        return arr
    if ((a <= 2) | (a >= 253)).all():
        arr = arr.copy()
        arr[..., 3] = np.where(arr[..., 3] >= 128, 255, 0)
        return arr
    return None

def useful_cell(arr: np.ndarray) -> bool:
    a = arr[..., 3]
    return int((a >= 128).sum()) >= 8

def pack_slice_mode(title: str) -> str:
    """Only slice large sheets when the pack is actually 32x32. 16x16 packs
    must not be cut into 32x32 (that glues four tiles together). Other packs
    contribute only already-sized 32x32 PNGs."""
    t = (title or "").lower()
    has32 = bool(re.search(r"32\s*[x×]\s*32", t))
    has16 = bool(re.search(r"16\s*[x×]\s*16", t))
    has8 = bool(re.search(r"(?<!\d)8\s*[x×]\s*8", t))
    if (has16 or has8) and not has32:
        return "exact_only"
    if has32:
        return "slice32"
    return "exact_only"

def cells_from_image(im: Image.Image, mode: str = "exact_only"):
    im = im.convert("RGBA")
    arr = key_magenta(np.asarray(im, dtype=np.uint8))
    h, w = arr.shape[:2]
    out = []
    if (w, h) == (32, 32):
        snapped = binary_or_snap(arr)
        if snapped is not None and useful_cell(snapped):
            out.append(snapped)
        return out
    if mode != "slice32":
        return out
    if w >= 32 and h >= 32 and w % 32 == 0 and h % 32 == 0:
        if (w // 32) * (h // 32) > 2048:
            return out
        for y in range(0, h, 32):
            for x in range(0, w, 32):
                cell = arr[y:y+32, x:x+32]
                snapped = binary_or_snap(cell)
                if snapped is not None and useful_cell(snapped):
                    out.append(snapped)
    return out

def iter_images_from_bytes(name: str, data: bytes):
    low = name.lower()
    if low.endswith((".png", ".gif", ".bmp", ".tga", ".webp")):
        try:
            im = Image.open(io.BytesIO(data))
            yield name, im
        except Exception:
            return
    elif low.endswith(".zip"):
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except Exception:
            return
        for info in zf.infolist():
            if info.is_dir() or info.file_size > MAX_FILE:
                continue
            n = info.filename
            if n.lower().endswith((".png", ".gif", ".bmp", ".tga", ".webp")):
                try:
                    yield n, Image.open(io.BytesIO(zf.read(info)))
                except Exception:
                    continue
    # 7z/rar skipped unless tools exist (handled by caller unpack)

def caption_from(path: str, pack_title: str) -> str:
    stem = Path(path).stem
    stem = re.sub(r"[_-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip().lower()
    stem = re.sub(r"^(tile|sprite|img|image|icon)\s*\d*\s*", "", stem).strip() or Path(path).stem.lower()
    return stem[:80]


def already(p: dict) -> bool:
    slug = (p.get("slug") or "").lower()
    url = (p.get("url") or "").lower()
    title = (p.get("title") or p.get("title_listing") or "")
    if slug in SKIP_SLUG or any(s.lstrip("/") in slug for s in SKIP_SLUG):
        return True
    if SKIP_TITLE_RE.search(title) or SKIP_SLUG_EXTRA.search(slug):
        return True
    # dungeon crawl already in pixl
    if "dungeon-crawl-32x32" in slug or "dungeon crawl 32x32" in title.lower():
        return True
    return False


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    packs = json.loads(CATALOG.read_text(encoding="utf-8"))
    cc0 = [p for p in packs if p.get("has_cc0") and p.get("files") and not already(p) and not p.get("sa_only")]
    # prefer titles that actually mention 32x32 or tileset/sprites
    def score(p):
        t = (p.get("title") or "") + " " + (p.get("title_listing") or "")
        s = 0
        if re.search(r"32\s*[x×]\s*32", t, re.I):
            s += 10
        if re.search(r"tileset|sprites?|character|icon|item|monster|dungeon", t, re.I):
            s += 3
        s += min(len(p.get("files") or []), 5)
        return -s
    cc0.sort(key=score)
    print(f"oga cc0 candidate packs {len(cc0)} / {len(packs)}", flush=True)
    kept, imgs = [], []
    stats = Counter()
    downloaded = 0
    for i, p in enumerate(cc0):
        if downloaded >= MAX_TOTAL:
            print("hit total download cap", flush=True)
            break
        title = p.get("title") or p.get("title_listing") or "unknown"
        author = p.get("author") or ""
        url = p.get("url") or ""
        pack_id = re.sub(r"[^a-z0-9_-]+", "_", (p.get("slug") or title).strip("/").replace("content/", ""))[:80]
        pack_dir = RAW / pack_id
        pack_dir.mkdir(exist_ok=True)
        (pack_dir / "SOURCE.json").write_text(json.dumps({
            "url": url, "title": title, "author": author, "licenses": p.get("licenses"),
        }, indent=2, ensure_ascii=False))
        n_before = len(kept)
        for f in p["files"]:
            furl, fname = f["url"], f.get("name") or f["url"].split("/")[-1]
            fname = re.sub(r"[^A-Za-z0-9._-]+", "_", fname)
            dest = pack_dir / fname
            if dest.exists() and dest.stat().st_size > 0:
                data = dest.read_bytes()
            else:
                # HEAD-less: try, skip if huge via reading with limit? urlopen content-length
                try:
                    time.sleep(0.6)
                    req = urllib.request.Request(furl, headers={"User-Agent": UA})
                    with urllib.request.urlopen(req, timeout=90) as r:
                        cl = r.headers.get("Content-Length")
                        if cl and int(cl) > MAX_FILE:
                            stats["skip_large"] += 1
                            continue
                        data = r.read(MAX_FILE + 1)
                    if len(data) > MAX_FILE:
                        stats["skip_large"] += 1
                        continue
                    dest.write_bytes(data)
                    downloaded += len(data)
                except Exception as e:
                    stats["download_fail"] += 1
                    print("  DL fail", fname, type(e).__name__, flush=True)
                    continue
            ext = Path(fname).suffix.lower()
            if ext == ".7z":
                stats["skip_7z"] += 1
                continue
            for inner_name, im in iter_images_from_bytes(fname, data):
                try:
                    cells = cells_from_image(im, pack_slice_mode(title))
                except Exception:
                    stats["decode_fail"] += 1
                    continue
                for ci, arr in enumerate(cells):
                    rec = {
                        "source": "OpenGameArt",
                        "licence": "CC0 1.0",
                        "synthetic": False,
                        "pack": title,
                        "creator": author,
                        "source_url": url,
                        "path": f"{inner_name}" + (f"#c{ci}" if len(cells) > 1 else ""),
                        "caption": caption_from(inner_name if len(cells) == 1 else f"{Path(inner_name).stem}_{ci}", title),
                        "label_source": "filename",
                        "split": "train",
                        "md5": hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest(),
                    }
                    kept.append(rec)
                    imgs.append(arr)
                    stats["cells"] += 1
        stats["packs_tried"] += 1
        got = len(kept) - n_before
        if got:
            stats["packs_with_32"] += 1
        if (i + 1) % 10 == 0 or got:
            print(f"  {i+1}/{len(cc0)} +{got} total_cells={len(kept)} downloaded_mb={downloaded/1e6:.1f} {title[:50]}", flush=True)
    if imgs:
        np.save(OUT / "oga_images.npy", np.stack(imgs, 0))
        with (OUT / "oga_meta.jsonl").open("w", encoding="utf-8") as f:
            for j, r in enumerate(kept):
                r["idx"] = j
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "oga_stats.json").write_text(json.dumps({"kept": len(kept), "counters": dict(stats), "downloaded_bytes": downloaded}, indent=2))
    print("oga done kept", len(kept), dict(stats), flush=True)

if __name__ == "__main__":
    main()
