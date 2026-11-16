#!/usr/bin/env python3
"""Download Kenney CC0 32x32 PNGs (not already in pixl Kenney packs). CPU/network only."""
from __future__ import annotations
import os  # env placeholders ($DATA_ROOT etc.)
import hashlib, json, time, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(os.path.expandvars("$PIXIE_ROOT")) / "data32"
OUT = ROOT / "clean" / "extra"
OUT.mkdir(parents=True, exist_ok=True)
IDX = Path(__file__).resolve().parent / "kenney-index.tsv"
# Official pack pages. Images come from the community CC0 mirror of Kenney's packs.
MIRROR = "https://raw.githubusercontent.com/lateralus426/kenney_assets-classified/main/"
# Packs already inside ocdbytes/pixl-data
ALREADY = {
    "1-Bit Pack", "1-Bit Platformer Pack", "Board Game Icons", "Cursor Pixel Pack",
    "Game Icons", "Monochrome Pirates", "Pico-8 City", "Pixel Shmup", "Pixel Vehicle Pack",
    "Tiny Battle", "Tiny Dungeon", "Tiny Farm", "Tiny Town",
}
UA = "PixieDataPrep/1.0 (CC0 dataset prep)"

def slug(pack: str) -> str:
    return pack.lower().replace(" ", "-").replace("_", "-")

def main():
    rows = []
    for line in IDX.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        path, desc, size, nbytes = parts[0], parts[1], parts[2], parts[3]
        if size != "32x32":
            continue
        segs = path.split("/")
        pack = segs[1] if len(segs) > 1 else segs[0]
        if pack in ALREADY:
            continue
        rows.append((path, pack, desc, int(nbytes) if nbytes.isdigit() else 0))
    print(f"kenney 32x32 not already in pixl: {len(rows)}", flush=True)
    raw_dir = ROOT / "raw" / "kenney32"
    raw_dir.mkdir(parents=True, exist_ok=True)
    kept = []
    imgs = []
    fails = 0
    for i, (path, pack, desc, nbytes) in enumerate(rows):
        dest = raw_dir / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size == 0:
            url = MIRROR + path
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    dest.write_bytes(r.read())
                time.sleep(0.05)
            except Exception as e:
                fails += 1
                print("FAIL", path, type(e).__name__, flush=True)
                continue
        try:
            im = Image.open(dest).convert("RGBA")
        except Exception as e:
            fails += 1
            print("OPEN", path, e)
            continue
        if im.size != (32, 32):
            continue
        arr = np.asarray(im, dtype=np.uint8)
        a = arr[..., 3]
        if not set(np.unique(a)).issubset({0, 255}):
            if ((a <= 2) | (a >= 253)).all():
                arr = arr.copy(); arr[..., 3] = np.where(arr[..., 3] >= 128, 255, 0)
            else:
                continue
        cap = desc.split(",")[-1].strip() if desc else dest.stem
        cap = cap.replace("Tile ", "tile ").strip().lower() or dest.stem.lower()
        rec = {
            "source": "kenney.nl",
            "licence": "CC0 1.0",
            "synthetic": False,
            "pack": pack,
            "creator": "Kenney",
            "source_url": f"https://kenney.nl/assets/{slug(pack)}",
            "path": path,
            "caption": cap,
            "label_source": "kenney-index",
            "split": "train",
            "md5": hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest(),
        }
        kept.append(rec); imgs.append(arr)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)} kept={len(kept)} fail={fails}", flush=True)
    if not imgs:
        print("no kenney images kept")
        return
    arr = np.stack(imgs, 0)
    # write under clean/extra as a partial named source file; merge script will concat
    np.save(OUT / "kenney_images.npy", arr)
    with (OUT / "kenney_meta.jsonl").open("w", encoding="utf-8") as f:
        for j, r in enumerate(kept):
            r["idx"] = j
            # dhash later in merge
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "kenney_stats.json").write_text(json.dumps({
        "rows_listed": len(rows), "kept": len(kept), "fails": fails,
        "by_pack": {p: sum(1 for r in kept if r["pack"]==p) for p in sorted({r["pack"] for r in kept})},
    }, indent=2))
    print("kenney done", len(kept), "fails", fails)

if __name__ == "__main__":
    main()
