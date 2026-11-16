#!/usr/bin/env python3
"""Combine Krea 2 Turbo sprite runs into one 32x32 dataset (new folder only).

Re-extracts every raw sheet with extract_v2 (unchanged gates), writes one
manifest row per cell, removes Blind-50/Eval-50 holdout prompts, drops exact
pixel duplicates, and builds a train-ready folder. Reads sources read-only.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, re, sys, time
from collections import defaultdict, Counter
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_v2 as V
V.pitch_of = None  # art_px is informational only; skipped for speed

# Exact-result speedup: the grown key on the full sheet is the same for every
# cell of a sheet, so memoize it per image object (spill() recomputes it 4/9x).
_orig_grown = V.is_bg_grown
_SHEET = {"img": None, "mask": None}
def _grown(rgb, seed_tol=V.KEY_TOL, grow_tol=V.GROW_TOL, rounds=V.GROW_ROUNDS):
    if rgb is _SHEET["img"] and (seed_tol, grow_tol, rounds) == (V.KEY_TOL, V.GROW_TOL, V.GROW_ROUNDS):
        if _SHEET["mask"] is None:
            _SHEET["mask"] = _orig_grown(rgb, seed_tol, grow_tol, rounds)
        return _SHEET["mask"]
    return _orig_grown(rgb, seed_tol, grow_tol, rounds)

ROOT = Path(os.path.expandvars("$DATA_ROOT"))
SOURCES = [
    # name, gpu, grid, expected sheet px, data dir (read-only), lora
    ("3080ti", "RTX 3080 Ti", 3, 576, Path(os.path.expandvars("$GEN_ROOT/krea2-gen-3080ti/data")), "p3-1500"),
    ("3060ti", "RTX 3060 Ti", 3, 576, ROOT / "raw/3060ti", "p3-1500"),
    ("5070ti", "RTX 5070 Ti", 2, 512, ROOT / "raw/5070ti", "pix32-final"),
]

def norm(s):
    s = s.lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def canon_hash(rgb, alpha):
    a = alpha.copy()
    rgb = np.where(a[..., None] > 0, rgb, 0).astype(np.uint8)
    return hashlib.sha1(np.dstack([rgb, a]).tobytes()).hexdigest()

def work(task):
    src, grid, exp_px, sheet_path, cells, out_dir = task
    V.is_bg_grown = _grown
    img = np.asarray(Image.open(sheet_path).convert("RGB"))
    _SHEET["img"], _SHEET["mask"] = img, None
    H, W = img.shape[:2]
    rows = []
    for c in cells:
        k = int(c["cell"])
        r, cc = divmod(k, grid)
        ex = V.extract_cell(img, grid, r, cc, px=32)
        s, pm = ex["score"], ex["purity"]
        b = V.bin_for(s, pm)
        sid = int(c["sheet_id"])
        sp_id = f"{src}-s{sid:06d}-c{k}"
        rel = f"extract/{src}/{b}/{sid // 1000:03d}/{sp_id}.png"
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.dstack([ex["rgb"], ex["alpha"]]), "RGBA").save(dst)
        rows.append({
            "id": sp_id, "source": src, "grid": f"{grid}x{grid}", "sheet_px": [W, H],
            "sheet": str(sheet_path), "sheet_id": sid, "cell": k,
            "caption": c.get("phrase"), "subject": c.get("subject"), "cat": c.get("cat"),
            "subject_id": c.get("sid"), "seed": c.get("seed"),
            "fg": s["fg"], "border": s["border"], "main_comp": s["main_comp"], "n_comp": s["n_comp"],
            "spill": s["spill"], "purity": round(pm, 3), "edge_touch": ex["edge_touch"],
            "fringe_dropped": ex["fringe_dropped"], "bin": b, "pass_extract_v2": b == "ok",
            "reject_reasons": V.reject_reasons(s, pm), "sprite_path": rel,
            "pixel_hash": canon_hash(ex["rgb"], ex["alpha"]),
            "orig_file": c.get("file"), "orig_usable": c.get("usable"),
            "extractor": V.VERSION,
        })
    return rows

def load_tasks(limit=None, only=None):
    tasks, meta = [], {}
    for name, gpu, grid, px, d, lora in SOURCES:
        if only and name not in only:
            continue
        by = defaultdict(list)
        for line in open(d / "labels.jsonl"):
            if line.strip():
                j = json.loads(line); by[int(j["sheet_id"])].append(j)
        sheets_on_disk = {p.name: p for p in (d / "sheets").rglob("sheet-*.png")}
        missing, badcount, n = [], [], 0
        for sid in sorted(by):
            cells = sorted(by[sid], key=lambda c: int(c["cell"]))
            if len(cells) != grid * grid:
                badcount.append(sid)
            rel = cells[0].get("sheet") or f"{sid // 1000:03d}/sheet-{sid:06d}.png"
            p = d / "sheets" / rel
            if not p.is_file():
                p = sheets_on_disk.get(Path(rel).name)
            if p is None or not p.is_file():
                missing.append(sid); continue
            tasks.append((name, grid, px, p, cells, Path(os.path.expandvars("$DATA_ROOT/test")))); n += 1
            if limit and n >= limit:
                break
        labelled = {Path(by[s][0].get("sheet") or "").name for s in by}
        meta[name] = {"gpu": gpu, "grid": f"{grid}x{grid}", "sheet_px": px, "lora": lora, "data": str(d),
                      "sheets_on_disk": len(sheets_on_disk), "sheets_labelled": len(by),
                      "label_rows": sum(len(v) for v in by.values()),
                      "sheets_missing_png": missing, "sheets_wrong_cellcount": badcount,
                      "unlabelled_sheet_pngs": sorted(set(sheets_on_disk) - labelled)[:50]}
    return tasks, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="sheets per source (test)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out-rows", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tasks, meta = load_tasks(a.limit, a.only)
    print(json.dumps(meta, indent=1), flush=True)
    out_rows = Path(a.out_rows or ROOT / "work/cells.jsonl")
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(out_rows.with_suffix(".sources.json"), "w"), indent=1)
    t0 = time.time(); done = 0
    with open(out_rows, "w") as f, Pool(a.workers) as pool:
        for rows in pool.imap_unordered(work, tasks, chunksize=4):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            done += 1
            if done % 250 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"{done}/{len(tasks)} sheets {el:.0f}s eta {el/done*(len(tasks)-done):.0f}s", flush=True)
                f.flush()
    print("EXTRACT_DONE", flush=True)

if __name__ == "__main__":
    main()
