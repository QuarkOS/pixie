"""Memory-capped palette/style sample grid + latency check. Does not poll, bind the
lock port, or touch the live worker. Safe to run next to pixie-chat if VRAM allows.

  PIXEL_MEM_FRACTION=0.30 python -u worker_v3.py --grid
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path

# Must be set before importing worker_v3 (it reads PIXEL_FAST_GRAPH at import).
os.environ["PIXEL_FAST_GRAPH"] = "0"
os.environ.setdefault("PIXEL_MAX_N", "4")
os.environ.setdefault("PIXEL_MEM_FRACTION", os.environ.get("PIXEL_MEM_FRACTION", "0.30"))

import numpy as np
import torch
import base64, io
from PIL import Image

import worker_v3 as W

HERE = Path(__file__).resolve().parent
OUT = HERE / "grids"
PROMPT = os.environ.get("PIXIE_COND_PROMPT", "red potion")
MEM_FRAC = float(os.environ.get("PIXEL_MEM_FRACTION", "0.30"))

PALETTES = [
    ("auto", None),
    ("ember", [376, 1635, 2342]),
    ("gold", [2343, 2827, 785]),
    ("forest", [2581, 1336, 3468]),
    ("ocean", [2573, 3865, 356]),
    ("royal", [214, 2703, 3473]),
    ("rose", [2442, 4085, 1069]),
    ("poison", [2279, 2903, 2720]),
    ("ice", [3327, 1231, 2659]),
    ("earth", [1169, 1532, 2683]),
    ("mono", [2775, 3859, 1345]),
    ("gameboy", [801, 2754, 1201]),
    ("sunset", [3316, 2999, 2842]),
    ("candy", [907, 4080, 3628]),
    ("blood", [3747, 2936, 3411]),
    ("copper", [2836, 2655, 2701]),
]
STYLES = [
    ("any", [0, 0, 0]),
    ("small-icon", [1, 1, 1]),   # small fill, dark outline, few colours
    ("medium", [2, 1, 2]),
    ("full-tile", [3, 1, 3]),
]


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def gpu_stats():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        used, total, util = [float(x.strip()) for x in out.split(",")]
        return {"used_mib": used, "total_mib": total, "util": util}
    except Exception as e:
        return {"error": str(e)}


def wait_idle(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = gpu_stats()
        if s.get("util", 100) <= 2 and s.get("used_mib", 99999) < 7000:
            return s
        time.sleep(1)
    return gpu_stats()


def load_light():
    """Model + CLIP only, small sampler, no reward-v7, no 128-wide CUDA graph."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC)
    W.SCORER_CFG = "feat_linear"
    W.USE_GRAPH = False
    W.N_SAMPLES_V7 = int(os.environ.get("PIXEL_MAX_N", "4"))
    return W.Gen()


def cond_of(pal, sty):
    if pal is None and (sty is None or sty == [0, 0, 0]):
        return None
    d = {}
    if pal is not None:
        d["palette"] = pal
    if sty is not None:
        d["style"] = sty
    return d or None


def main():
    OUT.mkdir(exist_ok=True)
    before = wait_idle()
    log("GPU before", before)
    if before.get("used_mib", 0) >= 7000:
        log("ABORT: GPU already busy, not loading a second model")
        sys.exit(2)
    g = load_light()
    after_load = gpu_stats()
    log("loaded light worker", after_load)

    # latency: n=8 default vs a real cond (same FLOPs; this is a parity check)
    def time_run(cond, n=4, reps=5):
        g.run(PROMPT, n, 1, 0, cond=cond)  # warmup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        xs = []
        for _ in range(reps):
            t0 = time.time()
            g.run(PROMPT, n, 1, 0, cond=cond)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            xs.append((time.time() - t0) * 1000)
        return {"ms": [round(x, 1) for x in xs], "mean_ms": round(float(np.mean(xs)), 1)}

    bench_default = time_run(None)
    bench_gold = time_run(cond_of([2343, 2827, 785], [1, 1, 1]))
    log("bench n=8 default", bench_default, "gold+small-icon", bench_gold)

    cells = []
    cell_imgs = []
    for pal_name, pal in PALETTES:
        row = []
        for sty_name, sty in STYLES:
            cond = cond_of(pal, sty)
            t0 = time.time()
            res, allr, ms, gen_ms, used = g.run(PROMPT, 4, 1, 0, cond=cond)
            png = res[0]["png"]
            raw = png.split(",", 1)[1]
            im = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGBA")
            # store a 128px cell
            im128 = im.resize((128, 128), Image.NEAREST)
            row.append(im128)
            cells.append({
                "palette": pal_name, "style": sty_name, "cond": cond,
                "ms": round(ms, 1), "gen_ms": round(gen_ms, 1),
                "score": res[0]["score"], "ncolors": res[0]["ncolors"], "fill": res[0]["fill"],
            })
        cell_imgs.append(row)

    pad, label_w, label_h, cell = 8, 96, 28, 128
    cols, rows = len(STYLES), len(PALETTES)
    Wim = label_w + cols * (cell + pad) + pad
    Him = label_h + rows * (cell + pad) + pad
    sheet = Image.new("RGBA", (Wim, Him), (17, 17, 17, 255))
    # labels are not drawn as TTF (avoid font deps); filenames + report carry names.
    # Draw a 1-px accent bar per named palette using its first swatch, via the cell itself.
    for r, row in enumerate(cell_imgs):
        for c, im in enumerate(row):
            x = label_w + c * (cell + pad)
            y = label_h + r * (cell + pad)
            sheet.paste(im, (x, y), im)
    grid_path = OUT / "palette_style_grid.png"
    sheet.save(grid_path)
    # also a caption map as a narrow sidecar text image would be overkill; write a legend txt
    legend = OUT / "palette_style_grid.txt"
    legend.write_text(
        "Rows (top to bottom): " + ", ".join(p[0] for p in PALETTES) + "\n"
        "Columns (left to right): " + ", ".join(s[0] for s in STYLES) + "\n"
        f"Prompt: {PROMPT}. One top-ranked sample of 4 per cell, feat_linear scorer, no v7.\n"
    )
    report = {
        "prompt": PROMPT,
        "gpu_before": before,
        "gpu_after_load": after_load,
        "gpu_after": gpu_stats(),
        "mem_fraction": MEM_FRAC,
        "bench_n8_default": bench_default,
        "bench_n8_gold_small": bench_gold,
        "note": (
            "A full 64-candidate reward-v7 job has the same sequence length and FLOPs "
            "with or without a non-default cond (the live worker already prefixes default "
            "NO_PALETTE + ANY). n=8 times here are a parity check, not the live 64-cand path."
        ),
        "cells": cells,
        "grid": str(grid_path),
        "legend": str(legend),
        "rows": [p[0] for p in PALETTES],
        "cols": [s[0] for s in STYLES],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    log("wrote", grid_path, "cells", len(cells))
    # free as much as we can; do not touch other processes
    del g
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("GPU after release", gpu_stats())
    print("GRID_OK", json.dumps({"grid": str(grid_path), "bench_default": bench_default, "bench_cond": bench_gold}))


if __name__ == "__main__":
    main()
