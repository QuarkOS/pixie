#!/usr/bin/env python3
"""Side-by-side visual check: live v3 decay img_040M vs condboost full_064M.

Loads one checkpoint at a time, memory-capped, no CUDA graph, no reward-v7,
does not poll or bind the chat lock port. Safe to run next to pixie-chat.
Does not write into any run/checkpoint directory.
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path

os.environ["PIXEL_FAST_GRAPH"] = "0"
os.environ.setdefault("PIXEL_MAX_N", "8")
os.environ.setdefault("PIXEL_MEM_FRACTION", "0.30")

HERE = Path("/path/to/pixie")
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import worker_v3 as W
import v3_model as M

OUT = Path("/path/to/pixie")
LIVE_CKPT = Path("/path/to/pixie")
COND_CKPT = Path("/path/to/pixie")
MEM_FRAC = float(os.environ["PIXEL_MEM_FRACTION"])
CELL = 128
# Not Blind-50 or eval-50 (sampling those would be allowed; these stay off the lists anyway).
PROMPTS = [
    "copper golem",
    "tiny azure jelly",
    "rusted cutlass",
    "golden crown",
    "forest treant",
    "fire elemental",
    "dwarven axeman",
    "twisted oak staff",
    "clockwork beetle",
    "pumpkin spirit",
    "coral trident",
    "ornate jewelry box",
]
SEEDS = [101, 202, 303]
GUIDE_PROMPTS = ["red potion", "copper golem"]  # red potion is eval-50; sampling only
GUIDE_SEEDS = [404, 505, 606]
PALETTES = [
    ("auto", None),
    ("ember", [376, 1635, 2342]),
    ("gold", [2343, 2827, 785]),
    ("ocean", [2573, 3865, 356]),
    ("gameboy", [801, 2754, 1201]),
]
STYLE_ROWS = [
    ("any", [0, 0, 0]),
    ("small-icon", [1, 1, 1]),
    ("full-tile", [3, 1, 3]),
]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG = (18, 18, 20, 255)
FG = (230, 230, 230, 255)
MUTED = (160, 164, 170, 255)
ACCENT = (120, 180, 255, 255)


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


def font(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, sz)


def checker(w, h, s=8):
    im = Image.new("RGBA", (w, h), (40, 40, 44, 255))
    px = im.load()
    c2 = (55, 55, 60, 255)
    for y in range(h):
        for x in range(w):
            if ((x // s) + (y // s)) % 2:
                px[x, y] = c2
    return im


def to_cell(rgba, size=CELL):
    arr = np.asarray(rgba, dtype=np.uint8)
    im = Image.fromarray(arr, "RGBA").resize((size, size), Image.NEAREST)
    bg = checker(size, size)
    bg.paste(im, (0, 0), im)
    return bg


def rgba_from_idx(ids, palette):
    return W.rgba_from_indices(ids, palette)


def sample_fixed(g, prompt, seeds, cond=None):
    n = len(seeds)
    row = W.trim(W.t26(g.tok, prompt), g.TL)
    blank = [M.CLIP_BOS] + [M.CLIP_EOS] * (g.TL - 1)
    h = g._encode([row] * n)
    hu = g._encode([blank] * n)
    seed_t = torch.tensor(seeds, dtype=torch.int64, device=g.device)
    ct = g.cond_tensor(cond, n) if hasattr(g, "cond_tensor") else None
    t0 = time.time()
    with torch.no_grad():
        img = g.sampler.sample(h, hu, seeds=seed_t, cond=ct)
    if g.device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000
    ind = img.cpu().numpy()
    sprites = [rgba_from_idx(np.asarray(r, dtype=np.int64).reshape(-1), g.palette) for r in ind]
    return sprites, ms


def load_model(ckpt: Path):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC)
    W.SCORER_CFG = "feat_linear"
    W.USE_GRAPH = False
    W.N_SAMPLES_V7 = int(os.environ.get("PIXEL_MAX_N", "8"))
    W.CKPT = Path(ckpt)
    W.WANT_SHA = ""
    log("loading", ckpt)
    t0 = time.time()
    g = W.Gen()
    # fix run label to the actual parent folder
    g.info["run"] = Path(ckpt).parent.name
    g.info["file"] = Path(ckpt).name
    log("loaded", g.info.get("run"), g.info.get("file"),
        "step", g.info.get("step"), "images", g.info.get("images"),
        "n_pal", g.info.get("n_pal"), "style", g.info.get("style"),
        "sha", str(g.info.get("sha"))[:16],
        f"{time.time()-t0:.1f}s", gpu_stats())
    return g


def release(g):
    try:
        del g.sampler
        del g.model
        del g.clip
        if getattr(g, "v7", None) is not None:
            del g.v7
    except Exception:
        pass
    del g
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("released", gpu_stats())


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
    before = gpu_stats()
    log("GPU before", before)
    if before.get("used_mib", 0) >= 7000:
        log("ABORT: GPU already busy, not loading a second model")
        sys.exit(2)

    results = {"live": {"plain": {}, "guide": {}}, "condboost": {"plain": {}, "guide": {}}}
    timings = {}
    infos = {}

    for tag, ckpt in (("live", LIVE_CKPT), ("condboost", COND_CKPT)):
        g = load_model(ckpt)
        infos[tag] = {k: g.info[k] for k in g.info if k != "config"}
        # plain prompts
        t_all = 0.0
        for prompt in PROMPTS:
            sprites, ms = sample_fixed(g, prompt, SEEDS, cond=None)
            t_all += ms
            results[tag]["plain"][prompt] = sprites
            log(tag, "plain", prompt, f"{ms:.0f}ms")
        timings[tag + "_plain_ms"] = round(t_all, 1)
        # guidance
        t_all = 0.0
        for prompt in GUIDE_PROMPTS:
            results[tag]["guide"][prompt] = {"palettes_any": {}, "gold_styles": {}}
            for pal_name, pal in PALETTES:
                cond = cond_of(pal, [0, 0, 0])
                sprites, ms = sample_fixed(g, prompt, [GUIDE_SEEDS[0]], cond=cond)
                t_all += ms
                results[tag]["guide"][prompt]["palettes_any"][pal_name] = sprites[0]
                log(tag, "pal", prompt, pal_name, f"{ms:.0f}ms")
            for sty_name, sty in STYLE_ROWS:
                cond = cond_of([2343, 2827, 785], sty)  # gold
                sprites, ms = sample_fixed(g, prompt, [GUIDE_SEEDS[1]], cond=cond)
                t_all += ms
                results[tag]["guide"][prompt]["gold_styles"][sty_name] = sprites[0]
                log(tag, "style", prompt, sty_name, f"{ms:.0f}ms")
        timings[tag + "_guide_ms"] = round(t_all, 1)
        release(g)

    # ---------- draw ----------
    pad = 10
    label_w = 220
    header_h = 86
    section_h = 36
    row_h = CELL + 16
    col_gap = 16
    group_gap = 28
    n_s = len(SEEDS)
    # two groups of 3 cells
    group_w = n_s * (CELL + 4) - 4
    content_w = label_w + group_w + group_gap + group_w + pad
    # main section
    y = header_h
    y += section_h + len(PROMPTS) * row_h
    # guidance palette section title + per prompt: 2 model rows * palette cols
    pal_cell = 112
    pal_row_h = pal_cell + 16
    y += section_h  # "guidance palettes"
    y += len(GUIDE_PROMPTS) * (22 + 2 * pal_row_h + 8)
    y += section_h  # "guidance styles (gold)"
    y += len(GUIDE_PROMPTS) * (22 + 2 * pal_row_h + 8)
    y += 24
    Wim = max(content_w, label_w + len(PALETTES) * (pal_cell + 8) + pad) + pad
    Him = y + pad
    sheet = Image.new("RGBA", (Wim, Him), BG)
    dr = ImageDraw.Draw(sheet)

    def text(xy, s, sz=16, bold=False, fill=FG, anchor="lt"):
        dr.text(xy, s, font=font(sz, bold), fill=fill, anchor=anchor)

    text((pad, 12), "Pixie v3 visual check  —  live decay vs condboost", 22, True)
    text((pad, 42),
         "Same seeds, CFG 2.0, temp 0.8, top-k 32, default cond (no palette/style) on the main grid.  "
         "13 Nov 2026.",
         13, fill=MUTED)
    live_info = infos["live"]
    cond_info = infos["condboost"]
    text((pad, 62),
         f"Live: {live_info.get('run')}/{live_info.get('file')}  step {live_info.get('step')}  "
         f"images {live_info.get('images')}    |    "
         f"Condboost: {cond_info.get('run')}/{cond_info.get('file')}  step {cond_info.get('step')}  "
         f"images {cond_info.get('images')}  (img_064M.pt has identical weights)",
         12, fill=MUTED)

    # column headers
    x_live = label_w
    x_cond = label_w + group_w + group_gap
    y = header_h
    text((x_live, y), "Live v3  (decay 40M)", 16, True, ACCENT)
    text((x_cond, y), "Condboost  (64M)", 16, True, ACCENT)
    y += section_h

    for prompt in PROMPTS:
        text((pad, y + CELL // 2), prompt, 15, True, anchor="lm")
        for i, sp in enumerate(results["live"]["plain"][prompt]):
            sheet.paste(to_cell(sp), (x_live + i * (CELL + 4), y))
        for i, sp in enumerate(results["condboost"]["plain"][prompt]):
            sheet.paste(to_cell(sp), (x_cond + i * (CELL + 4), y))
        y += row_h

    # palette guidance
    y += 8
    text((pad, y), "Palette guidance  —  style = any,  same seed per prompt.  Both models have cond slots.", 16, True)
    y += section_h
    pal_xs = [label_w + i * (pal_cell + 8) for i in range(len(PALETTES))]
    for prompt in GUIDE_PROMPTS:
        text((pad, y), prompt + "  (eval-50, sampling only)" if prompt == "red potion" else prompt, 14, True)
        y += 22
        # palette names
        for i, (name, _) in enumerate(PALETTES):
            text((pal_xs[i] + pal_cell // 2, y), name, 12, fill=MUTED, anchor="mt")
        y += 16
        for tag, label in (("live", "live"), ("condboost", "condboost")):
            text((pad, y + pal_cell // 2), label, 13, anchor="lm")
            for i, (name, _) in enumerate(PALETTES):
                sheet.paste(to_cell(results[tag]["guide"][prompt]["palettes_any"][name], pal_cell),
                            (pal_xs[i], y))
            y += pal_row_h
        y += 8

    text((pad, y), "Style guidance  —  palette = gold,  same seed per prompt.", 16, True)
    y += section_h
    sty_xs = [label_w + i * (pal_cell + 8) for i in range(len(STYLE_ROWS))]
    for prompt in GUIDE_PROMPTS:
        text((pad, y), prompt, 14, True)
        y += 22
        for i, (name, _) in enumerate(STYLE_ROWS):
            text((sty_xs[i] + pal_cell // 2, y), name, 12, fill=MUTED, anchor="mt")
        y += 16
        for tag, label in (("live", "live"), ("condboost", "condboost")):
            text((pad, y + pal_cell // 2), label, 13, anchor="lm")
            for i, (name, _) in enumerate(STYLE_ROWS):
                sheet.paste(to_cell(results[tag]["guide"][prompt]["gold_styles"][name], pal_cell),
                            (sty_xs[i], y))
            y += pal_row_h
        y += 8

    out_png = OUT / "condboost-check.png"
    sheet.save(out_png, "PNG")
    meta = {
        "out": str(out_png),
        "prompts": PROMPTS,
        "seeds": SEEDS,
        "guide_prompts": GUIDE_PROMPTS,
        "guide_seeds": {"palettes": GUIDE_SEEDS[0], "styles": GUIDE_SEEDS[1]},
        "live_ckpt": str(LIVE_CKPT),
        "cond_ckpt": str(COND_CKPT),
        "infos": infos,
        "timings": timings,
        "gpu_before": before,
        "gpu_after": gpu_stats(),
        "note": "img_064M.pt and full_064M.pt have identical model weights. Guidance palettes/styles match chat-worker-v3cond/cond_grid.py.",
    }
    (OUT / "condboost-check.json").write_text(json.dumps(meta, indent=2, default=str))
    log("wrote", out_png, "size", out_png.stat().st_size, "gpu", gpu_stats())
    print("GRID_OK", json.dumps({"grid": str(out_png), "timings": timings, "gpu_after": gpu_stats()}))


if __name__ == "__main__":
    main()