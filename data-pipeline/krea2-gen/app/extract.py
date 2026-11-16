#!/usr/bin/env python3
"""Split sprite sheets into cells, mode-pool each cell to an NxN sprite, clean the palette,
key out magenta, and score each cell for usability.
usage: extract.py RUN_DIR JOBS_JSON OUT_DIR --px 16|32 [--contact]"""
import argparse, json, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pitch import pitch_of

MAG = np.array([255, 0, 255])

def is_bg(rgb, tol=90):
    a = rgb.astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    fringe = (g < 70) & (r > 150) & (b > 150) & (np.abs(r - b) < 80)   # magenta-tinted edge pixels
    return (np.abs(a - MAG).max(-1) <= tol) | fringe

def mode_pool(cell, n):
    """cell HxWx3 -> nxnx3 using the most common (quantized) colour in the inner 60% of each block."""
    h, w = cell.shape[:2]
    out = np.zeros((n, n, 3), np.uint8); purity = np.zeros((n, n))
    for y in range(n):
        for x in range(n):
            y0, y1 = h * y / n, h * (y + 1) / n; x0, x1 = w * x / n, w * (x + 1) / n
            my, mx = (y1 - y0) * 0.2, (x1 - x0) * 0.2
            blk = cell[int(round(y0 + my)):max(int(round(y1 - my)), int(round(y0 + my)) + 1),
                       int(round(x0 + mx)):max(int(round(x1 - mx)), int(round(x0 + mx)) + 1)].reshape(-1, 3)
            q = (blk // 16).astype(int); key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
            vals, cnt = np.unique(key, return_counts=True); k = vals[cnt.argmax()]
            sel = blk[key == k]
            out[y, x] = np.median(sel, 0).astype(np.uint8); purity[y, x] = cnt.max() / len(key)
    return out, purity

def palette_clean(rgb, alpha, thresh=28, max_colors=None):
    px = rgb[alpha > 0].astype(int)
    if len(px) == 0: return rgb
    cols, cnt = np.unique(px, axis=0, return_counts=True)
    order = np.argsort(-cnt); reps = []
    for c in cols[order]:
        if not any(np.abs(c - r).max() <= thresh for r in reps):
            reps.append(c)
    reps = np.array(reps)
    if max_colors and len(reps) > max_colors: reps = reps[:max_colors]
    flat = rgb.reshape(-1, 3).astype(int)
    d = np.abs(flat[:, None, :] - reps[None]).max(-1)
    new = reps[d.argmin(1)].reshape(rgb.shape).astype(np.uint8)
    return np.where(alpha[..., None] > 0, new, rgb)

def score_cell(cell):
    fg = ~is_bg(cell)
    frac = fg.mean()
    h, w = fg.shape; b = max(2, int(min(h, w) * 0.03))
    border = np.concatenate([fg[:b].ravel(), fg[-b:].ravel(), fg[:, :b].ravel(), fg[:, -b:].ravel()]).mean()
    lab, n = ndimage.label(fg)
    if n:
        sizes = ndimage.sum(fg, lab, range(1, n + 1)); main = sizes.max() / fg.sum()
    else:
        main = 0
    return {"fg": round(float(frac), 3), "border": round(float(border), 3), "main_comp": round(float(main), 3)}

def spill(img, g, r, c):
    """share of the cell's interior boundary where foreground continues across into the neighbour cell"""
    H, W = img.shape[:2]; fg = (~is_bg(img)) & (img.astype(int).max(-1) > 70); tot = 0; hit = 0
    y0, y1 = round(H * r / g), round(H * (r + 1) / g); x0, x1 = round(W * c / g), round(W * (c + 1) / g)
    if r > 0: a, b = fg[y0, x0:x1], fg[y0 - 1, x0:x1]; tot += len(a); hit += (a & b).sum()
    if r < g - 1: a, b = fg[y1 - 1, x0:x1], fg[y1, x0:x1]; tot += len(a); hit += (a & b).sum()
    if c > 0: a, b = fg[y0:y1, x0], fg[y0:y1, x0 - 1]; tot += len(a); hit += (a & b).sum()
    if c < g - 1: a, b = fg[y0:y1, x1 - 1], fg[y0:y1, x1]; tot += len(a); hit += (a & b).sum()
    return float(hit) / max(tot, 1)

def structure_ok(s):
    """Everything in the usability gate except the 0.75 purity threshold."""
    return 0.04 <= s["fg"] <= 0.90 and s["spill"] <= 0.08 and s["main_comp"] >= 0.85

def usable(s, purity_mean):
    return structure_ok(s) and purity_mean >= 0.75

def bin_for(s, purity_mean):
    """ok / purity_fail (structure passes, purity < 0.75) / rejected (other gates fail)."""
    if structure_ok(s) and purity_mean >= 0.75:
        return "ok"
    if structure_ok(s):
        return "purity_fail"
    return "rejected"

def touches_32_edge(alpha):
    """True if any opaque pixel sits on the border of a 32x32 sprite."""
    a = alpha > 0
    return bool(a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs"); ap.add_argument("out"); ap.add_argument("--px", type=int, default=16)
    ap.add_argument("--src", default=None, help="dir with sheet PNGs (default: path in jobs)")
    a = ap.parse_args()
    jobs = [j for j in json.load(open(a.jobs)) if "warmup" not in j["out"]]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for j in jobs:
        p = j["out"] if a.src is None else os.path.join(a.src, os.path.basename(j["out"]))
        img = np.asarray(Image.open(p).convert("RGB")); H, W = img.shape[:2]; g = j["grid"]
        for r in range(g):
            for c in range(g):
                k = r * g + c
                cell = img[round(H * r / g):round(H * (r + 1) / g), round(W * c / g):round(W * (c + 1) / g)]
                s = score_cell(cell); s["spill"] = round(spill(img, g, r, c), 3)
                ppx, ppy, _, _ = pitch_of(cell); s["art_px"] = round(cell.shape[1] / ppx, 1)
                sp, pur = mode_pool(cell, a.px)
                alpha = np.where(is_bg(sp), 0, 255).astype(np.uint8)
                sp = palette_clean(sp, alpha)
                # drop single isolated stray pixels
                lab, n = ndimage.label(alpha > 0)
                if n > 1:
                    sizes = ndimage.sum(alpha > 0, lab, range(1, n + 1))
                    for i, sz in enumerate(sizes, 1):
                        if sz <= 1: alpha[lab == i] = 0
                fgb = alpha > 0
                pm = float(pur[fgb].mean()) if fgb.any() else 0.0
                ok = usable(s, pm)
                name = f"{os.path.splitext(os.path.basename(p))[0]}-{k}-{j['items'][k].replace(' ', '_')}.png"
                Image.fromarray(np.dstack([sp, alpha]), "RGBA").save(os.path.join(a.out, name))
                rows.append({"file": name, "subject": j["items"][k], "usable_auto": bool(ok), "purity": round(pm, 3), **s})
    json.dump(rows, open(os.path.join(a.out, "cells.json"), "w"), indent=1)
    n = len(rows); u = int(sum(bool(r["usable_auto"]) for r in rows))
    print(json.dumps({"cells": n, "usable_auto": u, "yield": round(u / max(n, 1), 3),
                      "purity_mean": round(float(np.mean([r["purity"] for r in rows])), 3),
                      "art_px_median": float(np.median([r["art_px"] for r in rows])),
                      "art_px_ge_px": int(sum(r["art_px"] >= a.px * 0.9 for r in rows))}))

if __name__ == "__main__":
    main()
