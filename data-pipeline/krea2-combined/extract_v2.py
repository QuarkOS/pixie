#!/usr/bin/env python3
"""Extractor v2 for 3x3 @ 576 (6 screen px per art px).

Separate from the live job's extract.py. Does not read or write the
running generation directory.

Changes vs v1 (app/extract.py):
- Wider magenta key tolerance (default KEY_TOL=110, was 90).
- Edge-only fringe cleanup: strip magenta-tinted pixels on the
  foreground outline (and a second pass), without applying the looser
  tint rule to interior colours (pink frosting, purple subjects).
- Recalibrated usability gates for 6 px/art px (see THRESHOLDS).
- edge_touch remains a flag, not a hard reject.

usage:
  extract_v2.py JOBS_JSON_OR_JSONL OUT_DIR --px 32 --src SHEET_DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# optional pitch (art_px); never required
try:
    _APP = Path(os.environ.get("PITCH_APP_DIR", "/nonexistent"))
    if _APP.is_dir():
        sys.path.insert(0, str(_APP))
    from pitch import pitch_of  # type: ignore
except Exception:
    pitch_of = None

MAG = np.array([255, 0, 255])
VERSION = "extract_v2_3x3_576"

# Locked after the labelled 378-sprite sweep (see REPORT.md).
#
# A naive widen of the global key (tol=110) plus a loose magenta-tint rule
# punches holes in pink/purple subjects (jelly donut frosting, crystal
# dragon, nail polish, grapes). v2 instead:
#   1) Keeps the v1 seed key (tol=90 + the original tint rule) on the 32x32.
#   2) Widens tolerance only by flood-filling from already-keyed magenta
#      into nearby near-magenta (grow_tol=120, 6 px) on the raw 192 cell,
#      which is what spill/fg/main_comp are scored on.
#   3) Strips only near-exact magenta (tol<=100) on the 32x32 outline.
KEY_TOL = 90
GROW_TOL = 120
GROW_ROUNDS = 6
TINT_G_MAX = 70
TINT_RB_MIN = 150
TINT_RB_DIFF = 80
EDGE_KEY_TOL = 100
FRINGE_ROUNDS = 2

THRESHOLDS = {
    "fg_min": 0.04,
    "fg_max": 1.00,        # 0.90 was rejecting complete full-frame objects
    "spill_max": 0.08,     # kept; grown key drops false spill on peach / nightstand
    "main_comp_min": 0.55, # 0.85 was rejecting a correctly broken pill
    "purity_min": 0.35,    # 0.75 was tuned for 8 px/art px; 6 px runs lower
}


def is_magenta_key(rgb, tol=KEY_TOL):
    a = rgb.astype(int)
    return np.abs(a - MAG).max(-1) <= tol


def is_magenta_tint(rgb, g_max=TINT_G_MAX, rb_min=TINT_RB_MIN, rb_diff=TINT_RB_DIFF):
    a = rgb.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (g < g_max) & (r > rb_min) & (b > rb_min) & (np.abs(r - b) < rb_diff)


def is_bg(rgb, tol=KEY_TOL):
    """Seed key used on the 32x32 sprite (same rule as extract v1)."""
    return is_magenta_key(rgb, tol=tol) | is_magenta_tint(rgb)


def is_bg_grown(rgb, seed_tol=KEY_TOL, grow_tol=GROW_TOL, rounds=GROW_ROUNDS):
    """Widen the key only from already-keyed magenta into adjacent
    near-magenta pixels. Used on raw 192-px cells for fg/spill/main_comp."""
    bg = is_bg(rgb, tol=seed_tol)
    grow = is_magenta_key(rgb, tol=grow_tol)
    for _ in range(max(0, int(rounds))):
        near = ndimage.binary_dilation(bg) & grow & ~bg
        if not near.any():
            break
        bg = bg | near
    return bg


def is_edge_fringe(rgb, tol=EDGE_KEY_TOL):
    """Near-exact magenta only (not a loose pink/purple tint)."""
    return is_magenta_key(rgb, tol=tol)


def strip_edge_fringe(rgb, alpha, rounds=FRINGE_ROUNDS, tol=EDGE_KEY_TOL):
    """Clear near-exact magenta outline pixels. Interior colours stay."""
    alpha = alpha.copy()
    tint = is_edge_fringe(rgb, tol=tol)
    dropped = np.zeros(alpha.shape, dtype=np.int32)
    for _ in range(max(0, int(rounds))):
        fg = alpha > 0
        if not fg.any():
            break
        bg = ~fg
        edge = fg & ndimage.binary_dilation(bg)
        drop = edge & tint
        dropped += drop.astype(np.int32)
        alpha[drop] = 0
    return alpha, int(dropped.sum())


def mode_pool(cell, n):
    """cell HxWx3 -> nxnx3 using the most common (quantized) colour in the inner 60% of each block."""
    h, w = cell.shape[:2]
    out = np.zeros((n, n, 3), np.uint8)
    purity = np.zeros((n, n))
    for y in range(n):
        for x in range(n):
            y0, y1 = h * y / n, h * (y + 1) / n
            x0, x1 = w * x / n, w * (x + 1) / n
            my, mx = (y1 - y0) * 0.2, (x1 - x0) * 0.2
            blk = cell[
                int(round(y0 + my)): max(int(round(y1 - my)), int(round(y0 + my)) + 1),
                int(round(x0 + mx)): max(int(round(x1 - mx)), int(round(x0 + mx)) + 1),
            ].reshape(-1, 3)
            q = (blk // 16).astype(int)
            key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
            vals, cnt = np.unique(key, return_counts=True)
            k = vals[cnt.argmax()]
            sel = blk[key == k]
            out[y, x] = np.median(sel, 0).astype(np.uint8)
            purity[y, x] = cnt.max() / len(key)
    return out, purity


def palette_clean(rgb, alpha, thresh=28, max_colors=None):
    px = rgb[alpha > 0].astype(int)
    if len(px) == 0:
        return rgb
    cols, cnt = np.unique(px, axis=0, return_counts=True)
    order = np.argsort(-cnt)
    reps = []
    for c in cols[order]:
        if not any(np.abs(c - r).max() <= thresh for r in reps):
            reps.append(c)
    reps = np.array(reps)
    if max_colors and len(reps) > max_colors:
        reps = reps[:max_colors]
    flat = rgb.reshape(-1, 3).astype(int)
    d = np.abs(flat[:, None, :] - reps[None]).max(-1)
    new = reps[d.argmin(1)].reshape(rgb.shape).astype(np.uint8)
    return np.where(alpha[..., None] > 0, new, rgb)


def drop_stray_pixels(alpha, max_size=1):
    alpha = alpha.copy()
    lab, n = ndimage.label(alpha > 0)
    if n > 1:
        sizes = ndimage.sum(alpha > 0, lab, range(1, n + 1))
        for i, sz in enumerate(sizes, 1):
            if sz <= max_size:
                alpha[lab == i] = 0
    return alpha


def score_cell(cell, tol=KEY_TOL, grown=True):
    fg = ~(is_bg_grown(cell, seed_tol=tol) if grown else is_bg(cell, tol=tol))
    frac = fg.mean()
    h, w = fg.shape
    b = max(2, int(min(h, w) * 0.03))
    border = np.concatenate(
        [fg[:b].ravel(), fg[-b:].ravel(), fg[:, :b].ravel(), fg[:, -b:].ravel()]
    ).mean()
    lab, n = ndimage.label(fg)
    if n:
        sizes = ndimage.sum(fg, lab, range(1, n + 1))
        main = sizes.max() / fg.sum()
        n_comp = int(n)
    else:
        main = 0
        n_comp = 0
    return {
        "fg": round(float(frac), 3),
        "border": round(float(border), 3),
        "main_comp": round(float(main), 3),
        "n_comp": n_comp,
    }


def spill(img, g, r, c, tol=KEY_TOL, grown=True):
    """Share of the cell's interior boundary where foreground continues into the neighbour cell."""
    H, W = img.shape[:2]
    bg = is_bg_grown(img, seed_tol=tol) if grown else is_bg(img, tol=tol)
    fg = (~bg) & (img.astype(int).max(-1) > 70)
    tot = 0
    hit = 0
    y0, y1 = round(H * r / g), round(H * (r + 1) / g)
    x0, x1 = round(W * c / g), round(W * (c + 1) / g)
    if r > 0:
        a, b = fg[y0, x0:x1], fg[y0 - 1, x0:x1]
        tot += len(a)
        hit += (a & b).sum()
    if r < g - 1:
        a, b = fg[y1 - 1, x0:x1], fg[y1, x0:x1]
        tot += len(a)
        hit += (a & b).sum()
    if c > 0:
        a, b = fg[y0:y1, x0], fg[y0:y1, x0 - 1]
        tot += len(a)
        hit += (a & b).sum()
    if c < g - 1:
        a, b = fg[y0:y1, x1 - 1], fg[y0:y1, x1]
        tot += len(a)
        hit += (a & b).sum()
    return float(hit) / max(tot, 1)


def touches_32_edge(alpha):
    a = alpha > 0
    return bool(a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any())


def structure_ok(s, thr=None):
    thr = thr or THRESHOLDS
    return (
        thr["fg_min"] <= s["fg"] <= thr["fg_max"]
        and s["spill"] <= thr["spill_max"]
        and s["main_comp"] >= thr["main_comp_min"]
    )


def usable(s, purity_mean, thr=None):
    thr = thr or THRESHOLDS
    return structure_ok(s, thr) and purity_mean >= thr["purity_min"]


def bin_for(s, purity_mean, thr=None):
    """ok / purity_fail (structure passes, purity below min) / rejected."""
    thr = thr or THRESHOLDS
    if structure_ok(s, thr) and purity_mean >= thr["purity_min"]:
        return "ok"
    if structure_ok(s, thr):
        return "purity_fail"
    return "rejected"


def reject_reasons(s, purity_mean, thr=None):
    thr = thr or THRESHOLDS
    reasons = []
    if s["fg"] < thr["fg_min"]:
        reasons.append(f"fg<{thr['fg_min']} ({s['fg']})")
    if s["fg"] > thr["fg_max"]:
        reasons.append(f"fg>{thr['fg_max']} ({s['fg']})")
    if s["spill"] > thr["spill_max"]:
        reasons.append(f"spill>{thr['spill_max']} ({s['spill']})")
    if s["main_comp"] < thr["main_comp_min"]:
        reasons.append(f"main_comp<{thr['main_comp_min']} ({s['main_comp']})")
    if purity_mean < thr["purity_min"]:
        reasons.append(f"purity<{thr['purity_min']} ({round(purity_mean, 3)})")
    return reasons


def extract_cell(img, g, r, c, px=32, tol=KEY_TOL, fringe_rounds=FRINGE_ROUNDS, do_fringe=True):
    """Extract one cell from a raw sheet. Returns rgb32, alpha32, score, purity, extras."""
    H, W = img.shape[:2]
    cell = img[round(H * r / g): round(H * (r + 1) / g), round(W * c / g): round(W * (c + 1) / g)]
    s = score_cell(cell, tol=tol)
    s["spill"] = round(spill(img, g, r, c, tol=tol), 3)
    if pitch_of is not None:
        try:
            ppx, ppy, _, _ = pitch_of(cell)
            s["art_px"] = round(cell.shape[1] / max(ppx, 1e-6), 1)
        except Exception:
            s["art_px"] = None
    else:
        s["art_px"] = None
    sp, pur = mode_pool(cell, px)
    alpha = np.where(is_bg(sp, tol=tol), 0, 255).astype(np.uint8)
    fringe_dropped = 0
    if do_fringe:
        alpha, fringe_dropped = strip_edge_fringe(sp, alpha, rounds=fringe_rounds)
    alpha = drop_stray_pixels(alpha)
    sp = palette_clean(sp, alpha)
    fgb = alpha > 0
    pm = float(pur[fgb].mean()) if fgb.any() else 0.0
    return {
        "rgb": sp,
        "alpha": alpha,
        "score": s,
        "purity": pm,
        "fringe_dropped": fringe_dropped,
        "edge_touch": touches_32_edge(alpha),
        "cell": cell,
    }


def load_jobs(path):
    """Accept eval jobs.json (list, items/out) or gen jobs.jsonl (cells/sheet)."""
    path = Path(path)
    if path.suffix == ".jsonl":
        jobs = [json.loads(l) for l in path.open() if l.strip()]
    else:
        jobs = json.load(path.open())
        if isinstance(jobs, dict):
            jobs = jobs.get("jobs", jobs.get("sheets", [jobs]))
    out = []
    for j in jobs:
        if "warmup" in str(j.get("out", "")):
            continue
        grid = int(j.get("grid") or 3)
        if "items" in j:
            phrases = list(j["items"])
        elif "cells" in j:
            phrases = [c["phrase"] if isinstance(c, dict) else str(c) for c in j["cells"]]
        else:
            phrases = [f"cell-{k}" for k in range(grid * grid)]
        sheet = j.get("out") or j.get("sheet") or j.get("path") or j.get("file")
        sheet_id = j.get("sheet_id", j.get("id"))
        out.append({"grid": grid, "phrases": phrases, "sheet": sheet, "sheet_id": sheet_id, "raw": j})
    return out


def resolve_sheet(job, src):
    p = job["sheet"]
    if src:
        if p:
            cand = Path(src) / Path(p).name
            if cand.is_file():
                return cand
        # fallback: common names
        sid = job["sheet_id"]
        for name in (
            f"sheet-{sid:03d}.png" if isinstance(sid, int) else None,
            f"sheet-{sid:06d}.png" if isinstance(sid, int) else None,
            f"sheet-{sid}.png" if sid is not None else None,
        ):
            if name and (Path(src) / name).is_file():
                return Path(src) / name
        pngs = sorted(Path(src).glob("sheet-*.png"))
        # if sheet_id is an index into the src dir
        if isinstance(sid, int) and pngs:
            # try exact match already failed; leave to caller
            pass
        if p and Path(p).is_file():
            return Path(p)
        raise FileNotFoundError(f"sheet not found for job {job['sheet_id']}: {p} in {src}")
    if p and Path(p).is_file():
        return Path(p)
    raise FileNotFoundError(f"sheet not found: {p}")


def main():
    ap = argparse.ArgumentParser(description="Krea 3x3 extractor v2 (does not touch a running job's data dir).")
    ap.add_argument("jobs")
    ap.add_argument("out")
    ap.add_argument("--px", type=int, default=32)
    ap.add_argument("--src", default=None, help="directory with raw sheet PNGs")
    ap.add_argument("--dry-run", action="store_true", help="print counts only, still write this out dir")
    ap.add_argument("--no-fringe", action="store_true")
    ap.add_argument("--key-tol", type=int, default=KEY_TOL)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for sub in ("ok", "purity_fail", "rejected"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    jobs = load_jobs(args.jobs)
    rows = []
    for j in jobs:
        p = resolve_sheet(j, args.src)
        img = np.asarray(Image.open(p).convert("RGB"))
        g = j["grid"]
        for k, phrase in enumerate(j["phrases"]):
            r, c = divmod(k, g)
            ex = extract_cell(
                img, g, r, c, px=args.px, tol=args.key_tol,
                do_fringe=not args.no_fringe,
            )
            s, pm = ex["score"], ex["purity"]
            dest = bin_for(s, pm)
            reasons = reject_reasons(s, pm)
            name = f"{p.stem}-{k}-{phrase.replace(' ', '_')}.png"
            rgba = Image.fromarray(np.dstack([ex["rgb"], ex["alpha"]]), "RGBA")
            rgba.save(os.path.join(args.out, dest, name))
            rows.append({
                "file": name,
                "subject": phrase,
                "bin": dest,
                "usable_auto": dest == "ok",
                "purity": round(pm, 3),
                "edge_touch": ex["edge_touch"],
                "fringe_dropped": ex["fringe_dropped"],
                "reject_reasons": reasons,
                "extractor": VERSION,
                **s,
            })
    json.dump(rows, open(os.path.join(args.out, "cells.json"), "w"), indent=1)
    n = len(rows)
    counts = {b: int(sum(r["bin"] == b for r in rows)) for b in ("ok", "purity_fail", "rejected")}
    summary = {
        "extractor": VERSION,
        "thresholds": THRESHOLDS,
        "key_tol": args.key_tol,
        "fringe": not args.no_fringe,
        "cells": n,
        **counts,
        "yield": round(counts["ok"] / max(n, 1), 3),
        "purity_mean": round(float(np.mean([r["purity"] for r in rows])), 3) if rows else 0,
        "edge_touch": int(sum(r["edge_touch"] for r in rows)),
        "fringe_dropped_mean": round(float(np.mean([r["fringe_dropped"] for r in rows])), 3) if rows else 0,
    }
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
