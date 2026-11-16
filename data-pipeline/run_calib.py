#!/usr/bin/env python3
"""Calibrate extract_v2 gates on the labelled 378-sprite set. Read-only on live gen data."""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import extract_v2 as V

# v1 is_bg for old-gate comparison (same constants as live extract.py)
sys.path.insert(0, "krea2-gen")
import extract as V1  # noqa: E402

_root = os.environ.get("KREA_CALIB_ROOT")
if not _root:
    raise SystemExit("set KREA_CALIB_ROOT")
ROOT = Path(_root)
CACHE = ROOT / "cache_cells.pkl"
CONTACTS = ROOT / "contacts"
CONTACTS.mkdir(exist_ok=True)


def font(s):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", s)
    except Exception:
        return ImageFont.load_default()


def load_labels():
    rows = [json.loads(l) for l in (ROOT / "labels.jsonl").open()]
    return {r["id"]: r for r in rows}


def load_jobs_and_sheets():
    items = []
    # eval
    ev_dir = ROOT / "sample_eval/p3_s1500_m1.0"
    jobs = json.load(open(ev_dir / "jobs.json"))
    jobs = [j for j in jobs if "warmup" not in str(j.get("out", ""))]
    for si, j in enumerate(jobs):
        p = ev_dir / Path(j["out"]).name
        if not p.is_file():
            p = ev_dir / f"sheet-{si:03d}.png"
        img = np.asarray(Image.open(p).convert("RGB"))
        g = j["grid"]
        for k, phrase in enumerate(j["items"]):
            r, c = divmod(k, g)
            H, W = img.shape[:2]
            cell = img[round(H * r / g): round(H * (r + 1) / g), round(W * c / g): round(W * (c + 1) / g)]
            items.append({
                "id": f"ev-{si:03d}-{k}",
                "source": "eval",
                "phrase": phrase,
                "img": img, "g": g, "r": r, "c": c,
                "cell": cell, "sheet_path": str(p),
            })
    # gen
    gen_dir = ROOT / "sample_gen"
    for line in open(gen_dir / "jobs.jsonl"):
        j = json.loads(line)
        sid = j["sheet_id"]
        p = gen_dir / "sheets" / "000" / f"sheet-{sid:06d}.png"
        if not p.is_file():
            # search
            hits = list((gen_dir / "sheets").rglob(f"sheet-{sid:06d}.png"))
            p = hits[0] if hits else p
        img = np.asarray(Image.open(p).convert("RGB"))
        g = j.get("grid") or 3
        for k, cdict in enumerate(j["cells"]):
            r, c = divmod(k, g)
            H, W = img.shape[:2]
            cell = img[round(H * r / g): round(H * (r + 1) / g), round(W * c / g): round(W * (c + 1) / g)]
            items.append({
                "id": f"g-{sid:03d}-{k}",
                "source": "gen",
                "phrase": cdict["phrase"],
                "img": img, "g": g, "r": r, "c": c,
                "cell": cell, "sheet_path": str(p),
            })
    return items


def ensure_pooled(items, px=32):
    if CACHE.is_file():
        cached = pickle.loads(CACHE.read_bytes())
        if cached.get("px") == px and len(cached["pooled"]) == len(items):
            print(f"loaded pooled cache {CACHE} n={len(items)}")
            return cached["pooled"]
    print(f"mode_pool {len(items)} cells at {px} ...")
    pooled = {}
    for i, it in enumerate(items):
        sp, pur = V.mode_pool(it["cell"], px)
        pooled[it["id"]] = {"rgb": sp, "purity_map": pur}
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)}")
    CACHE.write_bytes(pickle.dumps({"px": px, "pooled": pooled}))
    print(f"wrote {CACHE}")
    return pooled


def leftover_magenta(rgb, alpha):
    fg = alpha > 0
    if not fg.any():
        return 0
    tint = V.is_edge_fringe(rgb)
    return int((tint & fg).sum())


def unsafe_is_bg(rgb, tol=110):
    """The rejected global widen + loose tint (eats pink/purple subjects)."""
    a = rgb.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    key = np.abs(a - np.array([255, 0, 255])).max(-1) <= tol
    loose = (g < 100) & (r > 120) & (b > 120) & (np.abs(r - b) < 100)
    return key | loose


def unsafe_fns(tol=110):
    def is_bg(rgb):
        return unsafe_is_bg(rgb, tol=tol)

    def score(cell):
        fg = ~unsafe_is_bg(cell, tol=tol)
        frac = fg.mean()
        h, w = fg.shape
        b = max(2, int(min(h, w) * 0.03))
        border = np.concatenate(
            [fg[:b].ravel(), fg[-b:].ravel(), fg[:, :b].ravel(), fg[:, -b:].ravel()]
        ).mean()
        from scipy import ndimage as nd
        lab, n = nd.label(fg)
        main = (nd.sum(fg, lab, range(1, n + 1)).max() / fg.sum()) if n and fg.sum() else 0
        return {"fg": round(float(frac), 3), "border": round(float(border), 3), "main_comp": round(float(main), 3), "n_comp": int(n)}

    def spill(img, g, r, c):
        return V.spill(img, g, r, c, tol=tol, grown=False)  # not used much

    return is_bg, score, spill


def extract_variant(it, pooled, tol, do_fringe, fringe_rounds, is_bg_fn, score_fn, spill_fn, fringe_mode=None):
    """is_bg_fn(rgb) -> mask; score/spill take img/cell with that key."""
    cell = it["cell"]
    img, g, r, c = it["img"], it["g"], it["r"], it["c"]
    s = score_fn(cell)
    s["spill"] = round(spill_fn(img, g, r, c), 3)
    sp = pooled[it["id"]]["rgb"]
    pur = pooled[it["id"]]["purity_map"]
    alpha = np.where(is_bg_fn(sp), 0, 255).astype(np.uint8)
    fringe_dropped = 0
    if do_fringe:
        if fringe_mode == "unsafe":
            # loose edge tint + key 130, 2 rounds (the destructive variant)
            from scipy import ndimage as nd
            a = sp.astype(int)
            r_, g_, b_ = a[..., 0], a[..., 1], a[..., 2]
            tint = (np.abs(a - np.array([255, 0, 255])).max(-1) <= 130) | (
                (g_ < 100) & (r_ > 120) & (b_ > 120) & (np.abs(r_ - b_) < 100)
            )
            alpha = alpha.copy()
            dropped = 0
            for _ in range(fringe_rounds):
                fg = alpha > 0
                drop = fg & nd.binary_dilation(~fg) & tint
                dropped += int(drop.sum())
                alpha[drop] = 0
            fringe_dropped = dropped
        else:
            alpha, fringe_dropped = V.strip_edge_fringe(sp, alpha, rounds=fringe_rounds)
    alpha = V.drop_stray_pixels(alpha)
    # palette clean does not change alpha
    fgb = alpha > 0
    pm = float(pur[fgb].mean()) if fgb.any() else 0.0
    return {
        "score": s,
        "purity": pm,
        "alpha": alpha,
        "rgb": sp,
        "fringe_dropped": fringe_dropped,
        "leftover_magenta": leftover_magenta(sp, alpha),
        "edge_touch": V.touches_32_edge(alpha),
        "fg32": int(fgb.sum()),
    }


def v1_fns():
    def is_bg(rgb):
        return V1.is_bg(rgb)

    def score(cell):
        return V1.score_cell(cell)

    def spill(img, g, r, c):
        return V1.spill(img, g, r, c)

    return is_bg, score, spill


def v2_fns(tol):
    def is_bg(rgb):
        return V.is_bg(rgb, tol=tol)

    def score(cell):
        return V.score_cell(cell, tol=tol)

    def spill(img, g, r, c):
        return V.spill(img, g, r, c, tol=tol)

    return is_bg, score, spill


def apply_gate(ex, thr):
    s = dict(ex["score"])
    return V.bin_for(s, ex["purity"], thr), V.reject_reasons(s, ex["purity"], thr)


OLD_THR = {
    "fg_min": 0.04,
    "fg_max": 0.90,
    "spill_max": 0.08,
    "main_comp_min": 0.85,
    "purity_min": 0.75,
}


def metrics(rows, labels):
    """rows: list of {id, bin}. good kept = label good and bin==ok.
    Visual labels are judgements."""
    tp = fp = tn = fn = 0  # positive = keep (ok)
    for r in rows:
        lab = labels[r["id"]]["label"]
        keep = r["bin"] == "ok"
        if lab == "good" and keep:
            tp += 1
        elif lab == "bad" and keep:
            fp += 1
        elif lab == "bad" and not keep:
            tn += 1
        else:
            fn += 1
    n_good = tp + fn
    n_bad = tn + fp
    prec = tp / max(tp + fp, 1)
    rec_good = tp / max(n_good, 1)
    rec_bad = tn / max(n_bad, 1)
    return {
        "n": len(rows),
        "n_good": n_good,
        "n_bad": n_bad,
        "good_kept": tp,
        "good_rejected": fn,
        "bad_kept": fp,
        "bad_rejected": tn,
        "precision_ok_are_good": round(prec, 4),
        "recall_good_kept": round(rec_good, 4),
        "recall_bad_rejected": round(rec_bad, 4),
        "ok": sum(r["bin"] == "ok" for r in rows),
        "purity_fail": sum(r["bin"] == "purity_fail" for r in rows),
        "rejected": sum(r["bin"] == "rejected" for r in rows),
    }


def run():
    labels = load_labels()
    items = load_jobs_and_sheets()
    assert len(items) == len(labels), (len(items), len(labels))
    pooled = ensure_pooled(items)
    id2it = {it["id"]: it for it in items}

    # --- extract variants ---
    variants = {
        "v1_no_edge_strip": {"tol": 90, "fringe": False, "fns": "v1"},
        "v1_plus_safe_edge_strip": {"tol": 90, "fringe": True, "fns": "v1"},
        "v2_grown_no_strip": {"tol": 90, "fringe": False, "fns": "v2"},
        "v2_grown_safe_strip": {"tol": 90, "fringe": True, "fns": "v2"},
        # unsafe baseline (global tol=110 + loose tint edge), for the effect writeup
        "unsafe_global_tol110_loose_strip": {"tol": 110, "fringe": True, "fns": "unsafe"},
    }
    extracted = {}
    for name, cfg in variants.items():
        if cfg["fns"] == "v1":
            is_bg, score, spill = v1_fns()
            fringe_fn = None  # use V.strip_edge_fringe (now safe, key<=100)
        elif cfg["fns"] == "unsafe":
            is_bg, score, spill = unsafe_fns(cfg["tol"])
            fringe_fn = "unsafe"
        else:
            is_bg, score, spill = v2_fns(cfg["tol"])
            fringe_fn = None
        extracted[name] = {}
        for it in items:
            extracted[name][it["id"]] = extract_variant(
                it, pooled, cfg["tol"], cfg["fringe"], 2, is_bg, score, spill,
                fringe_mode=fringe_fn,
            )
        print(f"extracted {name}")

    # fringe effect vs v1_no_edge_strip
    focus = [
        "g-001-4", "g-001-8", "g-021-0", "g-006-6", "g-000-0", "g-009-2",
        "g-028-5", "g-018-6", "g-020-8", "g-020-7", "g-004-8", "g-004-7",
        "g-024-5", "ev-005-8",
    ]
    fringe_report = {"variants": {}, "focus": {}}
    base = "v1_no_edge_strip"
    for name, byid in extracted.items():
        drops = [ex["fringe_dropped"] for ex in byid.values()]
        left = [ex["leftover_magenta"] for ex in byid.values()]
        fg = [ex["fg32"] for ex in byid.values()]
        base_fg = [extracted[base][i]["fg32"] for i in byid]
        dfg = [a - b for a, b in zip(fg, base_fg)]
        fringe_report["variants"][name] = {
            "sprites_with_edge_strip": int(sum(d > 0 for d in drops)),
            "fringe_dropped_mean": round(float(np.mean(drops)), 3),
            "fringe_dropped_max": int(max(drops)),
            "leftover_magenta_mean": round(float(np.mean(left)), 3),
            "leftover_magenta_sum": int(sum(left)),
            "fg32_mean": round(float(np.mean(fg)), 2),
            "fg32_delta_vs_v1_mean": round(float(np.mean(dfg)), 3),
            "sprites_fg_shrunk_vs_v1": int(sum(d < 0 for d in dfg)),
            "sprites_fg_grew_vs_v1": int(sum(d > 0 for d in dfg)),
        }
    for i in focus:
        if i not in labels:
            continue
        fringe_report["focus"][i] = {
            "phrase": labels[i]["phrase"],
            "label": labels[i]["label"],
            "variants": {
                n: {
                    "fg32": extracted[n][i]["fg32"],
                    "fringe_dropped": extracted[n][i]["fringe_dropped"],
                    "leftover_magenta": extracted[n][i]["leftover_magenta"],
                    "purity": round(extracted[n][i]["purity"], 3),
                    "spill": extracted[n][i]["score"]["spill"],
                    "fg": extracted[n][i]["score"]["fg"],
                    "main_comp": extracted[n][i]["score"]["main_comp"],
                }
                for n in extracted
            },
        }
    (ROOT / "fringe_effect.json").write_text(json.dumps(fringe_report, indent=2))

    # --- threshold sweep on chosen extract (v2_tol110_strip) ---
    chosen_extract_name = "v2_grown_safe_strip"
    exs = extracted[chosen_extract_name]

    # also score old gate on v1 extract
    old_rows = []
    for it in items:
        ex = extracted["v1_no_edge_strip"][it["id"]]
        b, reasons = apply_gate(ex, OLD_THR)
        old_rows.append({"id": it["id"], "bin": b, "reasons": reasons, "ex": ex})
    old_m = metrics(old_rows, labels)

    grid = []
    purity_opts = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    fg_max_opts = [0.90, 0.95, 0.98, 0.995, 1.00]
    spill_opts = [0.08, 0.12, 0.16, 0.20, 0.25, 1.0]
    main_opts = [0.50, 0.55, 0.60, 0.70, 0.85]
    fg_min_opts = [0.04]

    best = None
    for pmin in purity_opts:
        for fgmax in fg_max_opts:
            for spmax in spill_opts:
                for mmin in main_opts:
                    for fgmin in fg_min_opts:
                        thr = {
                            "fg_min": fgmin,
                            "fg_max": fgmax,
                            "spill_max": spmax,
                            "main_comp_min": mmin,
                            "purity_min": pmin,
                        }
                        rows = []
                        for it in items:
                            b, reasons = apply_gate(exs[it["id"]], thr)
                            rows.append({"id": it["id"], "bin": b, "reasons": reasons})
                        m = metrics(rows, labels)
                        # Prefer keeping goods, then rejecting bads, then fewer hard rejects of goods.
                        # Tie-break: simpler/looser gates that still match (fg_max=1, moderate spill).
                        key = (
                            m["recall_good_kept"],
                            m["recall_bad_rejected"],
                            m["precision_ok_are_good"],
                            # prefer not using purity_fail as a dump: more ok is fine
                            m["ok"] / max(m["n"], 1),
                            # prefer looser fg_max and moderate spill (less overfit)
                            fgmax,
                            -abs(spmax - 0.20),
                            -pmin,
                        )
                        rec = {"thr": thr, "metrics": m, "key": [float(x) for x in key]}
                        grid.append(rec)
                        if best is None or key > tuple(best["key"]):
                            best = rec

    # Among candidates that keep ALL goods (or max recall_good), report the Pareto of recall_bad
    max_rg = max(g["metrics"]["recall_good_kept"] for g in grid)
    keep_all = [g for g in grid if g["metrics"]["recall_good_kept"] == max_rg]
    keep_all.sort(key=lambda g: (-g["metrics"]["recall_bad_rejected"], -g["metrics"]["precision_ok_are_good"]))

    # Chosen policy (not raw argmax if we want a readable gate):
    # After looking at keep_all[0], also consider a rounded default.
    chosen = keep_all[0]["thr"]
    # If several tols of the same metrics, prefer the documented rounded set
    # if it is in the keep-all / max-recall_good set.
    preferred = {
        "fg_min": 0.04,
        "fg_max": 1.00,
        "spill_max": 0.08,
        "main_comp_min": 0.55,
        "purity_min": 0.35,
    }
    pref_rows = []
    for it in items:
        b, reasons = apply_gate(exs[it["id"]], preferred)
        pref_rows.append({"id": it["id"], "bin": b, "reasons": reasons})
    pref_m = metrics(pref_rows, labels)

    # Use preferred if it matches max good-recall and is close on bad-recall
    if pref_m["recall_good_kept"] == max_rg:
        chosen = preferred
        chosen_m = pref_m
        chosen_rows = pref_rows
    else:
        chosen_m = keep_all[0]["metrics"]
        chosen = keep_all[0]["thr"]
        chosen_rows = []
        for it in items:
            b, reasons = apply_gate(exs[it["id"]], chosen)
            chosen_rows.append({"id": it["id"], "bin": b, "reasons": reasons})

    # Attach per-id results
    per = []
    for it in items:
        lab = labels[it["id"]]
        old = next(r for r in old_rows if r["id"] == it["id"])
        new = next(r for r in chosen_rows if r["id"] == it["id"])
        ex = exs[it["id"]]
        ex_old = extracted["v1_no_edge_strip"][it["id"]]
        per.append({
            "id": it["id"],
            "source": it["source"],
            "phrase": it["phrase"],
            "label": lab["label"],
            "label_reasons": lab["reasons"],
            "label_note": lab["note"],
            "labeler": "assistant_visual_judgement",
            "old_bin": old["bin"],
            "old_reasons": old["reasons"],
            "new_bin": new["bin"],
            "new_reasons": new["reasons"],
            "purity_v2": round(ex["purity"], 3),
            "purity_v1": round(ex_old["purity"], 3),
            "score_v2": ex["score"],
            "score_v1": ex_old["score"],
            "fringe_dropped": ex["fringe_dropped"],
            "edge_touch": ex["edge_touch"],
            "leftover_magenta_v2": ex["leftover_magenta"],
            "leftover_magenta_v1": ex_old["leftover_magenta"],
        })

    sweep_out = {
        "n_grid": len(grid),
        "old_thresholds": OLD_THR,
        "old_metrics": old_m,
        "max_recall_good": max_rg,
        "best_raw": {"thr": keep_all[0]["thr"], "metrics": keep_all[0]["metrics"]},
        "chosen": chosen,
        "chosen_metrics": chosen_m,
        "extract_variant": chosen_extract_name,
        "note": "Labels are assistant visual judgements, not ground truth. Many bad sprites are semantically wrong but structurally clean, so size/purity/spill/main_comp cannot reject them without also dropping goods.",
        # compact grid extrema
        "pareto_keep_all_good": [
            {"thr": g["thr"], "metrics": g["metrics"]}
            for g in keep_all[:8]
        ],
    }
    (ROOT / "sweep.json").write_text(json.dumps(sweep_out, indent=2))
    with (ROOT / "per_sprite.jsonl").open("w") as f:
        for r in per:
            f.write(json.dumps(r) + "\n")

    print("OLD", old_m)
    print("CHOSEN", chosen, chosen_m)
    print("best_raw", keep_all[0]["thr"], keep_all[0]["metrics"])

    # --- contacts ---
    def sprite_img(ex, scale=6):
        rgba = np.dstack([ex["rgb"], ex["alpha"]])
        im = Image.fromarray(rgba, "RGBA")
        ck = Image.new("RGBA", (32 * scale, 32 * scale), (0, 0, 0, 0))
        # checker
        bg = Image.new("RGBA", (32 * scale, 32 * scale))
        d = ImageDraw.Draw(bg)
        for y in range(0, 32 * scale, 8):
            for x in range(0, 32 * scale, 8):
                fill = (210, 210, 210, 255) if ((x // 8 + y // 8) % 2) == 0 else (170, 170, 170, 255)
                d.rectangle([x, y, x + 7, y + 7], fill=fill)
        sp = im.resize((32 * scale, 32 * scale), Image.Resampling.NEAREST)
        return Image.alpha_composite(bg, sp)

    def contact(ids, path, title, get_caption, cols=9, scale=5, cap_h=54):
        if not ids:
            Image.new("RGB", (64, 40), (255, 255, 255)).save(path)
            return
        pad, gap = 8, 8
        cell_w = 32 * scale
        cell_h = 32 * scale + cap_h
        rows_n = (len(ids) + cols - 1) // cols
        W = pad * 2 + cols * cell_w + (cols - 1) * gap
        H = 36 + pad + rows_n * cell_h + (rows_n - 1) * gap
        im = Image.new("RGB", (W, H), (245, 245, 245))
        d = ImageDraw.Draw(im)
        d.text((pad, 8), title, fill=(20, 20, 20), font=font(16))
        for i, sid in enumerate(ids):
            rr, cc = divmod(i, cols)
            x = pad + cc * (cell_w + gap)
            y = 36 + rr * (cell_h + gap)
            ex = exs[sid]
            im.paste(sprite_img(ex, scale), (x, y))
            rec = next(p for p in per if p["id"] == sid)
            cap = get_caption(rec)
            # wrap
            d.rectangle([x, y + 32 * scale, x + cell_w, y + cell_h], fill=(255, 255, 255))
            # color bar by judgement / bin
            col = (40, 140, 60) if rec["label"] == "good" else (180, 40, 40)
            d.rectangle([x, y + 32 * scale, x + 6, y + cell_h], fill=col)
            yy = y + 32 * scale + 2
            for line in cap.split("\n")[:5]:
                d.text((x + 8, yy), line[:42], fill=(20, 20, 20), font=font(10))
                yy += 10
        im.save(path)
        print("wrote", path, "n=", len(ids))

    new_ok = [p["id"] for p in per if p["new_bin"] == "ok"]
    new_rej = [p["id"] for p in per if p["new_bin"] != "ok"]
    recovered = [p["id"] for p in per if p["old_bin"] != "ok" and p["new_bin"] == "ok"]
    still_bad_kept = [p["id"] for p in per if p["label"] == "bad" and p["new_bin"] == "ok"]
    goods_dropped = [p["id"] for p in per if p["label"] == "good" and p["new_bin"] != "ok"]
    bads = [p["id"] for p in per if p["label"] == "bad"]

    def cap_full(rec):
        lab = rec["label"]
        why = ",".join(rec["label_reasons"]) or "good"
        gate = rec["new_bin"]
        rs = ",".join(rec["new_reasons"]) or "pass"
        return f"{rec['id']} {lab}/{why}\n{rec['phrase'][:40]}\n{gate}: {rs}\nold={rec['old_bin']}"

    contact(
        new_rej,
        CONTACTS / "contact_new_rejected.png",
        "NEW GATE rejected / purity_fail  |  green bar=judged GOOD, red bar=judged BAD (assistant judgement)",
        cap_full,
        cols=min(8, max(1, len(new_rej))),
        scale=5,
        cap_h=58,
    )
    contact(
        bads,
        CONTACTS / "contact_labelled_bad.png",
        "LABELLED BAD (assistant visual judgement)  |  caption shows new-gate decision",
        cap_full,
        cols=7,
        scale=6,
        cap_h=58,
    )
    contact(
        recovered,
        CONTACTS / "contact_recovered_old_reject_now_ok.png",
        "RECOVERED: old gate not-ok, new gate ok  |  bar=visual judgement",
        cap_full,
        cols=9,
        scale=4,
        cap_h=54,
    )
    contact(
        still_bad_kept,
        CONTACTS / "contact_new_ok_but_labelled_bad.png",
        "NEW GATE ok but LABELLED BAD (false accepts; mostly semantic — assistant judgement)",
        cap_full,
        cols=min(7, max(1, len(still_bad_kept))),
        scale=6,
        cap_h=58,
    )
    if goods_dropped:
        contact(
            goods_dropped,
            CONTACTS / "contact_new_rejected_but_labelled_good.png",
            "NEW GATE rejected but LABELLED GOOD (false rejects) — assistant judgement",
            cap_full,
            cols=8,
            scale=5,
            cap_h=58,
        )

    # All new-ok contact would be huge; do a compact 4x sheet in pages of 90
    def cap_ok(rec):
        return f"{rec['id']} {rec['phrase'][:36]}\np={rec['purity_v2']} old={rec['old_bin']}"

    # page all ok at 4x, 15 cols
    contact(
        new_ok,
        CONTACTS / "contact_new_ok.png",
        "NEW GATE ok  |  green bar=judged GOOD, red bar=judged BAD (assistant judgement)",
        cap_ok,
        cols=15,
        scale=3,
        cap_h=32,
    )

    # before/after fringe focus contact
    def fringe_pair_sheet(path):
        scale = 6
        names = [i for i in focus if i in exs]
        cols = 4  # v1 | v1+strip | v2 no | v2 strip   per sprite as a row? or 2-col pairs
        # each sprite: 2 tiles (v1, v2_strip)
        pad, gap = 8, 6
        cell = 32 * scale
        cap_h = 40
        W = pad * 2 + 2 * cell + gap + 220
        H = 40 + len(names) * (cell + cap_h + gap)
        im = Image.new("RGB", (W, H), (245, 245, 245))
        d = ImageDraw.Draw(im)
        d.text((pad, 8), "Fringe cleanup: left=v1 key only, right=v2 grown-score + safe edge strip (key<=100)", fill=(20, 20, 20), font=font(14))
        for i, sid in enumerate(names):
            y = 40 + i * (cell + cap_h + gap)
            im.paste(sprite_img(extracted["v1_no_edge_strip"][sid], scale), (pad, y))
            im.paste(sprite_img(extracted["v2_grown_safe_strip"][sid], scale), (pad + cell + gap, y))
            rec = labels[sid]
            a = extracted["v1_no_edge_strip"][sid]
            b = extracted["v2_grown_safe_strip"][sid]
            txt = (
                f"{sid}  [{rec['label']}] {rec['phrase'][:40]}\n"
                f"v1 fg32={a['fg32']} left_mag={a['leftover_magenta']} spill={a['score']['spill']}\n"
                f"v2 fg32={b['fg32']} drop={b['fringe_dropped']} left_mag={b['leftover_magenta']} spill={b['score']['spill']}"
            )
            yy = y
            for line in txt.split("\n"):
                d.text((pad + 2 * cell + gap + 8, yy), line, fill=(20, 20, 20), font=font(11))
                yy += 14
        im.save(path)
        print("wrote", path)

    fringe_pair_sheet(CONTACTS / "contact_fringe_before_after.png")

    # Save RGBA of chosen extract for dry-run comparison later? not needed.

    return {
        "old_m": old_m,
        "chosen": chosen,
        "chosen_m": chosen_m,
        "per": per,
        "extracted": extracted,
        "items": items,
        "labels": labels,
        "fringe_report": fringe_report,
        "old_rows": old_rows,
        "chosen_extract_name": chosen_extract_name,
    }


if __name__ == "__main__":
    run()
