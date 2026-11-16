#!/usr/bin/env python3
"""Read-only spot check of extract_v2 vs original run verdicts. Writes only to $R/spotcheck."""
import json, random, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

R = Path(os.path.expandvars("$DATA_ROOT"))
OUT = R / "spotcheck"; OUT.mkdir(exist_ok=True)
SRC = {"3080ti": Path(os.path.expandvars("$GEN_ROOT/krea2-gen-3080ti/data/labels.jsonl")),
       "3060ti": R / "raw/3060ti/labels.jsonl", "5070ti": R / "raw/5070ti/labels.jsonl"}
rows = [json.loads(l) for l in open(R / "manifest.jsonl")]
orig = {}
for s, p in SRC.items():
    for l in open(p):
        j = json.loads(l); orig[(s, j["sheet_id"], j["cell"])] = j
def obin(r):
    o = orig[(r["source"], r["sheet_id"], r["cell"])]
    return o.get("bin") or o["file"].split("/")[0]
for r in rows:
    r["obin"] = obin(r); r["o"] = orig[(r["source"], r["sheet_id"], r["cell"])]

# per-sprite pixel stats on the extracted 32x32
def pix_stats(path):
    a = np.asarray(Image.open(R / path).convert("RGBA")).astype(int)
    al = a[..., 3] > 0; rgb = a[..., :3]
    r_, g_, b_ = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mag = al & (g_ < 110) & (r_ > 140) & (b_ > 140) & (np.abs(r_ - b_) < 90)
    edge = np.zeros_like(al); edge[0] = edge[-1] = True; edge[:, 0] = edge[:, -1] = True
    from scipy import ndimage
    border_ring = al & ndimage.binary_dilation(~al)
    lab, n = ndimage.label(al)
    sizes = ndimage.sum(al, lab, range(1, n + 1)) if n else []
    return {"opaque": int(al.sum()), "edge_px": int((al & edge).sum()),
            "mag_px": int(mag.sum()), "mag_outline_px": int((mag & border_ring).sum()),
            "n_comp32": int(n), "small_comp32": int(sum(1 for z in sizes if z <= 3))}
for r in rows:
    if r["pass_extract_v2"]:
        r.update(pix_stats(r["sprite_path"]))

out = {}
# 1. groups
grp = defaultdict(Counter)
for r in rows:
    grp[r["source"]][f"orig_{r['obin']} -> v2_{r['bin']}"] += 1
out["groups"] = {s: dict(sorted(c.items())) for s, c in grp.items()}

# 3. why orig rejected (from original metrics where available)
def old_fail(m):
    f = []
    if m.get("purity") is None: return ["no_metrics"]
    if m["fg"] < 0.04: f.append("fg<0.04")
    if m["fg"] > 0.90: f.append("fg>0.90")
    if m["spill"] > 0.08: f.append("spill>0.08")
    if m["main_comp"] < 0.85: f.append("main_comp<0.85")
    if m["purity"] < 0.75: f.append("purity<0.75")
    return f
why = defaultdict(Counter); why_combo = defaultdict(Counter)
for r in rows:
    if r["obin"] != "ok" and r["pass_extract_v2"]:
        fl = old_fail(r["o"])
        for f in fl: why[r["source"]][f] += 1
        why_combo[r["source"]]["+".join(fl) or "none"] += 1
out["key_group_old_fail_reasons"] = {s: dict(c) for s, c in why.items()}
out["key_group_old_fail_combos"] = {s: dict(c.most_common(8)) for s, c in why_combo.items()}
def pct(v): v = np.array(v, float); return {q: round(float(np.percentile(v, q)), 3) for q in (5, 25, 50, 75, 95)} if len(v) else None
dist = {}
for s in ("3080ti", "3060ti", "5070ti"):
    rs = [r for r in rows if r["source"] == s]
    d = {}
    for m in ("purity", "fg", "main_comp", "spill"):
        d[m] = {"orig": pct([r["o"][m] for r in rs if r["o"].get(m) is not None]), "v2": pct([r[m] for r in rs])}
    dist[s] = d
out["metric_percentiles_all_cells"] = dist
# old gates applied to v2 metrics, and v2 gates applied to orig metrics
def gate(m, fgmax, mc, pur):
    return 0.04 <= m["fg"] <= fgmax and m["spill"] <= 0.08 and m["main_comp"] >= mc and m["purity"] >= pur
cross = {}
for s in ("3080ti", "3060ti"):
    rs = [r for r in rows if r["source"] == s]
    cross[s] = {"old_gates_on_orig_metrics": sum(gate(r["o"], .9, .85, .75) for r in rs),
                "old_gates_on_v2_metrics": sum(gate(r, .9, .85, .75) for r in rs),
                "v2_gates_on_orig_metrics": sum(gate(r["o"], 1.0, .55, .35) for r in rs),
                "v2_gates_on_v2_metrics": sum(gate(r, 1.0, .55, .35) for r in rs),
                "old_gates_but_purity>=0.35_on_orig": sum(gate(r["o"], .9, .85, .35) for r in rs)}
out["gate_cross_check"] = cross

# 4. options on the FINAL set
fin = [r for r in rows if r["final"]]
def cnt(f): 
    c = Counter(r["source"] for r in fin if f(r)); return {"total": sum(c.values()), **dict(c)}
opts = {
 "A_current_final": cnt(lambda r: True),
 "B_purity>=0.50": cnt(lambda r: r["purity"] >= .50),
 "B_purity>=0.60": cnt(lambda r: r["purity"] >= .60),
 "B_purity>=0.75(old)": cnt(lambda r: r["purity"] >= .75),
 "C_main_comp>=0.85": cnt(lambda r: r["main_comp"] >= .85),
 "C_main_comp>=0.95": cnt(lambda r: r["main_comp"] >= .95),
 "D_fg<=0.90": cnt(lambda r: r["fg"] <= .90),
 "E_drop_all_edge_touch": cnt(lambda r: not r["edge_touch"]),
 "E_edge_px<=4": cnt(lambda r: r["edge_px"] <= 4),
 "E_edge_px<=8": cnt(lambda r: r["edge_px"] <= 8),
 "E_edge_px<=16": cnt(lambda r: r["edge_px"] <= 16),
 "F_no_magenta_px": cnt(lambda r: r["mag_px"] == 0),
 "F_magenta_px<=2": cnt(lambda r: r["mag_px"] <= 2),
 "G_no_tiny_islands(<=3px comps)": cnt(lambda r: r["small_comp32"] == 0),
 "H_combo_purity>=0.5_mc>=0.85_edge_px<=8": cnt(lambda r: r["purity"] >= .5 and r["main_comp"] >= .85 and r["edge_px"] <= 8),
}
out["final_counts_by_option"] = opts
out["final_stats"] = {"sprites_with_magenta_px": sum(r["mag_px"] > 0 for r in fin),
                      "magenta_px_total": sum(r["mag_px"] for r in fin),
                      "magenta_on_outline_sprites": sum(r["mag_outline_px"] > 0 for r in fin),
                      "edge_px_percentiles_edge_touch": pct([r["edge_px"] for r in fin if r["edge_touch"]]),
                      "with_tiny_islands": sum(r["small_comp32"] > 0 for r in fin)}

# 2. contact sheets
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
except Exception:
    font = ImageFont.load_default()
def sheet(sample, name, cols=10, per=50):
    S, cap = 128, 30
    paths = []
    for k in range(0, len(sample), per):
        chunk = sample[k:k + per]; rows_n = (len(chunk) + cols - 1) // cols
        im = Image.new("RGB", (cols * (S + 6) + 6, rows_n * (S + cap + 6) + 6), (36, 36, 40))
        d = ImageDraw.Draw(im)
        for i, r in enumerate(chunk):
            x, y = 6 + (i % cols) * (S + 6), 6 + (i // cols) * (S + cap + 6)
            sp = Image.open(R / r["sprite_path"]).convert("RGBA").resize((S, S), Image.NEAREST)
            bg = Image.new("RGBA", (S, S), (200, 200, 200, 255))
            for yy in range(0, S, 16):
                for xx in range(0, S, 16):
                    if (xx // 16 + yy // 16) % 2: bg.paste((232, 232, 232, 255), (xx, yy, xx + 16, yy + 16))
            bg.alpha_composite(sp); im.paste(bg.convert("RGB"), (x, y))
            d.text((x + 2, y + 1), str(k + i), fill=(255, 0, 0), font=font)
            c = r["caption"][:24]
            d.text((x, y + S + 1), c, fill=(230, 230, 230), font=font)
            d.text((x, y + S + 14), f"{r['source']} p{r['purity']:.2f} m{r['main_comp']:.2f} e{r.get('edge_px',0)}", fill=(150, 150, 160), font=font)
        p = OUT / f"{name}_{k // per + 1}.png"; im.save(p); paths.append(str(p))
    with open(OUT / f"{name}.jsonl", "w") as f:
        for i, r in enumerate(sample):
            f.write(json.dumps({"idx": i, "id": r["id"], "caption": r["caption"], "source": r["source"], "obin": r["obin"],
                                "purity": r["purity"], "main_comp": r["main_comp"], "fg": r["fg"], "edge_px": r.get("edge_px"),
                                "mag_px": r.get("mag_px"), "sprite_path": r["sprite_path"]}) + "\n")
    return paths
random.seed(1115)
key = [r for r in rows if r["obin"] != "ok" and r["pass_extract_v2"]]
both = [r for r in rows if r["obin"] == "ok" and r["pass_extract_v2"]]
edge = [r for r in fin if r["edge_touch"]]
out["sample_pools"] = {"key_group": len(key), "both_passed": len(both), "edge_touch_final": len(edge)}
out["sheets"] = {"key": sheet(random.sample(key, 200), "key_origfail_v2pass"),
                 "both": sheet(random.sample(both, 100), "both_passed"),
                 "edge": sheet(random.sample(edge, 100), "edge_touch_final")}
# candidate list files (not applied)
for name, f in [("cand_H_purity0.5_mc0.85_edge8", lambda r: r["purity"] >= .5 and r["main_comp"] >= .85 and r["edge_px"] <= 8)]:
    with open(OUT / f"{name}.txt", "w") as fh:
        for r in fin:
            if f(r): fh.write(r["final_path"] + "\n")
json.dump(out, open(OUT / "spotcheck_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "sheets"}, indent=1))
