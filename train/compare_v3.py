"""Collect Pixie v3 results into one JSON + Markdown table (equal images seen)."""
import json, sys
from pathlib import Path

RUNS = Path(sys.argv[1]); V3 = Path(sys.argv[2])
names = {"v2ft": "v3-v2ft-restored", "pilot_a_pix256": "v3-pilot-a-pix256", "pilot_b_p50": "v3-pilot-b-p50"}
KEYS = ["probe_nll", "probe_nll_uncond", "iid_nll", "iid_rare_lt1k", "iid_mid_1k_5k", "iid_common_gt5k",
        "probe_rare_lt1k", "probe_mid_1k_5k", "probe_common_gt5k"]


def rows(run):
    p = RUNS / run / "metrics.jsonl"
    if not p.exists():
        return [], {}, {}
    ev, samp, start = [], {}, {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") == "eval":
            ev.append(r)
        elif r.get("event") == "samples":
            samp = r
        elif r.get("event") == "start":
            start = r
    return ev, samp, start


out = {"runs": {}}
md = ["# Pixie v3 results", ""]
base = V3 / "results" / "baselines.json"
if base.exists():
    b = json.loads(base.read_text())
    out["baselines"] = b
    md += ["## Baselines on the v3 probe/iid splits (fp32)", "", "| model | probe | iid | iid (restored subjects) | iid rare | iid mid | iid common | slot NLL (TL,TR,BL,BR, iid) |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for k, r in b.items():
        if k.startswith("_"):
            continue
        md.append(f"| {k} | {r.get('probe_nll', 0):.4f} | {r.get('iid_nll', 0):.4f} | {r.get('iidrestored_nll', 0):.4f} | {r.get('iid_rare_lt1k', 0):.4f} | "
                  f"{r.get('iid_mid_1k_5k', 0):.4f} | {r.get('iid_common_gt5k', 0):.4f} | {[round(x, 3) for x in r.get('iid_slot_nll', [])]} |")
    md.append("")
for key, run in names.items():
    ev, samp, start = rows(run)
    summ_p = RUNS / run / "train_summary.json"
    out["runs"][key] = {"run": run, "start": start, "evals": ev, "samples": samp,
                        "summary": json.loads(summ_p.read_text()) if summ_p.exists() else None}
    if not ev:
        continue
    md += [f"## {key} (`{run}`, {start.get('params', 0) / 1e6:.2f}M params)", "",
           "| images | probe | probe uncond | iid | iid restored | iid rare | iid mid | iid common | iid slot NLL |", "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in ev:
        md.append(f"| {r['images'] / 1e6:.2f}M | {r['probe_nll']:.4f} | {r.get('probe_nll_uncond', 0):.4f} | {r['iid_nll']:.4f} | {r.get('iidrestored_nll', 0):.4f} | "
                  f"{r.get('iid_rare_lt1k', 0):.4f} | {r.get('iid_mid_1k_5k', 0):.4f} | {r.get('iid_common_gt5k', 0):.4f} | "
                  f"{[round(x, 3) for x in r.get('iid_slot_nll', [])]} |")
    if samp:
        md.append(f"\nBoundary artifact ratio (gen/real): **{samp.get('boundary_artifact_ratio', 0):.3f}** "
                  f"(gen {samp.get('boundary_ratio_gen', 0):.3f}, real {samp.get('boundary_ratio_real', 0):.3f}, n={samp.get('n_samples')})")
    s = out["runs"][key]["summary"]
    if s:
        md.append(f"\nWall {s['wall_hours']} h, train {s['train_hours']} h, stop: {s['stop_reason']}, peak {s['peak_allocated_mib']} MiB")
    md.append("")
a = {r["images"]: r for r in out["runs"].get("pilot_a_pix256", {}).get("evals", []) if r.get("tag") != "final"}
bb = {r["images"]: r for r in out["runs"].get("pilot_b_p50", {}).get("evals", []) if r.get("tag") != "final"}
common = sorted(set(a) & set(bb))
if common:
    md += ["## Pilot (a) pixel vs (b) P50 at equal images seen (iid / probe; negative delta = P50 better)", "",
           "| images | a iid | b iid | Δ iid | a probe | b probe | Δ probe | a iid rare | b iid rare |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for im in common:
        x, y = a[im], bb[im]
        md.append(f"| {im / 1e6:.2f}M | {x['iid_nll']:.4f} | {y['iid_nll']:.4f} | {y['iid_nll'] - x['iid_nll']:+.4f} | "
                  f"{x['probe_nll']:.4f} | {y['probe_nll']:.4f} | {y['probe_nll'] - x['probe_nll']:+.4f} | "
                  f"{x.get('iid_rare_lt1k', 0):.4f} | {y.get('iid_rare_lt1k', 0):.4f} |")
(V3 / "results").mkdir(exist_ok=True)
(V3 / "results" / "results.json").write_text(json.dumps(out, indent=1))
(V3 / "results" / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
