#!/usr/bin/env python3
"""Re-extract 3x3 @ 576 raw sheets with extract_v2.

Does not read or write a running generation's output directories unless
you pass those paths yourself. For a later full pass, point --src at the
kept raw sheets and --out at a NEW folder.

Examples:
  # dry-run on a calibration sample (set KREA_CALIB_ROOT to that directory):
  python3 reextract.py --sample --out "$KREA_CALIB_ROOT/dryrun_sample"

  # later, all raw sheets into a new folder:
  python3 reextract.py \\
      --jobs "$DATA_ROOT/krea2-gen/data/jobs.jsonl" \\
      --src  "$DATA_ROOT/krea2-gen/data/sheets" \\
      --out  "$DATA_ROOT/krea2-gen/reextract_v2" \\
      --px 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import extract_v2 as V

def _calib(rel):
    root = os.environ.get("KREA_CALIB_ROOT")
    if not root:
        raise SystemExit("set KREA_CALIB_ROOT to the calibration directory")
    return Path(root) / rel

def iter_job_groups(sample=False, jobs=None, src=None):
    if sample:
        yield ("eval", _calib("sample_eval/p3_s1500_m1.0/jobs.json"), _calib("sample_eval/p3_s1500_m1.0"))
        yield ("gen", _calib("sample_gen/jobs.jsonl"), _calib("sample_gen/sheets"))
        return
    if not jobs:
        raise SystemExit("pass --jobs or --sample")
    yield ("run", Path(jobs), Path(src) if src else None)


def resolve_src_sheet(job, src):
    # extract_v2.resolve_sheet wants a dir; also search one level of shards
    try:
        return V.resolve_sheet(job, src)
    except FileNotFoundError:
        if not src:
            raise
        src = Path(src)
        name = Path(job["sheet"]).name if job.get("sheet") else None
        sid = job.get("sheet_id")
        candidates = []
        if name:
            candidates += list(src.rglob(name))
        if isinstance(sid, int):
            for pat in (f"sheet-{sid:06d}.png", f"sheet-{sid:03d}.png", f"sheet-{sid}.png"):
                candidates += list(src.rglob(pat))
        for c in candidates:
            if c.is_file():
                return c
        raise


def extract_group(tag, jobs_path, src, out_root, px, fringe, key_tol):
    dest = out_root / tag
    for sub in ("ok", "purity_fail", "rejected"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    jobs = V.load_jobs(jobs_path)
    rows = []
    missing = []
    for j in jobs:
        try:
            p = resolve_src_sheet(j, src)
        except FileNotFoundError as e:
            missing.append({"sheet_id": j.get("sheet_id"), "sheet": j.get("sheet"), "error": str(e)})
            continue
        img = np.asarray(Image.open(p).convert("RGB"))
        g = j["grid"]
        for k, phrase in enumerate(j["phrases"]):
            r, c = divmod(k, g)
            ex = V.extract_cell(
                img, g, r, c, px=px, tol=key_tol, do_fringe=fringe,
            )
            s, pm = ex["score"], ex["purity"]
            dest_bin = V.bin_for(s, pm)
            reasons = V.reject_reasons(s, pm)
            name = f"{p.stem}-{k}-{phrase.replace(' ', '_')}.png"
            Image.fromarray(np.dstack([ex["rgb"], ex["alpha"]]), "RGBA").save(dest / dest_bin / name)
            rows.append({
                "file": name,
                "subject": phrase,
                "bin": dest_bin,
                "usable_auto": dest_bin == "ok",
                "purity": round(pm, 3),
                "edge_touch": ex["edge_touch"],
                "fringe_dropped": ex["fringe_dropped"],
                "reject_reasons": reasons,
                "extractor": V.VERSION,
                "sheet": str(p),
                **s,
            })
    json.dump(rows, open(dest / "cells.json", "w"), indent=1)
    counts = {b: int(sum(r["bin"] == b for r in rows)) for b in ("ok", "purity_fail", "rejected")}
    summary = {
        "tag": tag,
        "extractor": V.VERSION,
        "thresholds": V.THRESHOLDS,
        "key_tol": key_tol,
        "grow_tol": V.GROW_TOL,
        "edge_key_tol": V.EDGE_KEY_TOL,
        "fringe": fringe,
        "jobs": str(jobs_path),
        "src": str(src) if src else None,
        "out": str(dest),
        "cells": len(rows),
        "missing_sheets": missing,
        **counts,
        "yield": round(counts["ok"] / max(len(rows), 1), 3),
        "purity_mean": round(float(np.mean([r["purity"] for r in rows])), 3) if rows else 0,
        "edge_touch": int(sum(r["edge_touch"] for r in rows)),
        "fringe_dropped_mean": round(float(np.mean([r["fringe_dropped"] for r in rows])), 3) if rows else 0,
    }
    json.dump(summary, open(dest / "summary.json", "w"), indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Re-extract raw 3x3 sheets with extract_v2 into a new folder.")
    ap.add_argument("--jobs", default=None, help="jobs.json or jobs.jsonl")
    ap.add_argument("--src", default=None, help="directory (or tree) of raw sheet PNGs")
    ap.add_argument("--out", required=True, help="NEW output folder (will be created)")
    ap.add_argument("--px", type=int, default=32)
    ap.add_argument("--sample", action="store_true", help="use the calib3x3 eval+gen sample")
    ap.add_argument("--no-fringe", action="store_true")
    ap.add_argument("--key-tol", type=int, default=V.KEY_TOL)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for tag, jobs, src in iter_job_groups(args.sample, args.jobs, args.src):
        print(f"extracting {tag} jobs={jobs} src={src} -> {out/tag}", flush=True)
        summaries.append(extract_group(tag, jobs, src, out, args.px, not args.no_fringe, args.key_tol))
    total = {
        "extractor": V.VERSION,
        "thresholds": V.THRESHOLDS,
        "out": str(out),
        "groups": summaries,
        "cells": sum(s["cells"] for s in summaries),
        "ok": sum(s["ok"] for s in summaries),
        "purity_fail": sum(s["purity_fail"] for s in summaries),
        "rejected": sum(s["rejected"] for s in summaries),
    }
    json.dump(total, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(total, indent=2))


if __name__ == "__main__":
    main()
