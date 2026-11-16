#!/usr/bin/env python3
"""Holdout filter + exact pixel dedup + train-ready folder + manifest + contact sheet."""
from __future__ import annotations
import json, os, random, re, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(os.path.expandvars("$DATA_ROOT"))
ORDER = {"3080ti": 0, "3060ti": 1, "5070ti": 2}

def norm(s):
    s = (s or "").lower().replace("_", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def load_holdouts():
    h = {}
    for p, tag in [(ROOT / "holdout/blind-50.repo.json", "blind50"),
                   (ROOT / "holdout/blind50.txt", "blind50"),
                   (ROOT / "holdout/eval50.txt", "eval50")]:
        if not p.exists():
            print("missing holdout file", p); sys.exit(2)
        items = json.load(open(p)) if p.suffix == ".json" else [l for l in open(p)]
        for it in items:
            n = norm(it if isinstance(it, str) else json.dumps(it))
            if n:
                h.setdefault(n, set()).add(tag)
    return h

def main():
    hold = load_holdouts()
    hold_list = sorted(hold)
    pats = [(n, re.compile(r"(^| )" + re.escape(n) + r"( |$)")) for n in hold_list]
    rows = [json.loads(l) for l in open(ROOT / "work/cells.jsonl")]
    rows.sort(key=lambda r: (ORDER[r["source"]], r["sheet_id"], r["cell"]))
    ids = Counter(r["id"] for r in rows)
    assert max(ids.values()) == 1, "duplicate ids"
    seen = {}
    for r in rows:
        cap = norm(r["caption"])
        r["holdout"] = None
        if cap in hold:
            r["holdout"] = {"match": "exact", "phrase": cap, "sets": sorted(hold[cap])}
        else:
            for n, rx in pats:
                if rx.search(cap):
                    r["holdout"] = {"match": "contains", "phrase": n, "sets": sorted(hold[n])}
                    break
        r["dup_of"] = None
        if r["pass_extract_v2"] and r["holdout"] is None:
            h = r["pixel_hash"]
            if h in seen:
                r["dup_of"] = seen[h]
            else:
                seen[h] = r["id"]
        r["final"] = bool(r["pass_extract_v2"] and r["holdout"] is None and r["dup_of"] is None)
        if not r["pass_extract_v2"]:
            r["status"] = "fail_" + r["bin"]
        elif r["holdout"]:
            r["status"] = "holdout_removed"
        elif r["dup_of"]:
            r["status"] = "duplicate_removed"
        else:
            r["status"] = "final"
        r["final_path"] = None
    # train-ready folder
    fin = ROOT / "final"
    if fin.exists():
        print("final/ already exists; refusing to overwrite"); sys.exit(3)
    (fin / "images").mkdir(parents=True)
    with open(fin / "metadata.jsonl", "w") as meta, open(fin / "captions.tsv", "w") as tsv:
        for r in rows:
            if not r["final"]:
                continue
            rel = f"images/{r['source']}/{r['sheet_id'] // 1000:03d}/{r['id']}.png"
            dst = fin / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(ROOT / r["sprite_path"], dst)
            dst.with_suffix(".txt").write_text(r["caption"] + "\n")
            r["final_path"] = "final/" + rel
            meta.write(json.dumps({"file_name": rel, "text": r["caption"], "id": r["id"], "source": r["source"],
                                   "grid": r["grid"], "seed": r["seed"], "edge_touch": r["edge_touch"]}) + "\n")
            tsv.write(f"{rel}\t{r['caption']}\n")
    with open(ROOT / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(ROOT / "rejected.jsonl", "w") as f:
        for r in rows:
            if not r["final"]:
                f.write(json.dumps({k: r[k] for k in ("id", "source", "status", "bin", "reject_reasons", "holdout", "dup_of", "caption", "sprite_path")}) + "\n")
    with open(ROOT / "flags_edge_touch.jsonl", "w") as f:
        for r in rows:
            if r["final"] and r["edge_touch"]:
                f.write(json.dumps({"id": r["id"], "final_path": r["final_path"], "caption": r["caption"]}) + "\n")
    # summary
    per = {}
    for src in ORDER:
        rs = [r for r in rows if r["source"] == src]
        per[src] = {
            "sheets": len({r["sheet_id"] for r in rs}),
            "raw_sprites": len(rs),
            "orig_run_usable": sum(1 for r in rs if r.get("orig_usable")),
            "pass_extract_v2": sum(r["pass_extract_v2"] for r in rs),
            "purity_fail": sum(r["bin"] == "purity_fail" for r in rs),
            "rejected_structure": sum(r["bin"] == "rejected" for r in rs),
            "holdout_removed": sum(r["status"] == "holdout_removed" for r in rs),
            "holdout_exact": sum(r["status"] == "holdout_removed" and r["holdout"]["match"] == "exact" for r in rs),
            "holdout_contains": sum(r["status"] == "holdout_removed" and r["holdout"]["match"] == "contains" for r in rs),
            "duplicates_removed": sum(r["status"] == "duplicate_removed" for r in rs),
            "final": sum(r["final"] for r in rs),
            "final_edge_touch": sum(r["final"] and r["edge_touch"] for r in rs),
            "unique_captions_final": len({r["caption"] for r in rs if r["final"]}),
        }
    tot = {k: sum(p[k] for p in per.values()) for k in next(iter(per.values()))}
    tot["unique_captions_final"] = len({r["caption"] for r in rows if r["final"]})
    hold_hits = Counter(r["holdout"]["phrase"] for r in rows if r["holdout"])
    summary = {"root": str(ROOT), "per_source": per, "total": tot,
               "holdout_phrases": len(hold), "holdout_hits_by_phrase": dict(hold_hits.most_common()),
               "dedup": "sha1 of 32x32 RGBA with RGB zeroed where alpha=0; among extract_v2 passes after holdout filter; first kept in order 3080ti,3060ti,5070ti then sheet,cell",
               "holdout_rule": "normalized caption equals a Blind-50/Eval-50 phrase, or contains one as a whole-word sequence"}
    json.dump(summary, open(ROOT / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    # contact sheet: 64 random final sprites, 8x8, 4x scale, grey checker
    random.seed(20261115)
    pick = random.sample([r for r in rows if r["final"]], 64)
    S, pad = 128, 8
    sheet = Image.new("RGB", (8 * (S + pad) + pad, 8 * (S + pad) + pad), (40, 40, 44))
    for i, r in enumerate(pick):
        im = Image.open(ROOT / r["final_path"]).convert("RGBA").resize((S, S), Image.NEAREST)
        bg = Image.new("RGBA", (S, S), (200, 200, 200, 255))
        for y in range(0, S, 16):
            for x in range(0, S, 16):
                if (x // 16 + y // 16) % 2:
                    bg.paste((230, 230, 230, 255), (x, y, x + 16, y + 16))
        bg.alpha_composite(im)
        sheet.paste(bg.convert("RGB"), (pad + (i % 8) * (S + pad), pad + (i // 8) * (S + pad)))
    sheet.save(ROOT / "contact_64_random_final.png")
    with open(ROOT / "contact_64_random_final.txt", "w") as f:
        for i, r in enumerate(pick):
            f.write(f"{i // 8},{i % 8}\t{r['id']}\t{r['caption']}\n")

if __name__ == "__main__":
    main()
