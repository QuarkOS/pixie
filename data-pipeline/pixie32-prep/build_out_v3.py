#!/usr/bin/env python3
"""Build out_v3 metadata from out_v2 plus grid recaption captions.

Copies out_v2 metadata, replaces real-sprite captions with the new grid
captions, and stores the previous caption in caption_orig. Sprites that
failed or were skipped keep the old caption (caption_orig is still set to
that same old text). Does not modify out_v2. images.npy and img_idx.npy
are symlinked to save disk.
"""
from __future__ import annotations

import argparse, csv, json, os, shutil
from pathlib import Path

HOME = Path.home()
V2 = Path(os.path.expandvars("$PIXIE_ROOT")) / "data32/out_v2"
V3 = Path(os.path.expandvars("$PIXIE_ROOT")) / "data32/out_v3"
CAPS = Path(os.path.expandvars("$PIXIE_ROOT")) / "data32/recaption_grid_full/captions.jsonl"


def log(*a):
    print(*a, flush=True)


def load_captions(path: Path) -> dict[int, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[int(r["idx"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, default=V2)
    ap.add_argument("--v3", type=Path, default=V3)
    ap.add_argument("--captions", type=Path, default=CAPS)
    args = ap.parse_args()
    v2, v3, cap_path = args.v2, args.v3, args.captions
    if not cap_path.exists():
        raise SystemExit(f"captions file missing: {cap_path}")
    caps = load_captions(cap_path)
    v3.mkdir(parents=True, exist_ok=True)
    (v3 / "cache").mkdir(exist_ok=True)

    n = n_real = n_new = n_kept = n_synth = 0
    meta_out = v3 / "meta.jsonl"
    with (v2 / "meta.jsonl").open(encoding="utf-8") as fin, meta_out.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r.get("idx", i)) != i:
                raise SystemExit(f"meta idx mismatch at line {i}: {r.get('idx')}")
            n += 1
            if r.get("synthetic"):
                n_synth += 1
            else:
                n_real += 1
                old = r.get("caption") or ""
                rec = caps.get(i)
                new = (rec or {}).get("caption") or ""
                ok = bool(rec and rec.get("ok") and new.strip())
                r["caption_orig"] = old
                if ok:
                    r["caption"] = new
                    r["caption_source"] = "vlm-recaption-grid"
                    r["caption_model"] = "vlm-recaption"
                    r["caption_thinking"] = "low"
                    r["caption_style"] = "6-16token-grid16"
                    n_new += 1
                else:
                    # keep old caption
                    r["caption"] = old
                    r["caption_source"] = r.get("caption_source") or r.get("label_source") or "existing"
                    r["recaption_kept_old"] = True
                    n_kept += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    man_in = v2 / "manifest.csv"
    man_out = v3 / "manifest.csv"
    with man_in.open(encoding="utf-8", newline="") as fin, man_out.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        fields = list(reader.fieldnames or [])
        if "caption_orig" not in fields:
            # place after caption if present
            if "caption" in fields:
                idx = fields.index("caption") + 1
                fields = fields[:idx] + ["caption_orig"] + fields[idx:]
            else:
                fields.append("caption_orig")
        if "caption_source" not in fields:
            fields.append("caption_source")
        writer = csv.DictWriter(fout, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            path = row.get("path") or ""
            try:
                idx = int(path.rsplit(":", 1)[-1])
            except ValueError:
                idx = None
            if idx is not None and idx in caps and caps[idx].get("ok") and (caps[idx].get("caption") or "").strip():
                row["caption_orig"] = row.get("caption") or ""
                row["caption"] = caps[idx]["caption"]
                row["caption_source"] = "vlm-recaption-grid"
            else:
                row["caption_orig"] = row.get("caption") or ""
                if not row.get("caption_source"):
                    row["caption_source"] = row.get("caption_source") or ""
            writer.writerow(row)

    def link_or_copy(src: Path, dst: Path):
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

    link_or_copy(v2 / "images.npy", v3 / "images.npy")
    if (v2 / "cache" / "img_idx.npy").exists():
        link_or_copy(v2 / "cache" / "img_idx.npy", v3 / "cache" / "img_idx.npy")
    for name in ("weights.json", "stats.json", "quantization_real_cc0.json"):
        s = v2 / name
        if s.exists():
            shutil.copy2(s, v3 / name)

    recap = {
        "source_v2": str(v2),
        "out_v3": str(v3),
        "captions_file": str(cap_path),
        "n": n,
        "synthetic": n_synth,
        "real": n_real,
        "real_recaptioned": n_new,
        "real_kept_old": n_kept,
        "images_symlink": str(v3 / "images.npy"),
        "meta": str(meta_out),
        "manifest": str(man_out),
    }
    # keep summary.json from v2 but add recaption counts
    summary = {}
    if (v2 / "summary.json").exists():
        summary = json.loads((v2 / "summary.json").read_text())
    summary["recaption"] = recap
    (v3 / "summary.json").write_text(json.dumps(summary, indent=2))
    (v3 / "recaption_apply.json").write_text(json.dumps(recap, indent=2))
    log(json.dumps(recap, indent=2))
    if n_new + n_kept != n_real:
        raise SystemExit(f"real count mismatch: new {n_new} kept {n_kept} real {n_real}")


if __name__ == "__main__":
    main()
