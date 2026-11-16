#!/usr/bin/env python3
"""Fast sprite batch: 4x4 sheets @512, LoRA 0.7, 5 steps, pipelined requests (2 in flight).
usage: fast_batch.py SUBJECTS.txt OUTDIR [--seed N] [--base URL]
SUBJECTS.txt: one subject per line (or several per line separated by '|'); '#' lines ignored.
Writes OUTDIR/sheets/sheet-NNN.png, OUTDIR/sprites32/*.png, OUTDIR/sprites16/*.png, OUTDIR/cells.json"""
import argparse, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_jobs import sheet_prompt, sigmas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects"); ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, default=2000); ap.add_argument("--base", default="http://127.0.0.1:17890")
    ap.add_argument("--grid", type=int, default=4); ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--lora", type=float, default=0.7); ap.add_argument("--steps", type=int, default=5)
    a = ap.parse_args()
    subs = []
    for line in open(a.subjects):
        line = line.strip()
        if not line or line.startswith("#"): continue
        subs += [s.strip() for s in line.split("|") if s.strip() and not s.strip().isdigit()]
    per = a.grid * a.grid
    while len(subs) % per: subs.append(subs[len(subs) % per])  # pad last sheet by repeating subjects
    os.makedirs(f"{a.outdir}/sheets", exist_ok=True)
    jobs = []
    for k in range(len(subs) // per):
        items = subs[k * per:(k + 1) * per]
        jobs.append({"prompt": sheet_prompt(items, a.grid, 32), "items": items, "grid": a.grid, "seed": a.seed + k,
                     "width": a.size, "height": a.size, "steps": a.steps, "sigmas": sigmas(a.steps),
                     "lora_mult": a.lora, "out": f"{a.outdir}/sheets/sheet-{k:03d}.png"})
    jf = f"{a.outdir}/jobs.json"; json.dump(jobs, open(jf, "w"), indent=1)
    subprocess.run([sys.executable, f"{HERE}/bench.py", jf, "--base", a.base, "--depth", "2", "--warmup", "0",
                    "--result", f"{a.outdir}/timing.json"], check=True)
    py = os.environ.get("SPRITE_PY", sys.executable)
    for px in (32, 16):
        subprocess.run([py, f"{HERE}/extract.py", jf, f"{a.outdir}/sprites{px}", "--px", str(px)], check=True)

if __name__ == "__main__":
    main()
