import os  # env placeholders ($DATA_ROOT etc.)
import json, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
R = Path(os.path.expandvars("$DATA_ROOT")); OUT = R / "spotcheck"
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
fin = [json.loads(l) for l in open(R / "manifest.jsonl")]
fin = [r for r in fin if r["final"]]
def stats(r):
    a = np.asarray(Image.open(R / r["sprite_path"]).convert("RGBA")).astype(int)
    al = a[..., 3] > 0; rr, g, b = a[..., 0], a[..., 1], a[..., 2]
    mag = al & (g < 110) & (rr > 140) & (b > 140) & (np.abs(rr - b) < 90)
    ring = al & ndimage.binary_dilation(~al)
    e = np.zeros_like(al); e[0] = e[-1] = True; e[:, 0] = e[:, -1] = True
    sides = sum(bool(x.any()) for x in (al[0], al[-1], al[:, 0], al[:, -1]))
    return a, mag & ring, int((al & e).sum()), sides
random.seed(7)
def sheet(sample, name, mark):
    S, cap, cols = 128, 30, 10
    im = Image.new("RGB", (cols * (S + 6) + 6, 5 * (S + cap + 6) + 6), (36, 36, 40)); d = ImageDraw.Draw(im)
    for i, r in enumerate(sample):
        a, m, ep, sides = stats(r)
        x, y = 6 + (i % cols) * (S + 6), 6 + (i // cols) * (S + cap + 6)
        sp = Image.fromarray(a.astype(np.uint8), "RGBA").resize((S, S), Image.NEAREST)
        bg = Image.new("RGBA", (S, S), (200, 200, 200, 255)); bg.alpha_composite(sp); bg = bg.convert("RGB")
        if mark:
            dd = ImageDraw.Draw(bg)
            for yy, xx in zip(*np.nonzero(m)): dd.rectangle((xx * 4, yy * 4, xx * 4 + 3, yy * 4 + 3), outline=(0, 255, 0))
        im.paste(bg, (x, y)); d.text((x + 2, y + 1), str(i), fill=(255, 0, 0), font=font)
        d.text((x, y + S + 1), r["caption"][:24], fill=(230, 230, 230), font=font)
        d.text((x, y + S + 14), f"{r['source']} mag{int(m.sum())} e{ep} s{sides}", fill=(150, 150, 160), font=font)
    im.save(OUT / f"{name}.png")
    json.dump([{"idx": i, "id": r["id"], "caption": r["caption"]} for i, r in enumerate(sample)], open(OUT / f"{name}.json", "w"))
# full pass for counts
mo, sides_ct, rows2 = 0, {}, []
for r in fin:
    a, m, ep, sides = stats(r)
    rows2.append((r, int(m.sum()), ep, sides))
mag = [r for r, m, ep, s in rows2 if m > 0]
heavy = [r for r, m, ep, s in rows2 if ep >= 32]
twoside = [r for r, m, ep, s in rows2 if s >= 2]
sheet(random.sample(mag, 50), "magenta_outline_marked", True)
sheet(random.sample(heavy, 50), "edge_px_ge32", False)
from collections import Counter
res = {"final": len(fin), "mag_outline_sprites": len(mag), "mag_outline_px_total": sum(m for _, m, _, _ in rows2),
       "edge_px>=32": len(heavy), "edge_px>=24": sum(ep >= 24 for _, _, ep, _ in rows2),
       "touch_sides_hist": dict(Counter(s for *_, s in rows2)),
       "touch>=2_sides": len(twoside), "touch>=3_sides": sum(s >= 3 for *_, s in rows2)}
for name, lst in [("cand_magenta_outline_cleanup", mag), ("cand_flag_edge_px_ge32", heavy)]:
    with open(OUT / f"{name}.txt", "w") as f:
        for r in lst: f.write(r["final_path"] + "\n")
json.dump(res, open(OUT / "spotcheck2_summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
