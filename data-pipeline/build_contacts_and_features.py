#!/usr/bin/env python3
"""Build 8x contact pages and a feature table from raw 3x3 @ 576 sheets. Read-only on sources."""
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "krea2-gen")
import extract as X

_root = os.environ.get("KREA_CALIB_ROOT")
if not _root:
    raise SystemExit("set KREA_CALIB_ROOT")
ROOT = Path(_root)
OUT = ROOT / "contacts"
FEAT = ROOT / "features.jsonl"
OUT.mkdir(parents=True, exist_ok=True)

def font(s):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", s)
    except Exception:
        return ImageFont.load_default()
F, FS, FB = font(13), font(11), font(18)

def checker(s=8):
    ck = Image.new("RGBA", (32*s, 32*s))
    d = ImageDraw.Draw(ck)
    for y in range(0, 32*s, 16):
        for x in range(0, 32*s, 16):
            d.rectangle([x,y,x+15,y+15], fill=(225,225,225,255) if ((x//16+y//16)%2) else (190,190,190,255))
    return ck

def extract_cell(img, g, r, c, px=32):
    H, W = img.shape[:2]
    cell = img[round(H*r/g):round(H*(r+1)/g), round(W*c/g):round(W*(c+1)/g)]
    s = X.score_cell(cell)
    s["spill"] = round(X.spill(img, g, r, c), 3)
    try:
        from pitch import pitch_of
        ppx, ppy, _, _ = pitch_of(cell)
        s["art_px"] = round(cell.shape[1] / max(ppx, 1e-6), 1)
    except Exception:
        s["art_px"] = None
    sp, pur = X.mode_pool(cell, px)
    alpha = np.where(X.is_bg(sp), 0, 255).astype(np.uint8)
    sp = X.palette_clean(sp, alpha)
    lab, n = X.ndimage.label(alpha > 0)
    if n > 1:
        sizes = X.ndimage.sum(alpha > 0, lab, range(1, n + 1))
        for i, sz in enumerate(sizes, 1):
            if sz <= 1:
                alpha[lab == i] = 0
        lab, n = X.ndimage.label(alpha > 0)
        sizes = X.ndimage.sum(alpha > 0, lab, range(1, n + 1)) if n else []
    else:
        sizes = X.ndimage.sum(alpha > 0, lab, range(1, n + 1)) if n else []
    fgb = alpha > 0
    pm = float(pur[fgb].mean()) if fgb.any() else 0.0
    dest = X.bin_for(s, pm)
    edge = bool(X.touches_32_edge(alpha))
    # extra metrics
    ys, xs = np.where(fgb)
    if len(ys):
        bbox = (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)
        bbox_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
    else:
        bbox = None
        bbox_area = 0
    # magenta / fringe leftover in the 32x32 rgb where alpha>0
    rgb = sp.astype(int)
    mag = np.abs(rgb - np.array([255,0,255])).max(-1)
    tint = (rgb[...,1] < 80) & (rgb[...,0] > 140) & (rgb[...,2] > 140) & (np.abs(rgb[...,0]-rgb[...,2]) < 90)
    fringe32 = int((((mag <= 110) | tint) & fgb).sum()) if fgb.any() else 0
    # fringe on the raw 192 cell border
    ch, cw = cell.shape[:2]
    b = 3
    border = np.zeros((ch,cw), bool)
    border[:b]=True; border[-b:]=True; border[:,:b]=True; border[:,-b:]=True
    cell_fg = ~X.is_bg(cell)
    # count borderline magenta-tinted pixels just inside the cell that current is_bg missed
    a = cell.astype(int)
    missed = ((np.abs(a-np.array([255,0,255])).max(-1) <= 130) | ((a[...,1]<90)&(a[...,0]>130)&(a[...,2]>130))) & ~X.is_bg(cell)
    return {
        "score": s, "purity": round(pm, 3), "bin_old": dest, "edge_touch": edge,
        "n_comp": int(n), "comp_sizes": [int(x) for x in sizes] if n else [],
        "fg_px": int(fgb.sum()), "bbox": bbox, "bbox_area": int(bbox_area),
        "fringe32": fringe32, "missed_magenta_in_cell": int(missed.sum()),
        "border": s.get("border"),
        "sprite": np.dstack([sp, alpha]),
        "cell": cell,
    }

def load_items():
    items = []
    # eval
    jobs = json.load(open(ROOT/"sample_eval/p3_s1500_m1.0/jobs.json"))
    jobs = [j for j in jobs if "warmup" not in j.get("out","")]
    for si, job in enumerate(jobs):
        sheet = ROOT/"sample_eval/p3_s1500_m1.0"/f"sheet-{si:03d}.png"
        img = np.asarray(Image.open(sheet).convert("RGB"))
        for k, name in enumerate(job["items"]):
            r, c = divmod(k, 3)
            rec = extract_cell(img, 3, r, c)
            rec.update(id=f"ev-{si:03d}-{k}", source="eval", sheet_id=si, cell=k,
                       phrase=name, sheet_path=str(sheet), cell_idx=k)
            items.append(rec)
    # gen sample
    for line in open(ROOT/"sample_gen/jobs.jsonl"):
        job = json.loads(line)
        si = job["sheet_id"]
        sheet = ROOT/"sample_gen/sheets/000"/f"sheet-{si:06d}.png"
        img = np.asarray(Image.open(sheet).convert("RGB"))
        for k, cell in enumerate(job["cells"]):
            r, c = divmod(k, 3)
            rec = extract_cell(img, 3, r, c)
            rec.update(id=f"g-{si:03d}-{k}", source="gen", sheet_id=si, cell=k,
                       phrase=cell["phrase"], sheet_path=str(sheet), cell_idx=k)
            items.append(rec)
    return items

def save_features(items):
    with FEAT.open("w") as f:
        for it in items:
            o = {k:v for k,v in it.items() if k not in ("sprite","cell")}
            o["score"] = it["score"]
            f.write(json.dumps(o)+"\n")
    print("features", FEAT, "n", len(items))

def page(items, path, title):
    S=8; cw, ch = 32*S+8, 32*S+40
    cols=9
    rows = (len(items)+cols-1)//cols
    im = Image.new("RGB", (8+cols*cw, 40+rows*ch), "white")
    d = ImageDraw.Draw(im)
    d.text((8,8), title, fill="black", font=FB)
    ck0 = checker(S)
    for i, it in enumerate(items):
        x = 8+(i%cols)*cw; y=40+(i//cols)*ch
        ck = ck0.copy()
        sp = Image.fromarray(it["sprite"], "RGBA")
        ck.alpha_composite(sp.resize((32*S,32*S), Image.NEAREST))
        im.paste(ck, (x,y))
        phrase = it["phrase"][:20]
        tag = f"{it['id']} {it['bin_old'][:4]} p={it['purity']:.2f}"
        if it["edge_touch"]: tag += "*"
        d.text((x, y+32*S+1), phrase, fill="black", font=F)
        d.text((x, y+32*S+16), tag, fill=(80,80,80), font=FS)
    im.save(path)
    print("wrote", path, im.size)

def main():
    items = load_items()
    save_features(items)
    # also save 32x32 sprites for inspection
    spdir = ROOT/"preview_sprites"
    spdir.mkdir(exist_ok=True)
    for it in items:
        Image.fromarray(it["sprite"], "RGBA").save(spdir/f"{it['id']}.png")
    ev = [it for it in items if it["source"]=="eval"]
    page(ev, OUT/"contact_eval_72.png", "EVAL 72  (old bin + purity; * = 32px edge)  — labels are later, this is just the sheet")
    gen = [it for it in items if it["source"]=="gen"]
    for a in range(0, 34, 8):
        chunk = [it for it in gen if a <= it["sheet_id"] < a+8]
        if not chunk: continue
        page(chunk, OUT/f"contact_gen_{a:03d}-{min(a+7,33):03d}.png",
             f"GEN sheets {a:03d}-{min(a+7,33):03d}  (old bin + purity; * = 32px edge)")
    # summary of old bins
    from collections import Counter
    print("old bins", Counter((it["source"], it["bin_old"]) for it in items))
    print("edge", sum(it["edge_touch"] for it in items), "/", len(items))

if __name__ == "__main__":
    main()
