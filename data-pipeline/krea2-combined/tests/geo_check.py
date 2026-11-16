import os  # env placeholders ($DATA_ROOT etc.)
import json, numpy as np
from PIL import Image
R=os.path.expandvars("$DATA_ROOT")
rows=[json.loads(l) for l in open(R+"/test/cells5070.jsonl")]
from collections import Counter
print("bins",Counter(r["bin"] for r in rows), "sheet_px", Counter(tuple(r["sheet_px"]) for r in rows))
agree=[];alpha_agree=[]
for r in rows:
    a=np.asarray(Image.open(R+"/test/"+r["sprite_path"]).convert("RGBA")).astype(int)
    b=np.asarray(Image.open(R+"/raw/5070ti/sprites32/"+r["orig_file"]).convert("RGBA")).astype(int)
    m=(a[...,3]>0)&(b[...,3]>0)
    alpha_agree.append(((a[...,3]>0)==(b[...,3]>0)).mean())
    agree.append((np.abs(a[...,:3]-b[...,:3]).max(-1)[m]<=24).mean() if m.any() else 1)
print("vs original 2x2 sprites: mean alpha-mask agreement %.3f, mean rgb agreement(<=24) %.3f, min alpha %.3f"%(np.mean(alpha_agree),np.mean(agree),min(alpha_agree)))
# visual: 6 sheets, each sheet (256px thumb) + its 4 re-extracted cells at 4x
sel=sorted({r["sheet_id"] for r in rows})[:6]
W=256+4*(128+8)+16; canvas=Image.new("RGB",(W,len(sel)*(264)),(40,40,44))
for i,sid in enumerate(sel):
    rs=[r for r in rows if r["sheet_id"]==sid]
    sh=Image.open(rs[0]["sheet"]).convert("RGB").resize((256,256))
    canvas.paste(sh,(4,4+i*264))
    for r in sorted(rs,key=lambda r:r["cell"]):
        sp=Image.open(R+"/test/"+r["sprite_path"]).convert("RGBA").resize((128,128),Image.NEAREST)
        bg=Image.new("RGBA",(128,128),(210,210,210,255)); bg.alpha_composite(sp)
        canvas.paste(bg.convert("RGB"),(268+r["cell"]*136,4+i*264+64))
canvas.save(R+"/test/geometry_2x2_check.png"); print("saved")
