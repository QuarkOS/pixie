import os  # env placeholders ($DATA_ROOT etc.)
import json, numpy as np, sys
from PIL import Image
import extract_v2 as V
V.pitch_of=None
rows=[json.loads(l) for l in open("cells.jsonl")]
bad=0; n=0
for r in rows[:90]:
    img=np.asarray(Image.open(r["sheet"]).convert("RGB")); g=int(r["grid"][0])
    ex=V.extract_cell(img,g,*divmod(r["cell"],g),px=32)
    s=ex["score"]; b=V.bin_for(s,ex["purity"])
    im=np.asarray(Image.open(r["sprite_path"].replace("extract/","extract/") if False else os.path.expandvars("$DATA_ROOT/test/")+r["sprite_path"]))
    same = b==r["bin"] and s["spill"]==r["spill"] and s["fg"]==r["fg"] and np.array_equal(im, np.dstack([ex["rgb"],ex["alpha"]]))
    n+=1; bad+= (not same)
print("checked",n,"mismatch",bad)
from collections import Counter
print(Counter(r["bin"] for r in rows))
lab={}
for l in open(os.path.expandvars("$GEN_ROOT/krea2-gen-3080ti/data/labels.jsonl")):
    j=json.loads(l)
    if j["sheet_id"]<40: lab[(j["sheet_id"],j["cell"])]=j
agree=sum(lab[(r["sheet_id"],r["cell"])]["bin"]==r["bin"] for r in rows)
print("bin agreement with original run labels",agree,"/",len(rows))
