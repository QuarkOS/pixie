"""Build table, 8x contact sheets, side-by-side from $KREA_EVAL_ROOT"""
import json,os,glob,re
from PIL import Image,ImageDraw,ImageFont
R=os.environ.get('KREA_EVAL_ROOT')
if not R:
    raise SystemExit('set KREA_EVAL_ROOT to the eval raw directory')
O=os.environ.get('KREA_EVAL_OUT') or os.path.dirname(R.rstrip('/'))
cfgs=sorted(d for d in os.listdir(R) if os.path.isdir(f'{R}/{d}'))
def font(s):
    try: return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',s)
    except: return ImageFont.load_default()
F=font(14); FB=font(20)
def items(c):
    j=json.load(open(f'{R}/{c}/jobs.json'))[1:]
    out=[]
    for k,job in enumerate(j):
        for i,name in enumerate(job['items']):
            fn=f"{R}/{c}/sprites/sheet-{k:03d}-{i}-{re.sub(r'[^a-z0-9]+','_',name.lower()).strip('_')}.png"
            if not os.path.exists(fn):
                g=glob.glob(f"{R}/{c}/sprites/sheet-{k:03d}-{i}-*.png"); fn=g[0] if g else None
            out.append((name,fn))
    return out
rows=[]
for c in cfgs:
    e=json.load(open(f'{R}/{c}/extract.json')); t=json.load(open(f'{R}/{c}/timing.json'))
    n_sheets=t['n_timed']; usable=e['usable_auto']
    sps=usable/(t['s_per_sheet']*n_sheets)
    rows.append((c,e['art_px_median'],f"{usable}/{e['cells']}",e['purity_mean'],t['s_per_sheet'],sps,t['peak_vram_mib']))
with open(f'{O}/table.md','w') as f:
    f.write('| config | art px (median) | usable | purity | s/sheet | usable sprites/s | peak VRAM MiB |\n|---|---|---|---|---|---|---|\n')
    for r in rows: f.write(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]} |\n')
print(open(f'{O}/table.md').read())
S=8; CW=32*S+16; LH=34
def bg(): return Image.new('RGB',(32*S,32*S),(200,200,200))
def tile(fn):
    b=Image.new('RGBA',(32*S,32*S),(0,0,0,0))
    ck=Image.new('RGBA',(32*S,32*S))
    d=ImageDraw.Draw(ck)
    for y in range(0,32*S,16):
        for x in range(0,32*S,16): d.rectangle([x,y,x+15,y+15],fill=(225,225,225,255) if (x//16+y//16)%2 else (190,190,190,255))
    if fn:
        im=Image.open(fn).convert('RGBA')
        if im.size!=(32,32): im=im.resize((32,32),Image.NEAREST)
        ck.alpha_composite(im.resize((32*S,32*S),Image.NEAREST))
    return ck
for c in cfgs:
    it=items(c); cols=9; rws=(len(it)+cols-1)//cols
    W=cols*CW; H=40+rws*(32*S+LH)
    im=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(im); d.text((8,8),f'{c}  (8x nearest, 72 benchmark subjects)',fill='black',font=FB)
    for n,(name,fn) in enumerate(it):
        x=(n%cols)*CW+8; y=40+(n//cols)*(32*S+LH)
        im.paste(tile(fn),(x,y)); d.text((x,y+32*S+4),name[:34],fill='black',font=F)
    im.save(f'{O}/contact_{c}_8x.png')
# side by side: rows = subjects, cols = configs  (split in 2 images of 36)
names=[n for n,_ in items(cfgs[0])]
maps={c:dict(items(c)) for c in cfgs}
for part in range(2):
    sub=names[part*36:(part+1)*36]; S2=4; T=32*S2+8
    W=220+len(cfgs)*T; H=40+len(sub)*(T)
    im=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(im)
    for j,c in enumerate(cfgs): d.text((220+j*T,10),c.replace('_m1.0',''),fill='black',font=F)
    for i,nm in enumerate(sub):
        y=40+i*T; d.text((6,y+50),nm[:28],fill='black',font=F)
        for j,c in enumerate(cfgs):
            im.paste(tile(maps[c].get(nm)).resize((32*S2,32*S2),Image.NEAREST),(220+j*T,y))
    im.save(f'{O}/side_by_side_3x3_vs_2x2_part{part+1}.png')
print('ok')
