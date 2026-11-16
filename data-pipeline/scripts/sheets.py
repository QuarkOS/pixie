import json,random,os,shutil
from PIL import Image
R=os.environ.get('KREA_LORA_ROOT')
if not R:
    raise SystemExit('set KREA_LORA_ROOT')
D=f'{R}/dataset'
shutil.rmtree(D,ignore_errors=True)
meta=json.load(open(f'{R}/sprites/meta.json')); caps=json.load(open(f'{R}/sprites/captions.json'))
for e in meta: e['caption']=caps[e['id']]; e['dark_outline']=e['outline']>=0.6
rnd=random.Random(42); ids=[e for e in meta]; rnd.shuffle(ids)
n_eval=round(0.05*len(ids)); ev=ids[:n_eval]; tr=ids[n_eval:]
for e in ev: e['split']='eval'
for e in tr: e['split']='train'
json.dump(meta,open(f'{R}/sprites/meta.json','w'),indent=1)
POS2=['top left','top right','bottom left','bottom right']
def sheet(cells,g,path):
    S=Image.new('RGBA',(g*256,g*256),(255,0,255,255))
    for i,e in enumerate(cells):
        S.alpha_composite(Image.open(f"{R}/sprites/{e['id']}.png").convert('RGBA').resize((256,256),Image.NEAREST),((i%g)*256,(i//g)*256))
    S.convert('RGB').save(path+'.png')
    if g==2: body=', '.join(f"{p}: {e['caption']}" for p,e in zip(POS2,cells))
    else: body='; '.join(f"row {r+1}: "+', '.join(e['caption'] for e in cells[r*g:(r+1)*g]) for r in range(g))
    tail=', dark outline, plain magenta background' if all(e['dark_outline'] for e in cells) else (', plain magenta background' if not any(e['dark_outline'] for e in cells) else ', mostly dark outlines, plain magenta background')
    cap=f"pixel art 32x32 game icons, {g}x{g} sprite sheet, {body}{tail}"
    open(path+'.txt','w').write(cap); return dict(file=os.path.relpath(path+'.png',D),caption=cap,sprites=[e['id'] for e in cells])
def build(pool,g,passes,outdir,seed):
    os.makedirs(outdir,exist_ok=True); r=random.Random(seed); man=[]; k=g*g; idx=0
    for p in range(passes):
        for flag in (True,False):
            sub=[e for e in pool if e['dark_outline']==flag]; r.shuffle(sub)
            if not sub: continue
            while len(sub)%k:
                c=[e for e in pool if e['dark_outline']==flag and e not in sub[-(len(sub)%k):]] or [e for e in pool if e not in sub[-(len(sub)%k):]]
                sub.append(r.choice(c))
            for i in range(0,len(sub),k):
                man.append(sheet(sub[i:i+k],g,f'{outdir}/{idx:04d}')); idx+=1
    with open(f'{outdir}/manifest.jsonl','w') as f:
        for x in man: f.write(json.dumps(x)+'\n')
    return len(man)
for g,res,passes in [(2,512,1),(4,1024,3)]:
    for split,pool in [('train',tr),('eval',ev)]:
        n=build(pool,g,passes if split=='train' else 1,f'{D}/sheets_{g}x{g}_{res}/{split}',seed=g*10+(split=='eval'))
        print(g,split,len(pool),'sprites ->',n,'sheets')
