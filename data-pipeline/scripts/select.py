import pandas as pd, numpy as np, io, re, glob, json, os, random, collections
from PIL import Image
import imagehash
R=os.environ.get('KREA_LORA_ROOT')
if not R:
    raise SystemExit('set KREA_LORA_ROOT')
os.makedirs(f'{R}/sprites',exist_ok=True)
random.seed(7)
MAG=np.array([255,0,255])
def norm(img):
    a=np.array(img.convert('RGBA'))
    shadow=(a[...,3]<250)&(a[...,:3].max(-1)<40); a[shadow,3]=0  # drop semi-transparent black drop shadows
    a[...,3]=np.where(a[...,3]>=128,255,0); a[a[...,3]==0]=0; return a
def ok(a):
    op=a[...,3]>0; n=op.sum()
    if n<40 or n>0.92*1024: return False,'coverage'
    rgb=a[...,:3][op].astype(int)
    if (np.abs(rgb-MAG).max(-1)<=90).sum()>2: return False,'magenta'
    if len({tuple(c) for c in rgb})<3: return False,'colors'
    return True,''
def outline(a):
    op=a[...,3]>0; p=np.pad(op,1)
    edge=op & ~(p[:-2,1:-1]&p[2:,1:-1]&p[1:-1,:-2]&p[1:-1,2:])
    rgb=a[...,:3][edge].astype(float); lum=rgb@[0.299,0.587,0.114]
    return float((lum<70).mean()) if len(lum) else 0
cands=[]
def add(src,key,img,label,group):
    a=norm(img); g,why=ok(a)
    if not g: return why
    cands.append(dict(src=src,key=key,a=a,label=label,group=group)); return 'ok'
stats=collections.Counter()
# pixl-data
df=pd.concat([pd.read_parquet(f'{R}/src/pixl_sprites_train.parquet'),pd.read_parquet(f'{R}/src/pixl_sprites_test.parquet')])
df=df[(df.width=='32')&(df.height=='32')&(df.pack!='gameicons_game-icons')]
KEEP_DC=re.compile(r'/(item/(weapon|armor|book|misc|ring|potion|amulet|food|wand|rod|gold|scroll|staff)|monster/(animals|fungi_plants|aquatic|dragons|undead|demons|nonliving|holy|eyes|amorphous)|dungeon/(trees|statues|altars|doors)|dungeon/(chest|large_box|boulder|blue_fountain|sparkling_fountain|dry_fountain|blood_fountain))')
for r in df.itertuples():
    if r.pack.startswith('dcss'):
        if not KEEP_DC.search(r.path) or '/artefact/' in r.path or 'unique' in r.path: stats['dcss_skip_cat']+=1; continue
        base=re.sub(r'(_\d+[a-z]?)?(_old|_new)?\.png$','',r.path.split('/')[-1])
        grp='dcss:'+r.path.rsplit('/',1)[0]+'/'+base
    else: grp=r.pack+':'+r.label
    img=Image.open(io.BytesIO(r.image['bytes']))
    stats[r.pack+':'+add(r.pack,r.path,img,r.label,grp)]+=1
# lazarini
NAMES={'W':'weapon','I':'item','P':'potion','A':'armour','Ac':'accessory','C':'clothing','E':'material'}
for f in sorted(glob.glob(f'{R}/src/lazarini/*.png')):
    n=os.path.basename(f)[:-4]; pre=n.split('_')[0]
    if pre=='S': stats['lazarini:skip_skill']+=1; continue
    a=norm(Image.open(f)); ys,xs=np.where(a[...,3]>0)
    if ys.max()-ys.min()>=32 or xs.max()-xs.min()>=32: stats['lazarini:too_big']+=1; continue
    cy=min(max((ys.min()+ys.max())//2-15,0),2); cx=min(max((xs.min()+xs.max())//2-15,0),2)
    y0=min(max(ys.min(),cy),ys.max()-31) if False else None
    # choose crop offset in [0,2] containing bbox
    oy=[o for o in range(3) if o<=ys.min() and ys.max()<o+32]; ox=[o for o in range(3) if o<=xs.min() and xs.max()<o+32]
    a=a[oy[len(oy)//2]:oy[len(oy)//2]+32, ox[len(ox)//2]:ox[len(ox)//2]+32]
    words=re.sub(r'([a-z])([A-Z])',r'\1 \2',re.sub(r'\d+','','_'.join(n.split('_')[1:]))).replace('_',' ').replace('C ','').strip().lower()
    label=(words+' '+NAMES[pre]) if pre=='P' else words
    stats['lazarini:'+add('lazarini',n,Image.fromarray(a),label,'laz:'+re.sub(r'\d+','',n))]+=1
OGA=['food-bluecarrot16_20201221','food-jetrel_20201222','food-thekingphoenix','food-arlantr_20201229','food-other_20230225','tool-thekingphoenix','tool-yd','tool-fleurman_20201226']
for d in OGA:
    for f in sorted(glob.glob(f'{R}/src/oga/{d}/**/*.png',recursive=True)):
        if '/orig/' in f or '/24x24/' in f: continue
        im=Image.open(f)
        if im.size!=(32,32): continue
        n=os.path.basename(f)[:-4]; base=re.sub(r'[_-]?\d+$','',n)
        label=base.replace('_',' ').replace('-',' ').strip().lower()
        if '/32x32/' in f: grp=d+':'+f.split('/32x32/')[1].split('/')[0]+'/'+re.sub(r'[_-]?\d+$','',base)
        else: grp=d+':'+base
        stats[d+':'+add(d,f.split('/src/oga/')[1],im,label,grp)]+=1
print(stats)
# dedup: phash within whole set, keep <=2 per group (dcss variants) and <=1 for animation-ish groups
for c in cands: c['h']=imagehash.phash(Image.fromarray(c['a']).convert('RGB').resize((64,64),Image.NEAREST))
random.shuffle(cands)
kept=[];bygroup=collections.Counter();drop=collections.Counter()
LIM={'food-bluecarrot16_20201221':1,'buch_platformer-in-the-forest':1,'ansimuz_sunny-land':1,'kenney_pixel-shmup':2}
for c in cands:
    lim=LIM.get(c['src'],1 if c['src'].startswith('dcss') else 3)
    if bygroup[c['group']]>=lim: drop['group']+=1; continue
    if any(c['h']-k['h']<=5 and np.abs(c['a'].astype(int)-k['a'].astype(int)).mean()<14 for k in kept): drop['near_dup']+=1; continue
    kept.append(c); bygroup[c['group']]+=1
print('kept',len(kept),drop, collections.Counter(k['src'] for k in kept))
# cap dcss so it doesn't dominate; target 840 total
non=[k for k in kept if not k['src'].startswith('dcss')]; dc=[k for k in kept if k['src'].startswith('dcss')]
TARGET=840
# balance dcss by top category
def cat(k):
    p=k['key'].split('/'); return p[3]+'/'+p[4].replace('.png','')
bycat=collections.defaultdict(list)
for k in dc: bycat[cat(k)].append(k)
need=TARGET-len(non); sel=[]
while len(sel)<need and any(bycat.values()):
    for cc in sorted(bycat):
        if bycat[cc] and len(sel)<need: sel.append(bycat[cc].pop())
final=non+sel; random.shuffle(final)
meta=[]
for i,k in enumerate(final):
    fn=f'{R}/sprites/{i:04d}.png'; Image.fromarray(k['a'],'RGBA').save(fn)
    meta.append(dict(id=f'{i:04d}',src=k['src'],key=k['key'],label=k['label'],outline=round(outline(k['a']),3)))
json.dump(meta,open(f'{R}/sprites/meta.json','w'),indent=1)
print('final',len(final),collections.Counter(m['src'] for m in meta))
print('dcss cats',collections.Counter(cat(k) for k in sel).most_common(40))
print('outline>=0.6',sum(m['outline']>=0.6 for m in meta))
