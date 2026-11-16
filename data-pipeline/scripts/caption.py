import json,os,io,base64,urllib.request,concurrent.futures as cf,time,re
from PIL import Image,ImageDraw
R=os.environ.get('KREA_LORA_ROOT')
if not R:
    raise SystemExit('set KREA_LORA_ROOT')
m=json.load(open(f'{R}/sprites/meta.json'))
KEY=os.environ['GEMINI_API_KEY_PIXIE']; MODEL=os.environ.get('CAP_MODEL','gemini-flash-latest')
out_p=f'{R}/sprites/captions.json'; caps=json.load(open(out_p)) if os.path.exists(out_p) else {}
bk=json.load(open(f'{R}/src/captions_by_key.json')) if os.path.exists(f'{R}/src/captions_by_key.json') else {}
caps={e['id']:bk[e['src']+'|'+e['key']] for e in m if e['src']+'|'+e['key'] in bk}
todo=[e for e in m if e['id'] not in caps]; print('reused',len(caps),'todo',len(todo)); B=20
def grid(batch):
    S=Image.new('RGB',(5*150,4*170),(150,150,150)); d=ImageDraw.Draw(S)
    for i,e in enumerate(batch):
        x,y=(i%5)*150+11,(i//5)*170+30
        S.paste(Image.open(f"{R}/sprites/{e['id']}.png").resize((128,128),Image.NEAREST),(x,y),Image.open(f"{R}/sprites/{e['id']}.png").resize((128,128),Image.NEAREST))
        d.text((x,y-22),f"#{i}",fill=(0,0,0))
    b=io.BytesIO(); S.save(b,'PNG'); return base64.b64encode(b.getvalue()).decode()
PROMPT="""These are {n} small 32x32 pixel-art game sprites on a grey background, numbered #0..#{k}. For each one write a short everyday caption (2 to 6 words) that a normal person would use to ask an image model for it: name the object plainly and its main colour or material, e.g. "red health potion", "wooden crossbow", "grey stone cat statue", "green slime blob", "loaf of bread". Use the filename hint only to understand what it is; replace game-specific or fantasy proper names with plain words (e.g. "zonguldrok lich" -> "skeleton wizard in robes"). No words like pixel, sprite, icon, art. Hints:
{hints}
Reply with JSON only: a list of objects {{"i": number, "caption": string}}."""
def call(batch):
    hints="\n".join(f"#{i}: {e['label']}" for i,e in enumerate(batch))
    body={"contents":[{"parts":[{"text":PROMPT.format(n=len(batch),k=len(batch)-1,hints=hints)},{"inline_data":{"mime_type":"image/png","data":grid(batch)}}]}],
          "generationConfig":{"responseMimeType":"application/json","temperature":0.2}}
    for t in range(4):
        try:
            rq=urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",data=json.dumps(body).encode(),headers={"Content-Type":"application/json","x-goog-api-key":KEY})
            r=json.load(urllib.request.urlopen(rq,timeout=120))
            txt=r['candidates'][0]['content']['parts'][-1]['text']; arr=json.loads(txt)
            return {batch[o['i']]['id']:re.sub(r'\s+',' ',o['caption']).strip().strip('.').lower() for o in arr if 0<=o['i']<len(batch)}
        except Exception as ex: err=str(ex)[:200]; time.sleep(3*(t+1))
    print('batch failed',err); return {}
bs=[todo[i:i+B] for i in range(0,len(todo),B)]
with cf.ThreadPoolExecutor(6) as ex:
    for res in ex.map(call,bs): caps.update(res)
json.dump(caps,open(out_p,'w'),indent=1); bk.update({e['src']+'|'+e['key']:caps[e['id']] for e in m if e['id'] in caps}); json.dump(bk,open(f'{R}/src/captions_by_key.json','w'),indent=1); print('captioned',len(caps),'of',len(m))
