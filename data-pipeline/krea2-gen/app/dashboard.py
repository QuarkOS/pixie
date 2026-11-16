#!/usr/bin/env python3
"""Single-file local dashboard (stdlib only). http://127.0.0.1:8765
Reads data/state.json + data/events.jsonl written by runner.py and polls nvidia-smi every 2 s."""
import argparse, json, os, shutil, subprocess, threading, time, urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

GPU = []; GPU_LOCK = threading.Lock(); SMI = shutil.which("nvidia-smi")

def gpu_loop():
    while True:
        s = {"t": time.time()}
        if SMI:
            try:
                o = subprocess.run([SMI, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip().splitlines()[0]
                n, u, mu, mt, tp, pw = [x.strip() for x in o.split(",")]
                f = lambda v: float(v) if v.replace(".", "", 1).isdigit() else None
                s.update(name=n, util=f(u), mem_used=f(mu), mem_total=f(mt), temp=f(tp), power=f(pw))
            except Exception as e:
                s["err"] = str(e)[:100]
        with GPU_LOCK:
            GPU.append(s); del GPU[:-1800]
        time.sleep(2)

class Agg:
    """Incrementally tails events.jsonl (20k+ lines) so each refresh is cheap."""
    def __init__(self, data):
        self.p = os.path.join(data, "events.jsonl"); self.off = 0; self.lock = threading.Lock()
        self.sheets = self.errors = self.spr = self.use = 0; self.last_err = None
        self.first_t = None; self.buckets = {}; self.recent = []; self.latest = []
    def update(self):
        if not os.path.exists(self.p): return
        with self.lock, open(self.p, "rb") as f:
            f.seek(self.off); chunk = f.read(); cut = chunk.rfind(b"\n") + 1; self.off += cut
            for line in chunk[:cut].splitlines():
                try: e = json.loads(line)
                except ValueError: continue
                if "error" in e: self.errors += 1; self.last_err = e["error"]; continue
                self.sheets += 1; self.spr += e["sprites"]; self.use += e["usable"]
                if self.first_t is None: self.first_t = e["t"] - e.get("gen_s", 0)
                k = int(e["t"] // 10) * 10; self.buckets[k] = self.buckets.get(k, 0) + e["sprites"]
                self.recent.append((e["t"], e["sprites"])); self.latest = (e.get("cells", []) + self.latest)[:48]
            if len(self.buckets) > 2000:
                for k in sorted(self.buckets)[:-2000]: del self.buckets[k]
            now = time.time(); self.recent = [r for r in self.recent if r[0] >= now - 900]
AGG = {}

def stats(data):
    try: st = json.load(open(os.path.join(data, "state.json")))
    except Exception: st = {"status": "not started"}
    a = AGG.setdefault(data, Agg(data)); a.update(); now = time.time()
    with a.lock:
        def rate(win):  # sprites/s over the last `win` seconds of this session
            rec = [n for t, n in a.recent if t >= now - win]
            span = min(win, now - min((t for t, _ in a.recent), default=now) + 1e-6)
            return sum(rec) / max(span, 1.0) if rec else 0.0
        sess0 = st.get("started", a.first_t or now)
        end = st.get("finished", now) if str(st.get("status", "")).startswith("finished") else now
        span = max(end - max(sess0, a.first_t or sess0), 1e-6)
        sess = [n for t, n in a.recent]  # avg over last 15 min window (robust to resume gaps)
        series = []
        if a.buckets:
            ks = sorted(a.buckets); k0 = max(ks[0], ks[-1] - 3600)
            series = [[k, a.buckets.get(k, 0) / 10] for k in range(k0, ks[-1] + 10, 10)]
        per = max(st.get("per_sheet", 4), 1); total = st.get("total_sheets", 0); done = st.get("done_sheets", 0)
        r15 = rate(900); eta = (total - done) * per / r15 if r15 > 0 and total > done else None
        spr_avg = a.spr / max(now - a.first_t, 1e-6) if a.first_t and not a.recent else rate(900)
        with GPU_LOCK: g = list(GPU[-900:])
        return {"state": st, "sheets": a.sheets, "errors": a.errors, "sprites": a.spr, "usable": a.use,
                "sprites_per_s_now": round(rate(60), 3), "sprites_per_s_avg": round(spr_avg, 3),
                "usable_per_s_avg": round(spr_avg * (a.use / a.spr if a.spr else 0), 3),
                "sheets_per_min": round(rate(60) * 60 / per, 2), "elapsed_s": round(now - sess0),
                "eta_s": round(eta) if eta else None, "series": series[-360:], "gpu": g, "latest": a.latest,
                "last_error": a.last_err, "now": now}

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Krea 2 sprites</title><style>
body{font:14px system-ui,Segoe UI,sans-serif;background:#111418;color:#e6e6e6;margin:0;padding:18px}
h1{font-size:18px;margin:0 0 4px}.sub{color:#8a94a0;margin-bottom:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.c{background:#1b2027;border-radius:8px;padding:10px 12px}.c b{display:block;font-size:22px;margin-top:3px}
.c span{color:#8a94a0;font-size:12px}.row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
.panel{background:#1b2027;border-radius:8px;padding:10px 12px}.panel h2{font-size:13px;margin:0 0 6px;color:#aab}
.bar{height:10px;background:#2a313b;border-radius:5px;overflow:hidden;margin:6px 0}.bar i{display:block;height:100%;background:#4cc38a}
.grid{display:grid;grid-template-columns:repeat(auto-fill,72px);gap:6px}.grid div{background:repeating-conic-gradient(#2a313b 0 25%,#222831 0 50%) 0 0/12px 12px;border-radius:4px;position:relative}
.grid img{width:64px;height:64px;image-rendering:pixelated;display:block;margin:4px}.grid .bad{outline:2px solid #c24}
svg{width:100%;height:150px}.err{color:#f77;font-size:12px;white-space:pre-wrap}.tag{background:#c80;color:#000;padding:1px 6px;border-radius:4px;font-size:12px}
</style></head><body><h1>Krea 2 sprite generator <span id="mode"></span></h1><div class="sub" id="sub">loading…</div>
<div class="cards" id="cards"></div>
<div class="panel" style="margin-top:10px"><h2>Queue</h2><div class="bar"><i id="pb" style="width:0"></i></div><div id="q"></div></div>
<div class="row"><div class="panel"><h2>Sprites/s (10 s buckets)</h2><svg id="ch1"></svg></div>
<div class="panel"><h2>GPU util % (green) · VRAM % (blue) · temp °C (orange)</h2><svg id="ch2"></svg></div></div>
<div class="panel" style="margin-top:10px"><h2>Latest sprites (32×32, shown 2×; red outline = auto-rejected)</h2><div class="grid" id="g"></div><div class="err" id="e"></div></div>
<script>
const $=id=>document.getElementById(id);
const fmt=s=>s==null?'–':s<90?Math.round(s)+' s':s<5400?Math.round(s/60)+' min':(s/3600).toFixed(1)+' h';
function line(svg,sets,ymax){const W=600,H=150,P=28;let x0=Infinity,x1=-Infinity;sets.forEach(s=>s.pts.forEach(p=>{x0=Math.min(x0,p[0]);x1=Math.max(x1,p[0])}));
 if(!isFinite(x0)){svg.innerHTML='<text x="10" y="80" fill="#667">no data yet</text>';return}
 if(x1==x0)x1=x0+1;let ym=ymax||Math.max(1e-6,...sets.flatMap(s=>s.pts.map(p=>p[1])))*1.15;
 const X=x=>P+(x-x0)/(x1-x0)*(W-P-6),Y=y=>H-16-(y/ym)*(H-26);let h=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
 for(let k=0;k<=4;k++){const y=ym*k/4;h+=`<line x1="${P}" x2="${W}" y1="${Y(y)}" y2="${Y(y)}" stroke="#2a313b"/><text x="0" y="${Y(y)+4}" fill="#667" font-size="10">${y<10?y.toFixed(1):Math.round(y)}</text>`}
 sets.forEach(s=>{if(s.pts.length)h+=`<polyline fill="none" stroke="${s.c}" stroke-width="2" points="${s.pts.map(p=>X(p[0])+','+Y(p[1])).join(' ')}"/>`});
 h+=`<text x="${P}" y="${H-2}" fill="#667" font-size="10">${new Date(x0*1000).toLocaleTimeString()}</text><text x="${W-60}" y="${H-2}" fill="#667" font-size="10">${new Date(x1*1000).toLocaleTimeString()}</text>`;
 svg.outerHTML=h.replace('<svg','<svg id="'+svg.id+'"')+'</svg>'}
async function tick(){try{const d=await (await fetch('/api/stats')).json();const st=d.state,g=d.gpu.length?d.gpu[d.gpu.length-1]:{};
 $('mode').innerHTML=st.mode=='replay-test'?'<span class="tag">TEST REPLAY – not GPU timings</span>':'';
 $('sub').textContent=`status: ${st.status||'?'} · config: ${st.config?`${st.config.grid}x${st.config.grid} @ ${st.config.size}, ${st.config.steps} steps, ${st.config.px}px, LoRA ${st.config.lora} x${st.config.lora_mult}, ${st.config.in_flight} in flight`:''} · GPU: ${g.name||'nvidia-smi not found'}`;
 const cards=[['Sprites total',d.sprites],['Usable (auto)',d.usable+(d.sprites?` (${Math.round(100*d.usable/d.sprites)}%)`:'')],['Sprites/s now (60 s)',d.sprites_per_s_now.toFixed(2)],['Sprites/s avg (15 min)',d.sprites_per_s_avg.toFixed(2)],['Usable/s avg (15 min)',d.usable_per_s_avg.toFixed(2)],['Sheets/min now',d.sheets_per_min],['Sheets done',`${d.sheets}`],['Errors',d.errors],['GPU util',g.util!=null?g.util+' %':'–'],['VRAM',g.mem_used!=null?(g.mem_used/1024).toFixed(1)+' / '+(g.mem_total/1024).toFixed(1)+' GB':'–'],['GPU temp',g.temp!=null?g.temp+' °C':'–'],['Power',g.power!=null?Math.round(g.power)+' W':'–'],['Session time',fmt(d.elapsed_s)],['ETA',fmt(d.eta_s)]];
 $('cards').innerHTML=cards.map(c=>`<div class="c"><span>${c[0]}</span><b>${c[1]}</b></div>`).join('');
 const tot=st.total_sheets||0,dn=st.done_sheets||0;$('pb').style.width=(tot?100*dn/tot:0)+'%';$('q').textContent=`${dn.toLocaleString()} / ${tot.toLocaleString()} sheets (${(dn*(st.per_sheet||4)).toLocaleString()} / ${(tot*(st.per_sheet||4)).toLocaleString()} sprites, queue = queue.jsonl) · ETA ${fmt(d.eta_s)}`;
 line($('ch1'),[{pts:d.series,c:'#4cc38a'}]);
 line($('ch2'),[{pts:d.gpu.filter(x=>x.util!=null).map(x=>[x.t,x.util]),c:'#4cc38a'},{pts:d.gpu.filter(x=>x.mem_used!=null).map(x=>[x.t,100*x.mem_used/x.mem_total]),c:'#5b9cf0'},{pts:d.gpu.filter(x=>x.temp!=null).map(x=>[x.t,x.temp]),c:'#f0a050'}],100);
 $('g').innerHTML=d.latest.map(s=>`<div class="${s.usable?'':'bad'}" title="${s.subject.replace(/"/g,'')}"><img src="/sprite/${encodeURI(s.file)}"></div>`).join('');
 $('e').textContent=d.last_error?'last error: '+d.last_error:'';}catch(e){$('sub').textContent='dashboard error: '+e}}
tick();setInterval(tick,2000);
</script></body></html>"""

def make_handler(data):
    sprites = os.path.realpath(os.path.join(data, "sprites32"))
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def send(self, code, body, ctype):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
            if path == "/": return self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/api/stats": return self.send(200, json.dumps(stats(data)).encode(), "application/json")
            if path.startswith("/sprite/"):
                p = os.path.realpath(os.path.join(sprites, path[8:]))
                if p.startswith(sprites + os.sep) and os.path.isfile(p):
                    return self.send(200, open(p, "rb").read(), "image/png")
            self.send(404, b"not found", "text/plain")
    return H

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--port", type=int, default=8765); a = ap.parse_args()
    data = os.path.abspath(a.data); os.makedirs(data, exist_ok=True)
    threading.Thread(target=gpu_loop, daemon=True).start()
    print(f"dashboard: http://127.0.0.1:{a.port}  (data: {data})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(data)).serve_forever()

if __name__ == "__main__":
    main()
