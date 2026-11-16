"""Pixel AI chat generation worker — v3 COND (palette/style cond, copy of v3fast).

Copy of the live v3 worker's queue/output contract (CFG 2.0, temp 0.8, top-k 32,
reward-v7 picker, 64 candidates → store 20 → show 4). Lives in chat-worker-v3fast/
and does not edit the live chat-worker-v3/ tree or any checkpoint.

Speed path: vectorized Gumbel pick, static KV cache, CUDA graph of the cached
trunk step, one batch of 64, CUDA-batched reward-v7. --selftest loads everything
and renders one prompt without binding the lock port or polling the app.
If hold_file exists after READY, wait until it is removed before polling.
"""
from __future__ import annotations

import base64, io, json, os, secrets, socket, sys, time, traceback, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG_PATH = Path(os.environ.get("PIXEL_WORKER_CONFIG") or (HERE / "config_v3.json"))
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))
def _load_worker_key() -> str:
    env = (os.environ.get("CHAT_WORKER_KEY") or "").strip()
    if env:
        return env
    path = HERE / "worker.key"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    raise SystemExit("missing CHAT_WORKER_KEY or worker.key")

KEY = _load_worker_key()
URLS = CFG["app_urls"]
CKPT = Path(CFG["ckpt"])
MODEL_KEY = CFG.get("model_key") or "v3"
SCALE = int(CFG.get("scale", 16))
SCORER_CFG = (CFG.get("scorer") or "feat_linear").strip()
N_SAMPLES_V7 = int(CFG.get("n_samples", 64))
N_SHOW_DEFAULT = int(CFG.get("n_show", 4))
CFG_SCALE = float(CFG.get("cfg_scale", 2.0))
TEMP = float(CFG.get("temperature", 0.8))
TOPK = int(CFG.get("top_k", 32))
WANT_SHA = CFG.get("ckpt_sha256") or ""
HOLD = Path(CFG["hold_file"]) if CFG.get("hold_file") else None
USE_GRAPH = bool(int(os.environ.get("PIXEL_FAST_GRAPH", "1")))

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CFG.get("py_extra") or "/path/to/pixie"))
sys.path.insert(0, str(CFG["v3_code"]))

os.environ.setdefault("HF_HOME", str(Path(CFG["v7_root"]) / "hf-cache"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(CFG["v7_root"]) / "hf-cache" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(CFG["v7_root"]) / "hf-cache" / "transformers"))
os.environ.setdefault("OPENCLIP_CACHE_DIR", str(Path(CFG["v7_root"]) / "hf-cache" / "openclip"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image
import v3_model as M
from tokenizers import Tokenizer

from fast_sample import FastSampler
from fast_v7 import FastRewardV7


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def tload(path):
    import pathlib
    saved = (getattr(pathlib, "WindowsPath", None), pathlib.PosixPath)
    pathlib.WindowsPath = pathlib.PureWindowsPath
    pathlib.PosixPath = pathlib.PurePosixPath
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        if saved[0] is not None:
            pathlib.WindowsPath = saved[0]
        pathlib.PosixPath = saved[1]


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# feat_linear.json (a two-feature logistic fit on private preference votes) is not
# shipped. Point PIXEL_FEAT_LINEAR at your own fit; without one the fallback scorer
# is neutral and every candidate scores 0.
_FL_PATH = Path(os.environ.get("PIXEL_FEAT_LINEAR") or HERE / "feat_linear.json")
FL = (json.loads(_FL_PATH.read_text()) if _FL_PATH.is_file()
      else {"mu": [0.0, 0.0], "sd": [1.0, 1.0], "w": [0.0, 0.0, 0.0]})
MU, SD, W = np.array(FL["mu"]), np.array(FL["sd"]), np.array(FL["w"])


def feat_linear(idx):
    a = np.asarray(idx)
    fill = float((a != 0).sum())
    ncol = float(len(set(a.tolist()) - {0}))
    z = (np.array([fill, ncol]) - MU) / SD
    return float(z @ W[:2] + W[2]), int(fill), int(ncol)


def rgba_from_indices(ids, palette):
    ids = np.asarray(ids, dtype=np.int64).reshape(16, 16)
    rgb = palette[ids]
    if rgb.shape[-1] == 4:
        rgba = rgb.copy()
        rgba[..., 3] = np.where(ids != 0, rgba[..., 3] if rgba[..., 3].max() > 0 else 255, 0)
        if rgba[..., 3].max() == 0:
            rgba[..., 3] = np.where(ids != 0, 255, 0)
    else:
        a = np.where(ids != 0, 255, 0).astype(np.uint8)
        rgba = np.dstack([rgb.astype(np.uint8), a])
    return rgba.astype(np.uint8)


def try_load_v7(palette):
    v7_root = Path(CFG.get("v7_root") or "/path/to/pixie")
    v7_src = v7_root / "src"
    v7_site = v7_root / ".venv" / "lib"
    for cand in sorted(v7_site.glob("python*/site-packages")) if v7_site.exists() else []:
        if str(cand) in sys.path:
            sys.path.remove(str(cand))
        sys.path.insert(0, str(cand))
    if str(v7_src) in sys.path:
        sys.path.remove(str(v7_src))
    sys.path.insert(0, str(v7_src))
    from scorer import RewardV7
    device = CFG.get("v7_device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    head = Path(CFG.get("v7_head") or (v7_root / "final_head.json"))
    scorer = RewardV7(head, palette, device=device)
    log("V7 loaded", head, "device", device, "models", list(scorer.models), "dim", len(scorer.w) - 1)
    return FastRewardV7(scorer, palette)


def clip_ids(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


def t26(tok, text):
    ids = clip_ids(tok, text)[:24]
    return [M.CLIP_BOS] + ids + [M.CLIP_EOS] * (25 - len(ids))


def trim(row, tl):
    a = list(row[:tl])
    n = sum(1 for x in row[1:] if x != M.CLIP_EOS)
    if n > tl - 2:
        a[tl - 1] = M.CLIP_EOS
    return a


class Gen:
    def __init__(self):
        self.palette = np.load(CFG["palette"], allow_pickle=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() and CFG.get("device", "cuda") == "cuda" else "cpu")
        sha = sha256(CKPT)
        if WANT_SHA and sha != WANT_SHA:
            raise SystemExit(f"v3 ckpt sha256 {sha} != config {WANT_SHA}")
        ck = tload(CKPT)
        c = M.V3Config(**ck["config"])
        self.model = M.build(c)
        miss, unexp = self.model.load_state_dict(ck["model"], strict=False)
        assert not unexp and set(miss) <= {"pr", "pc", "vmask"}, (miss, unexp)
        step, images, epoch = ck.get("step"), ck.get("images"), ck.get("epoch")
        cfg_saved = ck["config"]
        del ck
        self.model.to(self.device).eval()
        self.TL = int(self.model.config.text_len)
        self.tok = Tokenizer.from_file(str(CFG["clip_tokenizer"]))
        self.clip = M.ClipTextPenultimate()
        self.clip.load_hf(tload(CFG["clip_pt"]))
        self.clip = self.clip.to(self.device).to(torch.bfloat16).eval()
        self.info = {
            "key": MODEL_KEY, "model": "v3", "arch": "v3",
            "step": step, "images": images, "epoch": epoch,
            "sha": sha, "cfg_scale": CFG_SCALE,
            "text_len": self.TL, "n_pal": int(getattr(c, "n_pal", 0)),
            "style": bool(getattr(c, "style", False)),
            "run": CKPT.parent.name, "file": CKPT.name,
            "config": cfg_saved,
            "fast": True, "fast_graph": USE_GRAPH,
        }
        self.v7 = None
        self.scorer_name = "feat_linear"
        if SCORER_CFG == "reward_v7":
            try:
                self.v7 = try_load_v7(self.palette)
                self.scorer_name = "reward_v7"
            except Exception:
                traceback.print_exc()
                log("V7 failed to load, falling back to feat_linear")
                self.v7 = None
                self.scorer_name = "feat_linear"
        max_n = int(os.environ.get('PIXEL_MAX_N') or max(N_SAMPLES_V7, 64))
        self.sampler = FastSampler(
            self.model, self.device, max_batch=max_n * 2,
            cfg_scale=CFG_SCALE, temperature=TEMP, top_k=TOPK, use_graph=USE_GRAPH,
        )
        if USE_GRAPH and self.device.type == "cuda":
            # Capture the live size (and 16 for the quality test) once at startup.
            sizes = sorted({max_n * 2, 16 * 2})
            t0 = time.time()
            self.sampler.prepare(sizes)
            log("CUDA graphs ready", "batches", sizes, f"{(time.time()-t0)*1000:.0f}ms")
        log("MODEL", MODEL_KEY, "step", step, "images", images, "sha", sha[:12],
            "text_len", self.TL, "device", self.device, "scorer", self.scorer_name,
            "fast", True, "graph", USE_GRAPH)

    def _encode(self, rows):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return self.clip(torch.tensor(rows, dtype=torch.long, device=self.device))

    def _scores(self, prompt, idxs):
        fills, ncols = [], []
        for idx in idxs:
            sc, fill, ncol = feat_linear(idx)
            fills.append(fill)
            ncols.append(ncol)
        if self.v7 is not None:
            try:
                v7s = self.v7.score_indices(idxs, prompt)
                return v7s, fills, ncols, "reward_v7"
            except Exception:
                traceback.print_exc()
                log("V7 score failed, falling back to feat_linear for this job")
        scs = []
        for fill, ncol in zip(fills, ncols):
            z = (np.array([fill, ncol], dtype=np.float32) - MU) / SD
            scs.append(float(z @ W[:2] + W[2]))
        return np.array(scs, dtype=np.float32), fills, ncols, "feat_linear"

    def cond_tensor(self, cond, n):
        """Build a [n, n_pal+3] long tensor, or None to use the sampler default (NO_PALETTE + ANY).
        Missing/invalid cond is treated as default so this stays backwards compatible with old jobs."""
        c = self.model.config
        if not getattr(c, "n_pal", 0) and not getattr(c, "style", False):
            return None
        if not cond:
            return None
        pal = cond.get("palette") if isinstance(cond, dict) else None
        sty = cond.get("style") if isinstance(cond, dict) else None
        try:
            if pal is None:
                pids = [M.NO_PALETTE] * int(c.n_pal)
            else:
                pids = [int(x) for x in list(pal)[: int(c.n_pal)]]
                while len(pids) < int(c.n_pal):
                    pids.append(M.NO_PALETTE)
                for x in pids:
                    if not (x == M.NO_PALETTE or 1 <= x <= 4096):
                        return None
            if sty is None:
                tags = [0, 0, 0]
            else:
                tags = [int(x) for x in list(sty)[:3]]
                while len(tags) < 3:
                    tags.append(0)
                if not (0 <= tags[0] <= 3 and 0 <= tags[1] <= 2 and 0 <= tags[2] <= 3):
                    return None
            t = torch.tensor([pids + tags], dtype=torch.long, device=self.device).expand(n, -1).contiguous()
            return t
        except Exception:
            return None

    def run(self, prompt, n, show, ncand=0, cond=None):
        t0 = time.time()
        seeds = [secrets.randbelow(2**31 - 1) for _ in range(n)]
        row = trim(t26(self.tok, prompt), self.TL)
        blank = [M.CLIP_BOS] + [M.CLIP_EOS] * (self.TL - 1)
        h = self._encode([row] * n)
        hu = self._encode([blank] * n)
        seed_t = torch.tensor(seeds, dtype=torch.int64, device=self.device)
        ct = self.cond_tensor(cond, n)
        with torch.no_grad():
            img = self.sampler.sample(h, hu, seeds=seed_t, cond=ct)
        ind = img.cpu().numpy()
        idxs = [np.asarray(r, dtype=np.int64).reshape(-1) for r in ind]
        sprites = [rgba_from_indices(ix, self.palette) for ix in idxs]
        t_gen = time.time() - t0
        scs, fills, ncols, used = self._scores(prompt, idxs)
        rows = []
        for i, (sd, sc, fill, ncol, sp) in enumerate(zip(seeds, scs, fills, ncols, sprites)):
            rows.append(dict(i=i, seed=sd, score=round(float(sc), 5), fill=int(fill), ncolors=int(ncol),
                             sprite=np.asarray(sp, dtype=np.uint8)))
        rows.sort(key=lambda r: -r["score"])

        def _png(r):
            im = Image.fromarray(r["sprite"]).resize((16 * SCALE, 16 * SCALE), Image.NEAREST)
            buf = io.BytesIO()
            im.save(buf, "PNG", optimize=True)
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        out = []
        for r in rows[:show]:
            out.append(dict(seed=r["seed"], score=r["score"], fill=r["fill"], ncolors=r["ncolors"], png=_png(r)))
        self.last_candidates = [
            dict(idx=r["i"], rank=k + 1, seed=r["seed"], score=r["score"],
                 png=(out[k]["png"] if k < len(out) else _png(r)))
            for k, r in enumerate(rows[:ncand])
        ]
        ms = (time.time() - t0) * 1000
        return out, [dict(seed=r["seed"], score=r["score"]) for r in rows], ms, t_gen * 1000, used


def selftest():
    g = Gen()
    for prompt in ("red potion", "desrt dragon", "iron anchor"):
        res, allr, ms, gen_ms, used = g.run(prompt, N_SAMPLES_V7, N_SHOW_DEFAULT, 20)
        log("SELFTEST", repr(prompt), f"{ms:.0f}ms (sampling {gen_ms:.0f}ms)",
            "scorer", used, "n", len(allr), "candidates", len(g.last_candidates),
            "top", [x["score"] for x in res])
        out = HERE / f"selftest_v3_{prompt.replace(' ', '_')}.png"
        ims = [Image.open(io.BytesIO(base64.b64decode(r["png"].split(",", 1)[1]))) for r in res]
        sheet = Image.new("RGBA", (sum(i.width for i in ims) + 8 * (len(ims) - 1), ims[0].height), (229, 231, 228, 255))
        x = 0
        for i in ims:
            sheet.paste(i, (x, 0), i)
            x += i.width + 8
        sheet.save(out)
        log("wrote", out)
    log("SELFTEST_OK", json.dumps({k: g.info[k] for k in g.info if k != "config"}))


def post(path, body, timeout):
    last = None
    for base in URLS:
        try:
            rq = urllib.request.Request(
                base + path, data=json.dumps(body).encode(), method="POST",
                headers={"Content-Type": "application/json", "X-Worker-Key": KEY, "User-Agent": "pixel-chat-worker/3fast"},
            )
            with urllib.request.urlopen(rq, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            last = e
    raise last


def main():
    global _LOCK
    _LOCK = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _LOCK.bind(("127.0.0.1", int(CFG.get("lock_port", 8794))))
    except OSError:
        log("another worker is already running, exiting")
        sys.exit(3)
    g = Gen()
    info = {"model": g.info, "device": str(g.device), "host": socket.gethostname(), "scorer": g.scorer_name}
    g.run("red potion", 8, 1, 0)
    log("READY urls", URLS, "scorer", g.scorer_name, "n_samples", N_SAMPLES_V7 if g.scorer_name == "reward_v7" else 20,
        "fast", True, "graph", USE_GRAPH)
    if HOLD is not None and HOLD.exists():
        log("holding polls until", HOLD, "is removed")
        while HOLD.exists():
            time.sleep(0.4)
        log("hold released, starting polls")
    fails = 0
    while True:
        try:
            r = post("/api/worker/next", {"info": info}, timeout=40)
            fails = 0
            job = r.get("job")
            if not job:
                continue
            log("JOB", job["id"], repr(job["prompt"]), "cond", job.get("cond") or "default")
            if g.scorer_name == "reward_v7":
                n = N_SAMPLES_V7
            else:
                n = int(job.get("n", 20))
            show = int(job.get("show", N_SHOW_DEFAULT))
            try:
                ncand = int(job.get("n", 20))
                res, allr, ms, gen_ms, used = g.run(job["prompt"], n, show, ncand, cond=job.get("cond"))
                body = {"id": job["id"], "ok": True, "results": res, "all": allr, "ms": ms,
                        "model": g.info, "scorer": used, "candidates": g.last_candidates}
                log("DONE", job["id"], f"{ms:.0f}ms (sampling {gen_ms:.0f}ms)", "scorer", used,
                    "n", n, "cand", len(g.last_candidates), "top", [x["score"] for x in res])
            except Exception:
                traceback.print_exc()
                body = {"id": job["id"], "ok": False}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            for _ in range(3):
                try:
                    post("/api/worker/result", body, timeout=30)
                    break
                except Exception as e:
                    log("result post failed", e)
                    time.sleep(1)
        except Exception as e:
            fails += 1
            log("poll error", type(e).__name__, str(e)[:120])
            time.sleep(min(30, 2 * fails))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--grid" in sys.argv or "--bench" in sys.argv:
        from cond_grid import main as grid_main
        grid_main()
    else:
        main()
