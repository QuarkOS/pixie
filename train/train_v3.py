"""Pixie v3 trainer: arch v2ft (fine-tune the v2 SpriteV2), pix (pilot a), p50 (pilot b).

GPU-resident data (images int16, CLIP ids as int16 bit patterns), caption variants picked
on the GPU per batch (10% blank -> CFG, 17.5% subject-as-caption, 10% synonym where one
exists), h-flip except directional captions, fused AdamW, bf16 autocast, cosine LR sized
by --images. Eval in fp32 on the v3 probe (100 unseen subjects) and iid split, by bucket,
with per-slot NLL (2x2 position) at fixed image counts so runs compare at equal images seen.
metrics.jsonl keys follow the v2 trainer so the training pusher can read it.

s7 (Opus roadmap s7, all opt-in; defaults keep the old model definition):
  --text-len 18      CLIP text trimmed to BOS + 16 ids + EOS (causal CLIP: identical features for captions that fit)
  --cond             3 palette slots (E_pix + PALETTE_COND type emb) + 1 style slot (fill/outline/colour-count tags)
  --balance-alpha    rare-subject sampling p ~ n_s^-alpha (capped --balance-cap x), same views per epoch
  --schedule wsd     warmup -> constant LR; --save-at-images keeps permanent full checkpoints (full_XXXM.pt)
  --schedule decay   linear LR -> lr-min over the run (decay branch, --branch-from a full checkpoint)
  --ce-chunks N      CE in patch order, W_out + log-softmax per chunk under checkpointing (no raster logits)
  --clip-cache DIR   stream precomputed fp16 CLIP features (build_clip_cache_s7.py) with prefetch threads
Captions/flip/cond for a whole epoch are drawn on the GPU at epoch start (one generator per epoch, so resume
is exact); every batch has exactly round(caption_dropout*B) blank rows at the end, CLIP runs only on the rest.
"""
import argparse, glob, hashlib, json, math, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3_model as M  # noqa: E402
import s7_data as D  # noqa: E402

CB_NAMES = ["rare_lt1k", "mid_1k_5k", "common_gt5k"]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def u16(t):  # int16 bit pattern -> long CLIP id
    return t.long() & 0xFFFF


def as_i16(a):
    return torch.from_numpy(np.ascontiguousarray(a).view(np.int16))


def to_v2_text(t):
    """[B,26] CLIP-format ids -> v2 format [B,24]: ids between BOS and first EOS, +1, zero-padded."""
    ids = t[:, 1:25]
    is_eos = (t[:, 1:] == M.CLIP_EOS)
    n = torch.where(is_eos.any(1), is_eos.float().argmax(1), torch.full_like(is_eos[:, 0], 25, dtype=torch.long))
    valid = torch.arange(24, device=t.device)[None] < n[:, None]
    return torch.where(valid, ids + 1, torch.zeros_like(ids))


def empty_text(b, device, text_len=M.TEXT_LEN):
    t = torch.full((b, text_len), M.CLIP_EOS, dtype=torch.long, device=device)
    t[:, 0] = M.CLIP_BOS
    return t


class Net:
    """Uniform wrapper: logits(text26, img) -> [B,256,V] raster; sample(text26, cfg)."""

    def __init__(self, a, dev):
        self.arch = a.arch; self.dev = dev
        if a.arch == "v2ft":
            import v2_model as V2
            self.V2 = V2
            ck = torch.load(a.init, map_location="cpu", weights_only=False)
            sd = ck["model"]
            cfg = V2.V2Config(**{**ck.get("config", {}), "dropout": a.dropout}) if ck.get("config") else \
                V2.V2Config(d_model=a.width, n_layer=a.layers, n_head=a.heads, dropout=a.dropout,
                            resid_vocab=int(sd["text_resid.weight"].shape[0]))
            clip2v2 = sd["clip2v2"].numpy()
            self.model = V2.SpriteV2(cfg, np.load(a.v2_clip_table), clip2v2).to(dev)
            miss, unexp = self.model.load_state_dict(sd, strict=False)
            assert not unexp and set(miss) <= {"clip_table", "pr", "pc"}, (miss, unexp)
            self.init_info = {"init": str(a.init), "init_step": ck.get("step"), "init_best_dev": ck.get("best_dev")}
            self.clip = None
        else:
            cfg = M.V3Config(arch=a.arch, d_model=a.width, n_layer=a.layers, n_head=a.heads, dropout=a.dropout,
                             text_len=a.text_len, n_pal=3 if a.cond else 0, style=bool(a.cond))
            self.model = M.build(cfg).to(dev)
            self.clip = M.ClipTextPenultimate().to(dev)
            self.clip.load_hf(torch.load(a.clip, map_location="cpu", weights_only=False))
            self.clip.to(torch.bfloat16 if dev.type == "cuda" else torch.float32).eval()
            self.init_info = {}
        self.cfg = self.model.config
        self.TL = a.text_len

    def text_in(self, t26):
        if self.arch == "v2ft":
            return to_v2_text(t26)
        with torch.no_grad(), torch.autocast(device_type=self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            return self.clip(t26)

    def logits(self, t26, img, cond=None):
        x = self.text_in(t26)
        return self.model(x, img) if (cond is None or self.arch != "p50") else self.model(x, img, cond)

    def losses(self, lg, img, aw):
        return M.losses(lg, img, aw) if self.arch != "v2ft" else self.V2.losses(lg, img, aw)

    def sample(self, t26, cfg, generator, cond=None):
        if self.arch == "v2ft":
            return self.V2.sample(self.model, to_v2_text(t26), temperature=0.8, top_k=32, cfg=cfg, generator=generator)
        self.model.eval()
        h = self.text_in(t26); hu = self.text_in(empty_text(t26.shape[0], t26.device, t26.shape[1])) if cfg != 1.0 else None
        kw = {"cond": cond} if cond is not None else {}
        out = self.model.sample(h, hu, cfg=cfg, temperature=0.8, top_k=32, generator=generator, **kw)
        self.model.train()
        return out


@torch.no_grad()
def evaluate(net, E, bs=64, uncond=True):
    """fp32 eval. E: dict name -> (text long, img long, bucket int or None[, cond long or None])."""
    m = net.model; m.eval()
    slot = M.slot_of_pixel(next(m.parameters()).device)
    res = {}
    for name, ent in E.items():
        T, I, B = ent[:3]; C = ent[3] if len(ent) > 3 else None
        tot = torch.zeros((), dtype=torch.float64, device=I.device); per_row = []
        slots = torch.zeros(4, dtype=torch.float64, device=I.device); tot_u = 0.0
        for s in range(0, I.shape[0], bs):
            t = T[s:s + bs]; im = I[s:s + bs]
            lg = net.logits(t, im, None if C is None else C[s:s + bs]).float()
            _, nll, _, _ = net.losses(lg, im, 0.0)
            per_row.append(nll.mean(1).double())
            for k in range(4):
                slots[k] += nll[:, slot == k].double().sum()
            if uncond and name == "probe":
                _, nu, _, _ = net.losses(net.logits(empty_text(t.shape[0], t.device, t.shape[1]), im).float(), im, 0.0)
                tot_u += float(nu.double().sum())
        pr = torch.cat(per_row); n = I.shape[0]
        res[f"{name}_nll"] = float(pr.mean())
        res[f"{name}_slot_nll"] = [float(x) for x in (slots / (n * 64))]
        if B is not None:
            for k, cb in enumerate(CB_NAMES):
                sel = B == k
                if bool(sel.any()):
                    res[f"{name}_{cb}"] = float(pr[sel].mean())
        if uncond and name == "probe":
            res["probe_nll_uncond"] = tot_u / (n * 256)
    m.train()
    return res


def write_grid(ids, names, per, palette, path):
    from PIL import Image, ImageDraw
    sc, pad = 6, 4
    W = per * (16 * sc + pad) + pad + 120; H = len(names) * (16 * sc + pad) + pad
    sheet = Image.new("RGB", (W, H), (255, 255, 255)); dr = ImageDraw.Draw(sheet)
    chk = ((np.indices((16, 16)).sum(0) % 2) * 14 + 238).astype(np.uint8)
    for i, row in enumerate(ids):
        r, c = divmod(i, per)
        rgb = palette[row].reshape(16, 16, 3); alpha = (row != 0).reshape(16, 16)
        out = np.where(alpha[..., None], rgb, np.stack([chk] * 3, -1)).astype(np.uint8)
        x0 = 120 + pad + c * (16 * sc + pad); y0 = pad + r * (16 * sc + pad)
        if c == 0:
            dr.text((4, y0 + 40), names[r][:19], fill=(0, 0, 0))
        sheet.paste(Image.fromarray(out).resize((16 * sc, 16 * sc), Image.NEAREST), (x0, y0))
    Path(path).parent.mkdir(parents=True, exist_ok=True); sheet.save(path)
    np.save(str(path)[:-4] + "_ids.npy", np.asarray(ids, np.int16))


def do_samples(net, prompts, palette, outdir, real_img, dev, per=8, seed=1234):
    g = torch.Generator(device=dev); g.manual_seed(seed)
    rep = {}; allg = []
    for group in ("rare", "panel10"):
        names = list(prompts[group])
        t26 = np.array([prompts[group][p] for p in names for _ in range(per)], dtype=np.int64).astype(np.uint16)
        t = torch.from_numpy(D.trim(t26, net.TL).astype(np.int64)).to(dev)
        for cfg in (1.0, 2.0):
            ids = net.sample(t, cfg, g)
            allg.append(ids)
            write_grid(ids.cpu().numpy(), names, per, palette, Path(outdir) / f"{group}_x{per}_cfg{cfg:g}.png")
    import shutil  # the training pusher looks for this name
    shutil.copyfile(Path(outdir) / f"panel10_x{per}_cfg2.png", Path(outdir) / "panel10x8_cfg2.png")
    gen = torch.cat(allg)
    rg, gn, gd = M.boundary_ratio(gen); rr, rn, rd = M.boundary_ratio(real_img)
    rep.update({"boundary_ratio_gen": rg, "boundary_ratio_real": rr, "boundary_artifact_ratio": rg / rr,
                "gen_diff_cross": gn, "gen_diff_inside": gd, "real_diff_cross": rn, "real_diff_inside": rd,
                "n_samples": int(gen.shape[0])})
    return rep


class FeatPrefetch:
    """Streams cached CLIP features [nt, L, 768] fp16 for future steps into a ring of pinned buffers
    (os.preadv straight into pinned memory, several threads -> SSD queue depth > 1), then async H2D."""

    def __init__(self, path, L, nt, cids_for, total_steps, dev, ring=12, workers=6):
        self.fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        self.row = L * 768 * 2; self.cids_for = cids_for; self.total = total_steps; self.dev = dev; self.ring = ring
        self.buf = torch.empty((ring, nt, L, 768), dtype=torch.float16).pin_memory()
        self.np = self.buf.numpy().view(np.uint8); self.ev = [None] * ring; self.fut = {}
        self.ex = ThreadPoolExecutor(workers); self.wait_s = 0.0; self.n = 0

    def _fill(self, s, slot):
        e = self.ev[slot]
        if e is not None:
            e.synchronize()
        cids = self.cids_for(s); b = self.np[slot]
        for i in np.argsort(cids, kind="stable"):
            n = os.preadv(self.fd, [memoryview(b[i])], int(cids[i]) * self.row)
            assert n == self.row, (s, i, n)
        return slot

    def _submit(self, s):
        if s < self.total and s not in self.fut:
            self.fut[s] = self.ex.submit(self._fill, s, s % self.ring)

    def start(self, s0):
        for s in range(s0, s0 + self.ring):
            self._submit(s)

    def get(self, s):
        t = time.time(); slot = self.fut.pop(s).result(); self.wait_s += time.time() - t; self.n += 1
        g = self.buf[slot].to(self.dev, non_blocking=True)
        ev = torch.cuda.Event(); ev.record(); self.ev[slot] = ev
        self._submit(s + self.ring)
        return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["v2ft", "pix", "p50"], required=True)
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--palette", type=Path, required=True)
    ap.add_argument("--clip", type=Path); ap.add_argument("--init", type=Path); ap.add_argument("--v2-clip-table", type=Path)
    ap.add_argument("--images", type=float, help="total training images (schedule length)")
    ap.add_argument("--epochs", type=float, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--layers", type=int, default=12); ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--caption-dropout", type=float, default=0.1)
    ap.add_argument("--p-subj", type=float, default=0.175); ap.add_argument("--p-syn", type=float, default=0.10)
    ap.add_argument("--p-typo", type=float, default=0.0, help="per-sample prob of using the typo caption copy (needs data_typo)")
    ap.add_argument("--flip-p", type=float, default=0.5); ap.add_argument("--alpha-weight", type=float, default=0.25)
    ap.add_argument("--eval-every-images", type=float, default=1e6)
    ap.add_argument("--save-every-images", type=float, default=1e6, help="last.pt (full) + img_XXXM.pt (model only)")
    ap.add_argument("--save-at-images", default="", help="comma list: permanent FULL checkpoints full_XXXM.pt (with optimizer)")
    ap.add_argument("--eval-at-start", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke: stop after N steps (schedule unchanged)")
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--sample-per", type=int, default=8)
    ap.add_argument("--no-samples", action="store_true")
    ap.add_argument("--eval-limit", type=int, default=0, help="smoke: use only the first N rows of each eval set")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda")
    ap.add_argument("--compile", action="store_true", help="torch.compile the training forward+loss (Linux/triton)")
    ap.add_argument("--compile-mode", default="default", help="torch.compile mode (default | max-autotune-no-cudagraphs)")
    ap.add_argument("--nan-check-every", type=int, default=1, help="check loss finiteness every N steps (1 = every step, as on Windows)")
    # s7
    ap.add_argument("--text-len", type=int, default=26, help="CLIP positions: 26 = old (24 ids), 18 = s7 (16 ids)")
    ap.add_argument("--cond", action="store_true", help="palette (3) + style (1) conditioning slots")
    ap.add_argument("--p-pal-top1", type=float, default=0.25); ap.add_argument("--p-pal-top3", type=float, default=0.25)
    ap.add_argument("--p-style", type=float, default=0.4)
    ap.add_argument("--balance-alpha", type=float, default=0.0); ap.add_argument("--balance-cap", type=float, default=6.0)
    ap.add_argument("--schedule", choices=["cosine", "wsd", "decay"], default="cosine")
    ap.add_argument("--ce-chunks", type=int, default=0, help="0 = old raster-logits loss; 1 = patch-order loss; N>1 = chunked")
    ap.add_argument("--clip-cache", type=Path, help="folder from build_clip_cache_s7.py (stream features instead of online CLIP)")
    ap.add_argument("--prefetch-ring", type=int, default=12); ap.add_argument("--prefetch-workers", type=int, default=6)
    ap.add_argument("--branch-from", type=Path, help="start from a FULL checkpoint (model+optimizer), e.g. a decay branch")
    ap.add_argument("--bench-static", action="store_true", help="speed check: reuse the first batch every step (no data path)")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out = a.out; out.mkdir(parents=True, exist_ok=True); (out / "samples").mkdir(exist_ok=True)
    if (out / "train_summary.json").exists():
        print("already finished", flush=True); return
    log = open(out / "progress.log", "a", encoding="utf-8"); met = open(out / "metrics.jsonl", "a", encoding="utf-8")

    def emit(s):
        line = f"[{now()}] {s}"; print(line, flush=True); log.write(line + "\n"); log.flush()

    def record(d):
        met.write(json.dumps(d) + "\n"); met.flush()

    s7 = a.text_len != 26 or a.cond or a.ce_chunks or a.clip_cache
    assert not s7 or a.arch == "p50", "s7 options are for --arch p50"
    t0 = time.time()
    tr = np.load(a.data / "v3_train.npz"); ev = np.load(a.data / "v3_eval.npz")
    img_np = np.load(a.cache / "train_img.npy", mmap_mode="r")
    N = int(img_np.shape[0]) if not a.limit_rows else min(a.limit_rows, int(img_np.shape[0]))
    sl = slice(0, N)
    I = as_i16(img_np[sl]).to(dev)
    SID = torch.from_numpy(tr["subj_id"][sl].astype(np.int64)).to(dev); NF = torch.from_numpy(tr["noflip"][sl]).to(dev)
    if a.p_typo > 0:
        assert "text_typo" in tr.files, "--p-typo needs a data folder with text_typo/has_typo (v3/data_typo)"
    tab = D.build_caption_table(tr, a.text_len)
    CT = torch.from_numpy(tab["table"].view(np.int16)).to(dev)
    RM = torch.from_numpy(tab["row_main"][sl].astype(np.int64)).to(dev)
    RS = torch.from_numpy(tab["row_syn"][sl].astype(np.int64)).to(dev)
    RT = torch.from_numpy(tab["row_typo"][sl].astype(np.int64)).to(dev)
    SJ = torch.from_numpy(tab["subj"].astype(np.int64)).to(dev); BLANK = tab["blank"]
    cap_info = {"captions": tab["n"], "sha256": tab["sha256"][:16], "fit_ids": a.text_len - 2, "fit": tab["fit"]}
    assert int(I.min()) >= 0 and int(I.max()) < 4097
    palette = np.load(a.palette); prompts = json.loads((a.data / "prompts_v3.json").read_text())

    def evtext(k):
        return u16(as_i16(D.trim(ev[k], a.text_len)).to(dev))
    E = {"probe": (evtext("probe_text"), torch.from_numpy(np.load(a.cache / "probe_img.npy").astype(np.int64)).to(dev),
                   torch.from_numpy(ev["probe_cbucket"].astype(np.int64)).to(dev)),
         "iid": (evtext("iid_text"), torch.from_numpy(np.load(a.cache / "iid_img.npy").astype(np.int64)).to(dev),
                 torch.from_numpy(ev["iid_cbucket"].astype(np.int64)).to(dev))}
    seen = torch.from_numpy(ev["iid_seen_by_v2"]).to(dev)
    E["iidrestored"] = tuple(x[~seen] for x in E["iid"])  # iid rows of restored subjects: unseen by v1/v2 too
    cond_info = {}
    if a.cond:
        TOP3, STY = D.cond_stats(I, palette)
        st = STY.cpu().numpy(); t3 = TOP3.cpu().numpy()
        cond_info = {"fill_small_med_full": np.bincount(st[:, 0], minlength=4)[1:].tolist(),
                     "edge_dark_soft": np.bincount(st[:, 1], minlength=3)[1:].tolist(),
                     "colours_low_mid_high": np.bincount(st[:, 2], minlength=4)[1:].tolist(),
                     "rows_lt3_colours": int((t3 == D.NO_PALETTE).any(1).sum())}
        it3, ist = D.cond_stats(E["iid"][1], palette)   # extra eval: iid with its own top-3 palette + style tags
        E["iidcond"] = (*E["iid"], torch.cat([it3, ist], 1))
    if a.eval_limit:
        E = {k: tuple(x[:a.eval_limit] for x in v) for k, v in E.items()}
    W = None; bal_info = {}
    if a.balance_alpha > 0:
        w, n_s = D.balanced_weights(tr["subj_id"][sl], a.balance_alpha, a.balance_cap)
        W = torch.from_numpy(w).float().to(dev)
        bal_info = {"alpha": a.balance_alpha, "cap": a.balance_cap, "mult_min": round(float(w.min()), 3),
                    "mult_max": round(float(w.max()), 3), "n_s_min": int(n_s[n_s > 0].min()), "n_s_max": int(n_s.max())}
    torch.manual_seed(a.seed)
    net = Net(a, dev); model = net.model
    base_images = 0
    total_images = a.images if a.images else a.epochs * N
    a.total_steps = int(math.ceil(total_images / a.batch))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, fused=dev.type == "cuda")
    nparam = sum(p.numel() for p in params)
    B = a.batch; nb = int(round(a.caption_dropout * B)); nt = B - nb; spe = N // B
    save_at = sorted(float(x) for x in a.save_at_images.split(",") if x.strip())

    step = 0; train_secs = 0.0; next_eval = a.eval_every_images; next_save = a.save_every_images
    best = None
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False); opt.load_state_dict(ck["optimizer"])
        step, train_secs = ck["step"], ck["train_secs"]
        next_eval, next_save, best = ck["next_eval"], ck["next_save"], ck.get("best")
        base_images = ck.get("base_images", 0)
        torch.set_rng_state(ck["rng"]["torch"])
        if dev.type == "cuda" and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state(ck["rng"]["cuda"])
        emit(f"resumed from last.pt at step {step}")
    elif a.branch_from:
        ck = torch.load(a.branch_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True); opt.load_state_dict(ck["optimizer"])
        base_images = ck["images"] + ck.get("base_images", 0)
        next_eval = a.eval_every_images; next_save = a.save_every_images
        net.init_info = {"init": str(a.branch_from), "init_step": ck.get("step"), "init_images": base_images}
        emit(f"branch from {a.branch_from} (images {base_images})")
    emit(json.dumps({"event": "start", "arch": a.arch, "rows": N, "params": nparam, "config": net.cfg.to_dict(),
                     "total_steps": a.total_steps, "total_images": total_images, "load_s": round(time.time() - t0, 1),
                     "captions": cap_info, "cond": cond_info, "balance": bal_info, "blank_per_batch": nb,
                     **net.init_info, "args": {k: str(v) for k, v in vars(a).items()}}))
    record({"event": "start", "arch": a.arch, "rows": N, "params": nparam, "total_steps": a.total_steps,
            "total_images": total_images, "time": now(), "base_images": base_images, **net.init_info})

    def imgs(s):
        return base_images + s * B

    def lr_at(s):
        if a.schedule == "decay":
            return a.lr + (a.lr_min - a.lr) * min(1.0, s / max(1, a.total_steps))
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        if a.schedule == "wsd":
            return a.lr
        prog = min(1.0, (s - a.warmup) / max(1, a.total_steps - a.warmup))
        return a.lr_min + (a.lr - a.lr_min) * 0.5 * (1 + math.cos(math.pi * prog))

    def save(path, full=True):
        sd = {k: v for k, v in model.state_dict().items() if not k.startswith("clip_table")}
        obj = {"model": sd, "step": step, "train_secs": train_secs, "next_eval": next_eval, "next_save": next_save,
               "best": best, "images": step * B, "base_images": base_images, "arch": a.arch, "config": net.cfg.to_dict(),
               "args": vars(a), "epoch": step // spe, "cursor": (step % spe) * B,
               "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state() if dev.type == "cuda" else None}}
        if full:
            obj["optimizer"] = opt.state_dict()
        tmp = Path(str(path) + ".tmp"); torch.save(obj, tmp); os.replace(tmp, path)

    def run_eval(tag):
        nonlocal best
        r = evaluate(net, E)
        dev_nll = r["iid_nll"]
        best = dev_nll if best is None or dev_nll < best else best
        row = {"event": "eval", "step": step, "images": imgs(step), "epoch": round(imgs(step) / N, 3),
               "dev_nll": dev_nll, "best_dev_nll": best, "tag": tag, "text_len": a.text_len, **r, "time": now()}
        record(row); emit("eval " + json.dumps(row))
        nonlocal_last[0] = row
        return row

    # ---------------------------------------------------------------- epoch plans (all RNG on the GPU)
    plans = {}

    def make_plan(ep):
        g = torch.Generator(device=dev); g.manual_seed(a.seed * 1000 + ep)
        Mx = spe * B
        idx = torch.multinomial(W, Mx, replacement=True, generator=g) if W is not None else \
            torch.randperm(N, device=dev, generator=g)[:Mx]
        r = torch.rand(6, Mx, device=dev, generator=g)
        cid = RM[idx]
        rs = RS[idx]; cid = torch.where((r[0] < a.p_syn) & (rs >= 0), rs, cid)
        if a.p_typo > 0:
            rt = RT[idx]; cid = torch.where((r[1] < a.p_typo) & (rt >= 0), rt, cid)
        cid = torch.where(r[2] < a.p_subj, SJ[SID[idx]], cid)
        cid = torch.where(torch.arange(Mx, device=dev) % B >= nt, torch.full_like(cid, BLANK), cid)
        p = {"idx": idx, "cid": cid.int(), "flip": (r[3] < a.flip_p) & ~NF[idx]}
        if a.cond:
            k = torch.where(r[4] < 1 - a.p_pal_top1 - a.p_pal_top3, 0, torch.where(r[4] < 1 - a.p_pal_top3, 1, 3))
            slots = torch.where(torch.arange(3, device=dev)[None] < k[:, None], TOP3[idx], torch.full_like(TOP3[idx], D.NO_PALETTE))
            slots = slots.gather(1, torch.rand(Mx, 3, device=dev, generator=g).argsort(1))
            sty = torch.where((r[5] < a.p_style)[:, None], STY[idx], torch.zeros_like(STY[idx]))
            p["cond"] = torch.cat([slots, sty], 1).short()
        if a.clip_cache:
            p["cid_cpu"] = p["cid"].view(-1, B)[:, :nt].cpu().numpy().astype(np.int64)
        return p

    def plan(ep):
        if ep not in plans:
            plans[ep] = make_plan(ep)
            for k in [k for k in plans if k < ep - 1]:
                del plans[k]
        return plans[ep]

    # ---------------------------------------------------------------- text features
    blank_ids = torch.from_numpy(D.blank_caption(a.text_len).astype(np.int64)).to(dev)
    pf = None
    if a.arch == "p50":
        blank_feat = net.text_in(blank_ids).float()                       # [1,TL,768], CLIP once
    if a.clip_cache:
        meta = json.loads((a.clip_cache / "meta.json").read_text())
        assert meta["sha256_table"] == tab["sha256"] and meta["captions"] == tab["n"] and meta["text_len"] == a.text_len, \
            "clip cache does not match this caption table"
        bos_feat = torch.from_numpy(np.load(a.clip_cache / "bos_feat.npy")).float().to(dev)
        L = meta["stored_positions"]
        mm = np.memmap(a.clip_cache / meta["file"], dtype=np.float16, mode="r", shape=(meta["captions"], L, 768))
        chk = np.random.default_rng(0).choice(tab["n"], 256, replace=False)
        on = net.text_in(torch.from_numpy(tab["table"][chk].astype(np.int64)).to(dev)).float()
        cf = torch.from_numpy(np.ascontiguousarray(mm[chk])).float().to(dev)
        diff = float((on[:, 1:] - cf).abs().max()); bdiff = float((on[:, 0] - bos_feat).abs().max())
        emit(f"clip cache {a.clip_cache} rows {meta['captions']}: max |online - cached| {diff:.3g} (bos {bdiff:.3g})")
        del mm

        def cids_for(s):
            ep, k = divmod(s, spe)
            return plans[ep]["cid_cpu"][k]
        pf = FeatPrefetch(a.clip_cache / meta["file"], L, nt, cids_for, a.total_steps, dev,
                          ring=a.prefetch_ring, workers=a.prefetch_workers)

    def fwd_loss_s7(h_nt, img, cond):
        """h_nt: online -> CLIP ids [nt,TL]; cached -> features [nt,TL-1,768]. Blank rows are always the last nb."""
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            if a.clip_cache:
                x = torch.cat([bos_feat.expand(h_nt.shape[0], 1, 768), h_nt.float()], 1)
            else:
                x = net.clip(h_nt).float()
            x = torch.cat([x, blank_feat.expand(nb, -1, -1)], 0)
            return model.loss_fwd(x, img, a.alpha_weight, cond, chunks=a.ce_chunks)

    def fwd_loss_old(t, img):
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lg = net.logits(t, img)
        loss, nll, _, _ = net.losses(lg, img, a.alpha_weight)
        return loss, nll.mean()
    use_s7 = a.arch == "p50" and (a.ce_chunks or a.cond or a.clip_cache or a.text_len != 26)
    fwd = fwd_loss_s7 if use_s7 else fwd_loss_old
    if a.compile:
        fwd = torch.compile(fwd, mode=a.compile_mode, dynamic=False)
        emit(f"torch.compile enabled (mode={a.compile_mode})")
    emit(f"path: {'s7' if use_s7 else 'old-loss'}, text {'cached ' + str(a.clip_cache) if a.clip_cache else 'online CLIP'}, "
         f"blank rows/batch {nb}, ce_chunks {a.ce_chunks}, schedule {a.schedule}, steps/epoch {spe}")

    nonlocal_last = [None]
    if a.eval_at_start and step == 0:
        run_eval("start")
    plan(step // spe); plan(step // spe + 1)
    if pf:
        pf.start(step)
    model.train()
    win_t = time.time(); win_steps = 0; acc = torch.zeros((), device=dev)
    stop_reason = "done"; static = None
    while step < a.total_steps:
        ep, k = divmod(step, spe)
        if k == 0:
            plan(ep + 1)                     # next epoch ready before the prefetcher needs it
        if static is None or not a.bench_static:
            P = plan(ep); bs_ = slice(k * B, (k + 1) * B)
            idx = P["idx"][bs_]; img = I[idx].long()
            if a.flip_p > 0:
                img = torch.where(P["flip"][bs_][:, None], img.view(B, 16, 16).flip(-1).reshape(B, 256), img)
            cond = P["cond"][bs_].long() if a.cond else None
            if use_s7:
                txt = pf.get(step) if pf else u16(CT[P["cid"][bs_][:nt].long()])
                args_ = (txt, img, cond)
            else:
                args_ = (u16(CT[P["cid"][bs_].long()]), img)
            if a.bench_static:
                static = args_
        else:
            args_ = static
            if pf:
                pf.get(step)                 # keep the prefetcher moving
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        loss, nll = fwd(*args_)
        if (step % a.nan_check_every == 0) and not torch.isfinite(loss):
            record({"event": "nan_stop", "step": step}); emit(f"non-finite loss at step {step}, stopping")
            stop_reason = "nan"; break
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        step += 1; win_steps += 1
        acc += nll.detach()
        if step % a.log_every == 0:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - win_t; train_secs += dt; sps = win_steps / max(dt, 1e-9)
            eta = (a.total_steps - step) / max(sps, 1e-9)
            peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else 0
            row = {"step": step, "images": imgs(step), "epoch": round(imgs(step) / N, 3),
                   "train_nll": float(acc) / max(win_steps, 1), "lr": lr_at(step), "steps_per_sec": round(sps, 3),
                   "images_per_sec": round(sps * B, 1), "eta_sec": round(eta), "peak_allocated_mib": round(peak)}
            if pf:
                row["prefetch_wait_ms"] = round(1000 * pf.wait_s / max(pf.n, 1), 2); pf.wait_s = 0.0; pf.n = 0
            record(row)
            emit(f"step {step}/{a.total_steps} img {imgs(step) / 1e6:.2f}M nll {row['train_nll']:.4f} lr {row['lr']:.3e} "
                 f"{sps:.2f} step/s eta {eta / 3600:.2f}h peak {peak:.0f}MiB" + (f" pfwait {row['prefetch_wait_ms']}ms" if pf else ""))
            win_t = time.time(); win_steps = 0; acc.zero_()
        if step * B >= next_eval:
            run_eval("periodic"); next_eval += a.eval_every_images; win_t = time.time()
        if step * B >= next_save:
            next_save += a.save_every_images
            save(out / "last.pt"); save(out / f"img_{imgs(step) // 1000000:03d}M.pt", full=False); win_t = time.time()
        for x in save_at:
            if (step - 1) * B < x - base_images <= step * B:
                nm = f"full_{int(x // 1e6):03d}M.pt" if x % 1e6 == 0 else f"full_{x / 1e6:07.3f}M.pt"
                save(out / nm); emit(f"permanent checkpoint {nm}"); win_t = time.time()
        if a.max_steps and step >= a.max_steps:
            stop_reason = "max_steps"; break
        if a.max_hours and (time.time() - t0) / 3600 > a.max_hours:
            stop_reason = "max_hours"; emit("max-hours reached, finishing early"); break
    last_eval = nonlocal_last[0]
    final = last_eval if last_eval is not None and last_eval["step"] == step else run_eval("final")
    save(out / "last.pt")
    samp = {}
    if not a.no_samples:
        sd = out / "samples" / f"img_{imgs(step) / 1e6:05.2f}M"
        samp = do_samples(net, prompts, palette, sd, E["iid"][1], dev, per=a.sample_per)
        record({"event": "samples", "step": step, **samp}); emit("samples " + json.dumps(samp))
    summ = {"arch": a.arch, "stop_reason": stop_reason, "steps": step, "images": imgs(step), "params": nparam,
            "train_hours": round(train_secs / 3600, 3), "wall_hours": round((time.time() - t0) / 3600, 3),
            "final_eval": final, "samples": samp, **net.init_info,
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20) if dev.type == "cuda" else 0}
    record({"event": "done", "step": step, "time": now()})
    if stop_reason in ("done", "max_hours"):
        (out / "train_summary.json").write_text(json.dumps(summ, indent=1))
    else:
        (out / f"summary_{stop_reason}.json").write_text(json.dumps(summ, indent=1))
    emit("summary " + json.dumps(summ))
    if pf:
        pf.ex.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
