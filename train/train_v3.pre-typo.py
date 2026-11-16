"""Pixie v3 trainer: arch v2ft (fine-tune the v2 SpriteV2), pix (pilot a), p50 (pilot b).

GPU-resident data (images int16, CLIP ids as int16 bit patterns), caption variants picked
on the GPU per batch (10% blank -> CFG, 17.5% subject-as-caption, 10% synonym where one
exists), h-flip except directional captions, fused AdamW, bf16 autocast, cosine LR sized
by --images. Eval in fp32 on the v3 probe (100 unseen subjects) and iid split, by bucket,
with per-slot NLL (2x2 position) at fixed image counts so runs compare at equal images seen.
metrics.jsonl keys follow the v2 trainer so the training pusher can read it.
"""
import argparse, glob, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3_model as M  # noqa: E402

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


def empty_text(b, device):
    t = torch.full((b, M.TEXT_LEN), M.CLIP_EOS, dtype=torch.long, device=device)
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
            cfg = M.V3Config(arch=a.arch, d_model=a.width, n_layer=a.layers, n_head=a.heads, dropout=a.dropout)
            self.model = M.build(cfg).to(dev)
            self.clip = M.ClipTextPenultimate().to(dev)
            self.clip.load_hf(torch.load(a.clip, map_location="cpu", weights_only=False))
            self.clip.to(torch.bfloat16 if dev.type == "cuda" else torch.float32).eval()
            self.init_info = {}
        self.cfg = self.model.config

    def text_in(self, t26):
        if self.arch == "v2ft":
            return to_v2_text(t26)
        with torch.no_grad(), torch.autocast(device_type=self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            return self.clip(t26)

    def logits(self, t26, img):
        x = self.text_in(t26)
        return self.model(x, img)

    def losses(self, lg, img, aw):
        return M.losses(lg, img, aw) if self.arch != "v2ft" else self.V2.losses(lg, img, aw)

    def sample(self, t26, cfg, generator):
        if self.arch == "v2ft":
            return self.V2.sample(self.model, to_v2_text(t26), temperature=0.8, top_k=32, cfg=cfg, generator=generator)
        self.model.eval()
        h = self.text_in(t26); hu = self.text_in(empty_text(t26.shape[0], t26.device)) if cfg != 1.0 else None
        out = self.model.sample(h, hu, cfg=cfg, temperature=0.8, top_k=32, generator=generator)
        self.model.train()
        return out


@torch.no_grad()
def evaluate(net, E, bs=64, uncond=True):
    """fp32 eval. E: dict name -> (text26 long, img long, bucket int or None)."""
    m = net.model; m.eval()
    slot = M.slot_of_pixel(next(m.parameters()).device)
    res = {}
    for name, (T, I, B) in E.items():
        tot = torch.zeros((), dtype=torch.float64, device=I.device); per_row = []
        slots = torch.zeros(4, dtype=torch.float64, device=I.device); tot_u = 0.0
        for s in range(0, I.shape[0], bs):
            t = T[s:s + bs]; im = I[s:s + bs]
            lg = net.logits(t, im).float()
            _, nll, _, _ = net.losses(lg, im, 0.0)
            per_row.append(nll.mean(1).double())
            for k in range(4):
                slots[k] += nll[:, slot == k].double().sum()
            if uncond and name == "probe":
                _, nu, _, _ = net.losses(net.logits(empty_text(t.shape[0], t.device), im).float(), im, 0.0)
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
        names = list(prompts[group]); t = torch.tensor([prompts[group][p] for p in names for _ in range(per)], device=dev)
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
    ap.add_argument("--flip-p", type=float, default=0.5); ap.add_argument("--alpha-weight", type=float, default=0.25)
    ap.add_argument("--eval-every-images", type=float, default=1e6)
    ap.add_argument("--save-every-images", type=float, default=1e6)
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

    t0 = time.time()
    tr = np.load(a.data / "v3_train.npz"); ev = np.load(a.data / "v3_eval.npz")
    img_np = np.load(a.cache / "train_img.npy", mmap_mode="r")
    N = int(img_np.shape[0]) if not a.limit_rows else min(a.limit_rows, int(img_np.shape[0]))
    sl = slice(0, N)
    I = as_i16(img_np[sl]).to(dev)
    TM = as_i16(tr["text_main"][sl]).to(dev); TS = as_i16(tr["text_syn"][sl]).to(dev)
    HS = torch.from_numpy(tr["has_syn"][sl]).to(dev); SID = torch.from_numpy(tr["subj_id"][sl].astype(np.int64)).to(dev)
    ST = as_i16(tr["subj_text"]).to(dev); NF = torch.from_numpy(tr["noflip"][sl]).to(dev)
    assert int(I.min()) >= 0 and int(I.max()) < 4097
    E = {"probe": (u16(as_i16(ev["probe_text"]).to(dev)), torch.from_numpy(np.load(a.cache / "probe_img.npy").astype(np.int64)).to(dev),
                   torch.from_numpy(ev["probe_cbucket"].astype(np.int64)).to(dev)),
         "iid": (u16(as_i16(ev["iid_text"]).to(dev)), torch.from_numpy(np.load(a.cache / "iid_img.npy").astype(np.int64)).to(dev),
                 torch.from_numpy(ev["iid_cbucket"].astype(np.int64)).to(dev))}
    seen = torch.from_numpy(ev["iid_seen_by_v2"]).to(dev)
    E["iidrestored"] = tuple(x[~seen] for x in E["iid"])  # iid rows of restored subjects: unseen by v1/v2 too
    if a.eval_limit:
        E = {k: tuple(x[:a.eval_limit] for x in v) for k, v in E.items()}
    palette = np.load(a.palette); prompts = json.loads((a.data / "prompts_v3.json").read_text())
    torch.manual_seed(a.seed)
    net = Net(a, dev); model = net.model
    total_images = a.images if a.images else a.epochs * N
    a.total_steps = int(math.ceil(total_images / a.batch))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, fused=dev.type == "cuda")
    nparam = sum(p.numel() for p in params)
    emit(json.dumps({"event": "start", "arch": a.arch, "rows": N, "params": nparam, "config": net.cfg.to_dict(),
                     "total_steps": a.total_steps, "total_images": total_images, "load_s": round(time.time() - t0, 1),
                     **net.init_info, "args": {k: str(v) for k, v in vars(a).items()}}))
    record({"event": "start", "arch": a.arch, "rows": N, "params": nparam, "total_steps": a.total_steps,
            "total_images": total_images, "time": now(), **net.init_info})

    step = 0; epoch = 0; cursor = 0; train_secs = 0.0; next_eval = a.eval_every_images; next_save = a.save_every_images
    best = None
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False); opt.load_state_dict(ck["optimizer"])
        step, epoch, cursor, train_secs = ck["step"], ck["epoch"], ck["cursor"], ck["train_secs"]
        next_eval, next_save, best = ck["next_eval"], ck["next_save"], ck.get("best")
        torch.set_rng_state(ck["rng"]["torch"])
        if dev.type == "cuda" and ck["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state(ck["rng"]["cuda"])
        emit(f"resumed from last.pt at step {step}")

    def perm_for(ep):
        g = torch.Generator(device=dev); g.manual_seed(a.seed * 1000 + ep)
        return torch.randperm(N, device=dev, generator=g)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        prog = min(1.0, (s - a.warmup) / max(1, a.total_steps - a.warmup))
        return a.lr_min + (a.lr - a.lr_min) * 0.5 * (1 + math.cos(math.pi * prog))

    def save(path):
        sd = {k: v for k, v in model.state_dict().items() if not k.startswith("clip_table")}
        obj = {"model": sd, "optimizer": opt.state_dict(), "step": step, "epoch": epoch, "cursor": cursor,
               "train_secs": train_secs, "next_eval": next_eval, "next_save": next_save, "best": best,
               "images": step * a.batch, "arch": a.arch, "config": net.cfg.to_dict(), "args": vars(a),
               "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state() if dev.type == "cuda" else None}}
        tmp = Path(str(path) + ".tmp"); torch.save(obj, tmp); os.replace(tmp, path)

    def run_eval(tag):
        nonlocal best
        r = evaluate(net, E)
        dev_nll = r["iid_nll"]
        best = dev_nll if best is None or dev_nll < best else best
        row = {"event": "eval", "step": step, "images": step * a.batch, "epoch": round(step * a.batch / N, 3),
               "dev_nll": dev_nll, "best_dev_nll": best, "tag": tag, **r, "time": now()}
        record(row); emit("eval " + json.dumps(row))
        nonlocal_last[0] = row
        return row

    nonlocal_last = [None]
    if a.eval_at_start and step == 0:
        run_eval("start")
    perm = perm_for(epoch)
    model.train()
    win_t = time.time(); win_steps = 0; acc = {"nll": 0.0, "alpha": 0.0, "n": 0}
    stop_reason = "done"
    empty = empty_text(a.batch, dev)

    def fwd_loss(t, img):
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lg = net.logits(t, img)
        return net.losses(lg, img, a.alpha_weight)
    if a.compile:
        fwd_loss = torch.compile(fwd_loss, mode=a.compile_mode, dynamic=False)
        emit(f"torch.compile enabled (mode={a.compile_mode})")
    while step < a.total_steps:
        if cursor + a.batch > N:
            epoch += 1; cursor = 0; perm = perm_for(epoch)
        idx = perm[cursor:cursor + a.batch]; cursor += a.batch
        b = idx.shape[0]
        img = I[idx].long()
        u = torch.rand(4, b, device=dev)
        t = u16(TM[idx])
        t = torch.where(((u[1] < a.p_syn) & HS[idx])[:, None], u16(TS[idx]), t)
        t = torch.where((u[2] < a.p_subj)[:, None], u16(ST[SID[idx]]), t)
        t = torch.where((u[0] < a.caption_dropout)[:, None], empty[:b], t)
        flip = (u[3] < a.flip_p) & ~NF[idx]
        if a.flip_p > 0:
            img = torch.where(flip[:, None], img.view(b, 16, 16).flip(-1).reshape(b, 256), img)
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        loss, nll, an, _ = fwd_loss(t, img)
        if (step % a.nan_check_every == 0) and not torch.isfinite(loss):
            record({"event": "nan_stop", "step": step}); emit(f"non-finite loss at step {step}, stopping")
            stop_reason = "nan"; break
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        step += 1; win_steps += 1
        acc["nll"] += float(nll.mean().detach()) if step % 10 == 0 else 0.0
        acc["n"] += 1 if step % 10 == 0 else 0
        if step % a.log_every == 0:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - win_t; train_secs += dt; sps = win_steps / max(dt, 1e-9)
            eta = (a.total_steps - step) / max(sps, 1e-9)
            peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else 0
            row = {"step": step, "images": step * a.batch, "epoch": round(step * a.batch / N, 3),
                   "train_nll": acc["nll"] / max(acc["n"], 1), "lr": lr_at(step), "steps_per_sec": round(sps, 3),
                   "images_per_sec": round(sps * a.batch, 1), "eta_sec": round(eta), "peak_allocated_mib": round(peak)}
            record(row)
            emit(f"step {step}/{a.total_steps} img {step * a.batch / 1e6:.2f}M nll {row['train_nll']:.4f} lr {row['lr']:.3e} "
                 f"{sps:.2f} step/s eta {eta / 3600:.2f}h peak {peak:.0f}MiB")
            win_t = time.time(); win_steps = 0; acc = {"nll": 0.0, "alpha": 0.0, "n": 0}
        if step * a.batch >= next_eval:
            run_eval("periodic"); next_eval += a.eval_every_images; win_t = time.time()
        if step * a.batch >= next_save:
            next_save += a.save_every_images
            save(out / "last.pt"); save(out / f"img_{step * a.batch // 1000000:03d}M.pt"); win_t = time.time()
        if a.max_steps and step >= a.max_steps:
            stop_reason = "max_steps"; break
        if a.max_hours and (time.time() - t0) / 3600 > a.max_hours:
            stop_reason = "max_hours"; emit("max-hours reached, finishing early"); break
    last_eval = nonlocal_last[0]
    final = last_eval if last_eval is not None and last_eval["step"] == step else run_eval("final")
    save(out / "last.pt")
    samp = {}
    if not a.no_samples:
        sd = out / "samples" / f"img_{step * a.batch / 1e6:05.2f}M"
        samp = do_samples(net, prompts, palette, sd, E["iid"][1], dev, per=a.sample_per)
        record({"event": "samples", "step": step, **samp}); emit("samples " + json.dumps(samp))
    summ = {"arch": a.arch, "stop_reason": stop_reason, "steps": step, "images": step * a.batch, "params": nparam,
            "train_hours": round(train_secs / 3600, 3), "wall_hours": round((time.time() - t0) / 3600, 3),
            "final_eval": final, "samples": samp, **net.init_info,
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20) if dev.type == "cuda" else 0}
    record({"event": "done", "step": step, "time": now()})
    if stop_reason in ("done", "max_hours"):
        (out / "train_summary.json").write_text(json.dumps(summ, indent=1))
    else:
        (out / f"summary_{stop_reason}.json").write_text(json.dumps(summ, indent=1))
    emit("summary " + json.dumps(summ))


if __name__ == "__main__":
    main()
