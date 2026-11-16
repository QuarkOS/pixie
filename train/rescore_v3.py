"""Re-score existing checkpoints on the v3 probe + iid splits (fp32, same metrics as train_v3.evaluate).

--v1 NAME=PATH (pixelgen SpriteTransformer; uses the raw v1 token rows, so v1 sees exactly its own caption
encoding), --v2 NAME=PATH (SpriteV2; CLIP-id+1 text). Optional --samples DIR writes rare/panel grids for v2
checkpoints. pixelgen is imported from a COPY (--pixelgen-root), never modified."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3_model as M  # noqa: E402
from train_v3 import CB_NAMES, to_v2_text, write_grid  # noqa: E402


def metrics(nll, B, name):
    slot = M.slot_of_pixel(nll.device)
    pr = nll.double().mean(1)
    r = {f"{name}_nll": float(pr.mean()),
         f"{name}_slot_nll": [float(nll[:, slot == k].double().mean()) for k in range(4)]}
    for k, cb in enumerate(CB_NAMES):
        if bool((B == k).any()):
            r[f"{name}_{cb}"] = float(pr[B == k].mean())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--pixelgen-root", type=Path, required=True)
    ap.add_argument("--v2-clip-table", type=Path); ap.add_argument("--palette", type=Path)
    ap.add_argument("--v1", action="append", default=[]); ap.add_argument("--v2", action="append", default=[])
    ap.add_argument("--samples", type=Path); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(a.device)
    ev = np.load(a.data / "v3_eval.npz")
    sets = {}
    for name in ("probe", "iid"):
        tok = np.load(a.cache / f"{name}_tok.npy").astype(np.int64)
        img = np.load(a.cache / f"{name}_img.npy").astype(np.int64)
        txt = ev[f"{name}_text"].astype(np.int64)
        B = ev[f"{name}_cbucket"].astype(np.int64)
        if a.limit:
            tok, img, txt, B = tok[:a.limit], img[:a.limit], txt[:a.limit], B[:a.limit]
        assert (tok[:, 98:] - 259 == img).all()
        sets[name] = tuple(torch.from_numpy(x).to(dev) for x in (tok, img, txt, B))
    seen = torch.from_numpy(ev["iid_seen_by_v2"][:a.limit] if a.limit else ev["iid_seen_by_v2"]).to(dev)
    sets["iidrestored"] = tuple(x[~seen] for x in sets["iid"])  # restored subjects: never seen by v1/v2
    res = json.loads(a.out.read_text()) if a.out.exists() else {}
    sys.path.insert(0, str(a.pixelgen_root))
    for spec in a.v1:
        nm, path = spec.split("=", 1)
        from pixelgen.checkpoint import build_from_checkpoint
        from pixelgen.losses import image_token_logprobs
        model, payload = build_from_checkpoint(Path(path)); model.to(dev).eval()
        r = {"kind": "v1", "path": path, "step": payload.get("step")}
        with torch.no_grad():
            for name, (tok, img, txt, B) in sets.items():
                out = []
                for s in range(0, tok.shape[0], 64):
                    lg, _ = model(tok[s:s + 64])
                    out.append(-image_token_logprobs(lg.float(), tok[s:s + 64], model.config.palette_size))
                r.update(metrics(torch.cat(out), B, name))
        res[nm] = r; print(nm, json.dumps(r), flush=True)
        del model; torch.cuda.empty_cache() if dev.type == "cuda" else None
    for spec in a.v2:
        nm, path = spec.split("=", 1)
        import v2_model as V2
        ck = torch.load(path, map_location="cpu", weights_only=False); sd = ck["model"]
        cfgd = ck.get("config") or {}
        cfg = V2.V2Config(**cfgd) if cfgd else V2.V2Config(resid_vocab=int(sd["text_resid.weight"].shape[0]))
        cfg.dropout = 0.0
        model = V2.SpriteV2(cfg, np.load(a.v2_clip_table), sd["clip2v2"].numpy()).to(dev)
        model.load_state_dict(sd, strict=False); model.eval()
        r = {"kind": "v2", "path": path, "step": ck.get("step"), "best_dev_old_split": ck.get("best_dev")}
        with torch.no_grad():
            for name, (tok, img, txt, B) in sets.items():
                out = []
                for s in range(0, img.shape[0], 64):
                    lg = model(to_v2_text(txt[s:s + 64]), img[s:s + 64])
                    out.append(V2.losses(lg.float(), img[s:s + 64], 0.0)[1])
                r.update(metrics(torch.cat(out), B, name))
            if a.samples:
                prompts = json.loads((a.data / "prompts_v3.json").read_text()); pal = np.load(a.palette)
                g = torch.Generator(device=dev); g.manual_seed(1234)
                allg = []
                for group in ("rare", "panel10"):
                    names = list(prompts[group])
                    t = torch.tensor([prompts[group][p] for p in names for _ in range(8)], device=dev)
                    for cfgv in (1.0, 2.0):
                        ids = V2.sample(model, to_v2_text(t), temperature=0.8, top_k=32, cfg=cfgv, generator=g)
                        allg.append(ids)
                        write_grid(ids.cpu().numpy(), names, 8, pal, a.samples / nm / f"{group}_x8_cfg{cfgv:g}.png")
                rg, _, _ = M.boundary_ratio(torch.cat(allg)); rr, _, _ = M.boundary_ratio(sets["iid"][1])
                r.update({"boundary_ratio_gen": rg, "boundary_ratio_real": rr, "boundary_artifact_ratio": rg / rr})
        res[nm] = r; print(nm, json.dumps(r), flush=True)
        del model
    res["_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
