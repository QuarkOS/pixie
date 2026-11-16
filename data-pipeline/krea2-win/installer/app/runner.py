#!/usr/bin/env python3
"""Krea 2 sprite runner: reads subjects, sends 2x2 @ 512 sheet jobs to sd-server (N in flight),
extracts each cell to a true 32x32 RGBA PNG (magenta keyed out), and logs stats for the dashboard.
Resumable: sheets already in data/sheets are skipped.
usage: runner.py --config config.json   |   runner.py --replay DIR  (test mode, no GPU)"""
import argparse, base64, http.client, json, os, shutil, socket, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract as X
from pitch import pitch_of

SIGMAS5 = [1.0, 0.92664734, 0.82570166, 0.67798672, 0.44119975, 0.0]
LOCK = threading.Lock()
# errors meaning "sd-server is down/restarting" -> wait and retry the SAME sheet, never count as failed
CONN_ERRORS = (urllib.error.URLError, ConnectionError, TimeoutError, socket.timeout, http.client.HTTPException, OSError)

class ServerGone(Exception): pass

def keep_awake(on):
    """Windows: block system sleep while running (no admin); cleared on exit. No-op elsewhere."""
    if os.name != "nt": return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if on else 0x80000000)
    except Exception:
        pass

def safe_replace(src, dst, tries=40):
    """os.replace with retries: Defender/indexer can briefly lock fresh files on Windows (WinError 5/32)."""
    for k in range(tries):
        try:
            os.replace(src, dst); return
        except PermissionError:
            if k == tries - 1: raise
            time.sleep(0.05 * (k + 1))

def req(url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return f.status, json.loads(f.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:500]}
    except CONN_ERRORS + (ValueError,) as e:  # refused / reset / timeout / truncated reply -> server down
        raise ServerGone(f"{type(e).__name__}: {e}")

def wait_ready(base, timeout=1800):
    t0 = time.time()
    while timeout is None or time.time() - t0 < timeout:
        try:
            code, st = req(base + "/sdcpp/v1/models/status", timeout=5)
            if code == 200 and st.get("state") == "ready": return
            if code == 200 and st.get("state") == "error": raise SystemExit(f"model error: {st}")
        except ServerGone:
            pass
        time.sleep(2)
    raise SystemExit("sd-server not ready")

def prompt_2x2(items, px):
    tl, tr, bl, br = items
    return (f"pixel art {px}x{px} game icons, 2x2 sprite sheet, top left: {tl}, top right: {tr}, "
            f"bottom left: {bl}, bottom right: {br}, dark outline, plain magenta background")

def generate(base, cfg, prompt, seed, out_path):
    body = {"prompt": prompt, "negative_prompt": "", "width": cfg["size"], "height": cfg["size"],
            "seed": int(seed), "batch_count": 1,
            "sample_params": {"sample_method": "euler", "sample_steps": cfg["steps"], "custom_sigmas": SIGMAS5,
                              "guidance": {"txt_cfg": 1.0}},
            "lora": [{"path": cfg["lora"], "multiplier": cfg["lora_mult"]}], "preview": "none",
            "vae_tiling_params": {"enabled": False}, "output_format": "png", "embed_image_metadata": False}
    while True:  # loops only while the server is unreachable/restarting; real job failures raise RuntimeError
        try:
            return _generate_once(base, body, out_path)
        except ServerGone:
            wait_ready(base, timeout=None)

def _generate_once(base, body, out_path):
    code, sub = req(base + "/sdcpp/v1/img_gen", body, timeout=60)
    if code != 202:
        if code >= 500 or code in (404, 408, 429): raise ServerGone(f"submit HTTP {code}")
        raise RuntimeError(f"submit failed {code}: {sub}")
    poll = sub.get("poll_url") or f"/sdcpp/v1/jobs/{sub['id']}"
    if not poll.startswith("http"): poll = base + poll
    t0 = time.time()
    while time.time() - t0 < 900:
        code, job = req(poll, timeout=30)
        if code == 404 or code >= 500: raise ServerGone(f"poll HTTP {code} (server restarted?)")
        st = job.get("status")
        if st == "completed":
            png = base64.b64decode(job["result"]["images"][0]["b64_json"])
            tmp = out_path + ".tmp"
            with open(tmp, "wb") as f: f.write(png)
            safe_replace(tmp, out_path); return
        if st in ("failed", "cancelled"): raise RuntimeError(f"job {st}: {job}")
        time.sleep(0.05)
    raise ServerGone("job timeout (900 s)")

def extract_sheet(path, items, g, px, outdir, shard="000"):
    with Image.open(path) as im: img = np.asarray(im.convert("RGB"))
    H, W = img.shape[:2]; res = []
    stem = os.path.splitext(os.path.basename(path))[0]
    for r in range(g):
        for c in range(g):
            k = r * g + c
            cell = img[round(H*r/g):round(H*(r+1)/g), round(W*c/g):round(W*(c+1)/g)]
            s = X.score_cell(cell); s["spill"] = round(X.spill(img, g, r, c), 3)
            sp, pur = X.mode_pool(cell, px)
            alpha = np.where(X.is_bg(sp), 0, 255).astype(np.uint8)
            sp = X.palette_clean(sp, alpha)
            lab, n = ndimage.label(alpha > 0)
            if n > 1:
                sizes = ndimage.sum(alpha > 0, lab, range(1, n + 1))
                for i, sz in enumerate(sizes, 1):
                    if sz <= 1: alpha[lab == i] = 0
            fgb = alpha > 0
            pm = float(pur[fgb].mean()) if fgb.any() else 0.0
            ok = bool(X.usable(s, pm))
            safe = "".join(ch if ch.isalnum() else "_" for ch in items[k])[:60]
            name = f"{stem}-{k}-{safe}.png"
            sub = ("ok" if ok else "rejected") + "/" + shard
            os.makedirs(os.path.join(outdir, sub), exist_ok=True)
            Image.fromarray(np.dstack([sp, alpha]), "RGBA").save(os.path.join(outdir, sub, name))
            res.append({"file": f"{sub}/{name}", "subject": items[k], "usable": ok, "purity": round(pm, 3), **s})
    return res

def load_subjects(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"): continue
        out += [s.strip() for s in line.split("|") if s.strip()]
    return out

def log_event(data, ev):
    with LOCK:
        with open(os.path.join(data, "events.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")

def write_state(data, st):
    tmp = os.path.join(data, "state.json.tmp")
    with open(tmp, "w") as f: json.dump(st, f)
    safe_replace(tmp, os.path.join(data, "state.json"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json"))
    ap.add_argument("--replay", help="TEST: replay existing sheet PNGs + jobs.json from DIR instead of the GPU")
    ap.add_argument("--replay-delay", type=float, default=1.5)
    ap.add_argument("--data")
    a = ap.parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    cfg = json.load(open(a.config)) if os.path.exists(a.config) else {}
    cfg = {"size": 512, "steps": 5, "grid": 2, "px": 32, "lora": "pix32-final.safetensors", "lora_mult": 1.0,
           "in_flight": 3, "first_seed": 1000, "port": 17890, "subjects": "subjects.txt", **cfg}
    data = os.path.abspath(a.data or os.path.join(root, "data"))
    sheets, sprites = os.path.join(data, "sheets"), os.path.join(data, "sprites32")
    os.makedirs(sheets, exist_ok=True); os.makedirs(sprites, exist_ok=True)
    per = cfg["grid"] ** 2
    if a.replay:
        jobs = [j for j in json.load(open(os.path.join(a.replay, "jobs.json"))) if "warmup" not in j["out"]]
        work = [(i, j["items"], os.path.join(a.replay, os.path.basename(j["out"])), None, cfg["first_seed"] + i, None) for i, j in enumerate(jobs)]
        mode = "replay-test"
    elif cfg.get("queue"):
        qp = cfg["queue"] if os.path.isabs(cfg["queue"]) else os.path.join(root, cfg["queue"])
        work = []
        for line in open(qp, encoding="utf-8"):
            if not line.strip(): continue
            j = json.loads(line)
            work.append((j["sheet_id"], [c["phrase"] for c in j["cells"]], None, j["prompt"], j["seed"], j["cells"]))
        mode = "gpu"
    else:
        subs = load_subjects(os.path.join(root, cfg["subjects"]) if not os.path.isabs(cfg["subjects"]) else cfg["subjects"])
        subs += subs[: (-len(subs)) % per]  # pad last sheet
        work = [(i, subs[i*per:(i+1)*per], None, None, cfg["first_seed"] + i, None) for i in range(len(subs) // per)]
        mode = "gpu"
    total = len(work)
    done = set()
    evp = os.path.join(data, "events.jsonl")
    if os.path.exists(evp):
        for line in open(evp, encoding="utf-8"):
            try:
                e = json.loads(line)
                if "error" not in e: done.add(e["sheet"])
            except ValueError: pass
    done &= {w[0] for w in work}
    todo = [w for w in work if w[0] not in done]
    t_start = time.time()
    st = {"mode": mode, "total_sheets": total, "done_sheets": len(done), "per_sheet": per, "started": t_start,
          "status": "starting", "config": {k: cfg[k] for k in ("size", "steps", "grid", "px", "lora", "lora_mult", "in_flight")}}
    write_state(data, st)
    base = f"http://127.0.0.1:{cfg['port']}"
    if mode == "gpu":
        st["status"] = "waiting for sd-server (model loading)"; write_state(data, st)
        wait_ready(base, timeout=None)
    st["status"] = "running"; write_state(data, st)

    def attempt_sheet(i, items, src, prompt, seed, qcells, out):
        labels = []
        t0 = time.time()
        try:
            if src:
                time.sleep(a.replay_delay); shutil.copy(src, out + ".tmp"); safe_replace(out + ".tmp", out)
            else:
                generate(base, cfg, prompt or prompt_2x2(items, cfg["px"]), seed, out)
                if cfg.get("recompress_sheets", True):  # lossless, ~35% smaller than sd-server's PNG
                    with Image.open(out) as im: im.save(out + ".tmp.png", "PNG", optimize=True)
                    safe_replace(out + ".tmp.png", out)
            t1 = time.time()
            cells = extract_sheet(out, items, cfg["grid"], cfg["px"], sprites, f"{i // 1000:03d}")
            labels = [{"file": c["file"], "phrase": items[k], "sheet_id": i, "cell": k, "seed": seed,
                       "usable": c["usable"], "sheet": os.path.relpath(out, sheets).replace(os.sep, "/"),
                       **({"subject": qcells[k].get("subject"), "cat": qcells[k].get("cat"), "sid": qcells[k].get("sid")} if qcells else {})}
                      for k, c in enumerate(cells)]
            ev = {"t": time.time(), "sheet": i, "gen_s": round(t1 - t0, 3), "extract_s": round(time.time() - t1, 3),
                  "sprites": len(cells), "usable": sum(c["usable"] for c in cells), "cells": [{"file": c["file"], "subject": c["subject"], "usable": c["usable"]} for c in cells]}
        except Exception as e:  # keep going unattended
            ev = {"t": time.time(), "sheet": i, "error": str(e)[:400], "sprites": 0, "usable": 0}
        return ev, labels

    def one(w):
        i, items, src, prompt, seed, qcells = w
        os.makedirs(os.path.join(sheets, f"{i // 1000:03d}"), exist_ok=True)
        out = os.path.join(sheets, f"{i // 1000:03d}", f"sheet-{i:06d}.png")
        for attempt in range(3):  # unattended: if sd-server died, wait for the watchdog restart and retry
            ev, labels = attempt_sheet(i, items, src, prompt, seed, qcells, out)
            if "error" not in ev or mode != "gpu": break
            try: wait_ready(base, 1800)
            except BaseException: pass
        with LOCK:  # labels + event written together; a sheet only counts as done once both exist
            if "error" not in ev:
                with open(os.path.join(data, "labels.jsonl"), "a", encoding="utf-8") as f:
                    f.write("".join(json.dumps(r) + "\n" for r in labels))
            with open(os.path.join(data, "events.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(ev) + "\n")
            st["done_sheets"] += 0 if "error" in ev else 1
            write_state(data, st)

    with ThreadPoolExecutor(max_workers=max(1, int(cfg["in_flight"]))) as ex:
        list(ex.map(one, todo))
    fails = len(todo) - (st["done_sheets"] - len(done))
    st["status"] = "finished" if fails == 0 else f"finished with {fails} failed sheets (re-run Start to retry them)"
    st["finished"] = time.time(); write_state(data, st)
    print(json.dumps({"sheets": st["done_sheets"], "total": total, "seconds": round(time.time() - t_start, 1)}))

if __name__ == "__main__":
    keep_awake(True)
    try:
        main()
    finally:
        keep_awake(False)
