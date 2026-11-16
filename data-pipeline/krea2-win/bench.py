#!/usr/bin/env python3
"""Pipelined benchmark client for sd-server /sdcpp/v1/img_gen.
jobs file: JSON list of {prompt, seed, width, height, steps, sigmas, lora_mult, out}
--depth N keeps N jobs queued on the server (1 = old sequential behaviour).
--legacy reproduces the old client (embed metadata on, 0.25s poll, sequential)."""
import argparse, base64, json, os, sys, time, urllib.request, urllib.error, threading, subprocess

def req(url, method="GET", body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read(); return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        try: return e.code, json.loads(raw)
        except Exception: return e.code, {"error": raw.decode("utf-8", "replace")}

def body_for(j, legacy):
    b = {
        "prompt": j["prompt"], "negative_prompt": "",
        "width": j.get("width", 512), "height": j.get("height", 512),
        "seed": int(j["seed"]), "batch_count": 1,
        "sample_params": {"sample_method": "euler", "sample_steps": j["steps"],
                          "custom_sigmas": j["sigmas"], "guidance": {"txt_cfg": 1.0}},
        "vae_tiling_params": ({"enabled": True, "tile_size_w": j.get("vae_tile", 512), "tile_size_h": j.get("vae_tile", 512)} if j.get("vae_tile") else {"enabled": False}),
        "output_format": "png",
        "embed_image_metadata": bool(legacy),
    }
    if not legacy:
        b["preview"] = "none"
    if j.get("lora_mult", 1.0) > 0 and not j.get("no_lora"):
        b["lora"] = [{"path": j.get("lora", "checkpoint-1000.safetensors"), "multiplier": j.get("lora_mult", 1.0)}]
    return b

class VramSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.peak = 0; self.stop = False
    def run(self):
        p = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "100"],
                             stdout=subprocess.PIPE, text=True)
        for line in p.stdout:
            try: self.peak = max(self.peak, int(line.strip()))
            except ValueError: pass
            if self.stop: break
        p.kill()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs"); ap.add_argument("--base", default="http://127.0.0.1:17890")
    ap.add_argument("--depth", type=int, default=2); ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--warmup", type=int, default=1, help="jobs at start excluded from timing")
    ap.add_argument("--result", required=True)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    vs = VramSampler(); vs.start()
    poll_iv = 0.25 if a.legacy else 0.02
    pending = []  # (idx, poll_url, t_submit)
    done = {}
    nxt = 0
    t_first = None
    def submit(i):
        j = jobs[i]
        code, s = req(a.base + "/sdcpp/v1/img_gen", "POST", body_for(j, a.legacy))
        if code != 202: sys.exit(f"submit failed {code}: {s}")
        poll = s.get("poll_url") or f"/sdcpp/v1/jobs/{s['id']}"
        if not poll.startswith("http"): poll = a.base + poll
        pending.append((i, poll, time.time()))
    depth = 1 if a.legacy else a.depth
    while len(done) < len(jobs):
        while nxt < len(jobs) and len(pending) < depth:
            if nxt == a.warmup and t_first is None:
                # wait for warmup to drain so timing starts clean
                if pending: break
                t_first = time.time()
            submit(nxt); nxt += 1
        if not pending: continue
        i, poll, ts = pending[0]
        url = poll + ("?skip_current_image=true" if not a.legacy else "")
        code, job = req(url)
        st = job.get("status")
        if st == "completed":
            pending.pop(0)
            img = job["result"]["images"][0]["b64_json"]
            out = jobs[i]["out"]; os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as f: f.write(base64.b64decode(img))
            done[i] = {"server_ms": job.get("completed", 0) - job.get("started", 0), "t_done": time.time()}
        elif st in ("failed", "cancelled"):
            sys.exit(f"job {st}: {json.dumps(job)[:500]}")
        else:
            time.sleep(poll_iv)
    if t_first is None: t_first = time.time()
    t_end = time.time()
    vs.stop = True
    timed = [d for i, d in done.items() if i >= a.warmup]
    n = len(timed)
    res = {"n_timed": n, "wall_s": t_end - t_first, "s_per_sheet": (t_end - t_first) / max(n, 1),
           "server_ms_mean": sum(d["server_ms"] for d in timed) / max(n, 1), "peak_vram_mib": vs.peak,
           "depth": depth, "legacy": a.legacy}
    json.dump(res, open(a.result, "w"), indent=1); print(json.dumps(res))

if __name__ == "__main__":
    main()
