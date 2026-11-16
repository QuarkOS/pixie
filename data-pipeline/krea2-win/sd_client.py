#!/usr/bin/env python3
"""Talk to a local sd-server (sdcpp /v1/img_gen) and write one PNG."""
import argparse, base64, json, sys, time, urllib.error, urllib.request

SIGMAS = [1.0, 0.92664734, 0.82570166, 0.67798672, 0.44119975, 0.0]

def req(url, method="GET", body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw.decode()) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            parsed = json.loads(raw.decode()) if raw else {}
        except Exception:
            parsed = {"error": raw.decode("utf-8", "replace")}
        return e.code, parsed

def wait_ready(base, timeout=300):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            code, st = req(base + "/sdcpp/v1/models/status", timeout=5)
            last = (code, st)
            if code == 200 and st.get("state") == "ready":
                return st
            if code == 200 and st.get("state") == "error":
                raise SystemExit(f"server model error: {st}")
        except urllib.error.URLError:
            pass
        time.sleep(0.5)
    raise SystemExit(f"server not ready after {timeout}s; last={last}")

def make_prompt(tl, tr, bl, br):
    return (
        "pixel art 16x16 game icons, 2x2 sprite sheet, "
        f"top left: {tl}, top right: {tr}, bottom left: {bl}, bottom right: {br}, "
        "dark outline, plain magenta background"
    )

def generate(base, prompt, seed, out_path):
    body = {
        "prompt": prompt,
        "negative_prompt": "",
        "width": 512,
        "height": 512,
        "seed": int(seed),
        "batch_count": 1,
        "sample_params": {
            "sample_method": "euler",
            "sample_steps": 5,
            "custom_sigmas": SIGMAS,
            "guidance": {"txt_cfg": 1.0},
        },
        "lora": [{"path": "checkpoint-1000.safetensors", "multiplier": 1.0}],
        "vae_tiling_params": {"enabled": False},
        "output_format": "png",
        "embed_image_metadata": True,
    }
    code, submitted = req(base + "/sdcpp/v1/img_gen", method="POST", body=body, timeout=30)
    if code != 202:
        raise SystemExit(f"img_gen submit failed HTTP {code}: {submitted}")
    job_id = submitted["id"]
    poll = submitted.get("poll_url") or f"/sdcpp/v1/jobs/{job_id}"
    if not poll.startswith("http"):
        poll = base + poll
    t0 = time.time()
    while time.time() - t0 < 600:
        code, job = req(poll, timeout=30)
        if code not in (200, 202):
            raise SystemExit(f"job poll failed HTTP {code}: {job}")
        status = job.get("status")
        if status == "completed":
            images = job.get("result", {}).get("images") or []
            if not images:
                raise SystemExit(f"completed job had no images: {job}")
            png = base64.b64decode(images[0]["b64_json"])
            with open(out_path, "wb") as f:
                f.write(png)
            return job
        if status in ("failed", "cancelled"):
            raise SystemExit(f"job {status}: {job}")
        time.sleep(0.25)
    raise SystemExit(f"job {job_id} timed out")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:17890")
    ap.add_argument("--wait-ready", action="store_true")
    ap.add_argument("--tl")
    ap.add_argument("--tr")
    ap.add_argument("--bl")
    ap.add_argument("--br")
    ap.add_argument("--prompt")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--out", required=False)
    args = ap.parse_args()
    if args.wait_ready:
        st = wait_ready(args.base)
        print(json.dumps(st))
        return
    if not args.out:
        sys.exit("--out is required unless --wait-ready")
    prompt = args.prompt or make_prompt(args.tl, args.tr, args.bl, args.br)
    job = generate(args.base, prompt, args.seed, args.out)
    # compact timing-friendly result (no b64)
    slim = {k: job.get(k) for k in ("id", "status", "created", "started", "completed")}
    print(json.dumps(slim))

if __name__ == "__main__":
    main()
