#!/usr/bin/env python3
"""Opt-in sd.cpp sheet benchmark, with optional dedicated server startup."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
from contextlib import contextmanager
import csv
import errno
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import socket
import statistics
import subprocess
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

SIGMAS = [1.0, 0.92664734, 0.82570166, 0.67798672, 0.44119975, 0.0]
GPU_FIELDS = ["utilization.gpu", "memory.used", "power.draw", "temperature.gpu",
              "clocks.sm", "clocks.mem", "pstate"]
REASON_NAMES = ("sw_power_cap", "hw_slowdown", "sw_thermal_slowdown",
                "hw_thermal_slowdown", "hw_power_brake_slowdown")


def normalized(text):
    return " ".join("".join(c if c.isalnum() else " " for c in
                           unicodedata.normalize("NFKC", text).casefold()).split())


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def loopback_url(value):
    p = urllib.parse.urlsplit(value)
    if (p.scheme != "http" or p.hostname not in ("127.0.0.1", "localhost", "::1")
            or p.username or p.password or p.query or p.fragment):
        raise ValueError("server must be an HTTP loopback URL without credentials")
    return value.rstrip("/")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("redirect refused")


def request(url, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(req, timeout=timeout) as response:
        return response.status, json.load(response)


def read_prompts(path, holdout_paths, required):
    holds = []
    for hp in holdout_paths:
        values = json.loads(hp.read_text(encoding="utf-8"))
        if not isinstance(values, list) or len(values) != 50 or not all(isinstance(x, str) and normalized(x) for x in values):
            raise ValueError("each holdout must contain 50 nonempty strings")
        holds.extend(normalized(x) for x in values)
    prompts = [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    keys = [normalized(x) for x in prompts]
    if len(set(keys)) != len(keys) or any(not k for k in keys):
        raise ValueError("prompts must be nonempty and unique after punctuation normalization")
    if any(" " + h + " " in " " + k + " " for k in keys for h in holds):
        raise ValueError("prompt contains a held-out phrase; supply another corpus")
    if len(prompts) < required:
        raise ValueError(f"need at least {required} unique full sheet prompts")
    return prompts[:required]


def extract_v2(image, out, module):
    import numpy as np
    array = np.asarray(image.convert("RGB"))
    accepted = 0
    for row in range(3):
        for col in range(3):
            cell = module.extract_cell(array, 3, row, col)
            accepted += module.usable(cell["score"], cell["purity"])
            rgba = np.dstack([cell["rgb"], cell["alpha"]])
            Image.fromarray(rgba).save(out / f"cell-{row}-{col}.png")
    return accepted


def generate(server, body, out, timeout, extractor=None):
    start = time.perf_counter()
    status, submit = request(server + "/sdcpp/v1/img_gen", body, min(timeout, 30))
    if status != 202:
        raise ValueError("unexpected submission status")
    poll = urllib.parse.urljoin(server + "/", submit.get("poll_url") or
                                f"/sdcpp/v1/jobs/{submit['id']}")
    if urllib.parse.urlsplit(poll)[:2] != urllib.parse.urlsplit(server)[:2]:
        raise ValueError("cross-origin polling refused")
    while True:
        remaining = timeout - (time.perf_counter() - start)
        if remaining <= 0:
            raise TimeoutError("job deadline")
        _, job = request(poll, timeout=min(remaining, 30))
        if job.get("status") == "completed":
            encoded = job["result"]["images"][0]["b64_json"]
            break
        if job.get("status") in ("failed", "cancelled"):
            raise RuntimeError("generation failed")
        time.sleep(min(0.05, remaining))
    requested = time.perf_counter()
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        if source.size != (body["width"], body["height"]) or source.format != "PNG":
            raise ValueError("wrong output dimensions or format")
        image = source.copy()
        image.info.clear()
    decoded = time.perf_counter()
    out.mkdir()
    image.save(out / "sheet.png", optimize=True)
    compressed = time.perf_counter()
    accepted = extract_v2(image, out, extractor) if extractor else None
    end = time.perf_counter()
    return {"request_s": requested - start, "decode_s": decoded - requested,
            "compress_write_s": compressed - decoded,
            "extract_write_s": end - compressed if extractor else None,
            "total_s": end - start, "png_bytes": len(raw), "accepted_cells": accepted}


class GpuLog:
    def __init__(self, out, gpu, origin=None):
        self.out, self.gpu = out, gpu
        self.origin = time.perf_counter() if origin is None else origin
        self.stop = threading.Event()
        self.errors = 0
        self.samples = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def reasons(self):
        try:
            result = subprocess.run(["nvidia-smi", "--help-query-gpu"], capture_output=True,
                                    text=True, timeout=5, check=True)
            for prefix in ("clocks_event_reasons.", "clocks_throttle_reasons."):
                fields = [prefix + name for name in REASON_NAMES if re.search(
                    r"(?<![\w.])" + re.escape(prefix + name) + r"(?![\w.])", result.stdout)]
                if fields:
                    return fields
        except (OSError, subprocess.SubprocessError):
            self.errors += 1
        return []

    def run(self):
        optional = self.reasons()
        fields = GPU_FIELDS + optional
        query_fields = fields
        with self.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["elapsed_s"] + fields)
            while not self.stop.is_set():
                try:
                    result = subprocess.run(["nvidia-smi", "-i", self.gpu,
                        "--query-gpu=" + ",".join(query_fields), "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5, check=True)
                    values = next(csv.reader([result.stdout.strip()]))
                    if len(values) != len(query_fields):
                        raise ValueError("invalid GPU response")
                    values = [v.strip() for v in values]
                    for i, value in enumerate(values):
                        if i >= len(GPU_FIELDS) and value in ("Active", "Not Active"):
                            values[i] = "1" if value == "Active" else "0"
                        elif not re.fullmatch(r"[\d.N/A \[\]P-]+", value):
                            raise ValueError("invalid GPU response")
                    writer.writerow([round(time.perf_counter() - self.origin, 3)] + values
                                    + ["N/A"] * (len(fields) - len(values)))
                    handle.flush()
                    self.samples += 1
                except (OSError, subprocess.SubprocessError, ValueError, StopIteration, csv.Error):
                    self.errors += 1
                    query_fields = GPU_FIELDS
                self.stop.wait(2)


def safe_server_timings(path, offset, out):
    """Copy only allowlisted phase names and numeric seconds, never log lines."""
    patterns = {
        "text": r"(?:text encoding|conditioning).*?(\d+(?:\.\d+)?)\s*s\b",
        "sampling": r"(?:sampling|sample).*?(\d+(?:\.\d+)?)\s*s\b",
        "vae": r"(?:decoding|vae decode).*?(\d+(?:\.\d+)?)\s*s\b",
    }
    with path.open("rb") as src, out.open("w", encoding="utf-8") as dest:
        src.seek(offset)
        for raw in src:
            line = raw.decode("utf-8", errors="replace")
            for phase, pattern in patterns.items():
                match = re.search(pattern, line, re.I)
                if match:
                    dest.write(json.dumps({"phase": phase, "seconds": float(match[1])}) + "\n")


def idle_gpu(gpu):
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("unset CUDA_VISIBLE_DEVICES before selecting a physical GPU")
    result = subprocess.run(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
                             "--format=csv,noheader,nounits"], capture_output=True,
                            text=True, check=True, timeout=10)
    if any(line.strip() and line.strip() != str(os.getpid()) for line in result.stdout.splitlines()):
        raise ValueError("selected GPU is busy; dedicated benchmark server refused")


def server_command(args, url):
    if args.flash_attn is None or args.flash_attn_declared is not None or args.server_args is None:
        raise ValueError("--sd-server requires --server-args and --flash-attn, without --flash-attn-declared")
    if args.server_log is not None:
        raise ValueError("--server-log is only for an existing server")
    if not args.sd_server.is_file():
        raise ValueError("missing supplied sd-server executable")
    extra = json.loads(args.server_args.read_text(encoding="utf-8-sig"))
    if not isinstance(extra, list) or not all(isinstance(x, str) and x and "\x00" not in x for x in extra):
        raise ValueError("server args must be a JSON array of nonempty argument strings")
    owned = {"--flash-attn", "--diffusion-fa", "--vae-fa", "--listen-ip", "--listen-port"}
    if any(x.split("=", 1)[0] in owned for x in extra):
        raise ValueError("server args must not override benchmark flash or loopback settings")
    parsed = urllib.parse.urlsplit(url)
    if parsed.hostname != "127.0.0.1" or parsed.port is None or not 1 <= parsed.port <= 65535:
        raise ValueError("dedicated server requires an explicit IPv4 loopback port")
    return [str(args.sd_server), *extra, "--listen-ip", "127.0.0.1",
            "--listen-port", str(parsed.port), "--flash-attn", args.flash_attn]


def wait_port_free(port, deadline):
    while True:
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
            return
        except OSError as error:
            if error.errno != errno.EADDRINUSE and getattr(error, "winerror", None) != 10048:
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("benchmark loopback port did not become free") from None
            time.sleep(min(.2, remaining))


@contextmanager
def server_session(args, url):
    if args.sd_server is None:
        if args.flash_attn is not None or args.server_args is not None:
            raise ValueError("--flash-attn applies only with --sd-server; existing servers use --flash-attn-declared")
        yield {"server_mode": "existing", "flash_attn_operator_declared": args.flash_attn_declared,
               "flash_attn_startup_argument": None, "flash_attn_effective_verified": False}
        return
    command = server_command(args, url)
    idle_gpu(args.gpu)
    deadline = time.monotonic() + args.startup_timeout
    wait_port_free(urllib.parse.urlsplit(url).port, deadline)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=args.gpu)
    # Raw startup logs contain asset paths. Never copy them into benchmark artifacts.
    child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        while True:
            if child.poll() is not None:
                raise RuntimeError("dedicated benchmark server exited before readiness")
            try:
                _, status = request(url + "/sdcpp/v1/models/status", timeout=1)
                if status.get("state") == "ready":
                    break
            except (OSError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("dedicated benchmark server readiness deadline")
            time.sleep(.2)
        yield {"server_mode": "dedicated_child", "flash_attn_operator_declared": None,
               "flash_attn_startup_argument": args.flash_attn, "flash_attn_effective_verified": False}
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)


def run(args):
    server = loopback_url(args.server)
    if urllib.parse.urlsplit(server).path:
        raise ValueError("server URL must not have a path")
    if not re.fullmatch(r"\d+", args.gpu):
        raise ValueError("GPU must be one explicit numeric index")
    if args.size not in (512, 576) or args.trials < 1 or args.sheets < 1 or args.warmup < 1:
        raise ValueError("invalid workload size")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not math.isfinite(args.lora_mult):
        raise ValueError("timeout and LoRA multiplier must be finite; timeout must be positive")
    if not math.isfinite(args.startup_timeout) or not 0 < args.startup_timeout <= 600:
        raise ValueError("startup timeout must be positive and at most 600 seconds")
    if args.out.exists():
        raise FileExistsError("benchmark output exists; choose a new folder")
    if not args.inflight or any(i not in (1, 2, 3) for i in args.inflight) or len(set(args.inflight)) != len(args.inflight):
        raise ValueError("inflight must contain distinct values from 1,2,3")
    count = args.warmup + args.trials * len(args.inflight) * args.sheets
    prompts = read_prompts(args.prompts, [args.blind50, args.eval50], count)
    if args.extract_v2 and args.size != 576:
        raise ValueError("extract_v2 is supported only for 3x3 at 576")
    extractor = None
    if args.extract_v2:
        spec = importlib.util.spec_from_file_location("pixie_extract_v2", Path(__file__).parents[1] / "extract_v2.py")
        extractor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extractor)
    with server_session(args, server) as settings:
        return run_ready(args, server, prompts, extractor, settings)


def run_ready(args, server, prompts, extractor, settings):
    _, model = request(server + "/sdcpp/v1/models/status")
    if model.get("state") != "ready":
        raise ValueError("server model is not ready")
    offset = args.server_log.stat().st_size if args.server_log else None
    args.out.mkdir(parents=True, exist_ok=False)
    config = {"schema": 2, **settings,
        "conditioning_cache_operator_declared": args.conditioning_cache,
        "settings_verified": False, "gpu_index": int(args.gpu),
        "size": args.size, "steps": 5, "lora_multiplier": args.lora_mult,
        "lora_identifier_sha256": digest(args.lora), "prompt_set_sha256": digest("\n".join(prompts)),
        "seed": args.seed, "trials": args.trials, "sheets_per_trial": args.sheets,
        "inflight": args.inflight, "warmup_sheets": args.warmup,
        "extraction": "extract_v2_3x3_576" if extractor else "not_measured",
        "cache_hit_status": "unknown; prompts unique within this invocation"}
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    origin = time.perf_counter()
    gpu = GpuLog(args.out / "gpu.csv", args.gpu, origin)
    gpu.thread.start()
    rows, trials = [], []
    cursor = 0
    try:
        with (args.out / "requests.jsonl").open("w", encoding="utf-8") as events:
            def group(phase, trial, inflight, sheets):
                nonlocal cursor
                indexes = list(range(cursor, cursor + sheets))
                cursor += sheets
                start = time.perf_counter()
                def one(index):
                    body = {"prompt": prompts[index], "negative_prompt": "", "width": args.size,
                        "height": args.size, "seed": args.seed + index, "batch_count": 1,
                        "sample_params": {"sample_method": "euler", "sample_steps": 5,
                            "custom_sigmas": SIGMAS, "guidance": {"txt_cfg": 1.0}},
                        "lora": [{"path": args.lora, "multiplier": args.lora_mult}],
                        "preview": "none", "vae_tiling_params": {"enabled": False},
                        "output_format": "png", "embed_image_metadata": False}
                    row = {"phase": phase, "trial": trial, "inflight": inflight,
                           "index": index, "prompt_sha256": digest(prompts[index]), "ok": False}
                    row["start_s"] = time.perf_counter() - origin
                    try:
                        row.update(generate(server, body, args.out / f"sheet-{index:05d}", args.timeout, extractor))
                        row["request_end_s"] = row["start_s"] + row["request_s"]
                        row["ok"] = True
                    except Exception as error:
                        row["error_type"] = type(error).__name__
                    row["end_s"] = time.perf_counter() - origin
                    return row
                with concurrent.futures.ThreadPoolExecutor(max_workers=inflight) as pool:
                    done = list(pool.map(one, indexes))
                wall = time.perf_counter() - start
                for row in done:
                    events.write(json.dumps(row) + "\n")
                events.flush()
                rows.extend(done)
                success = sum(row["ok"] for row in done)
                result = {"phase": phase, "trial": trial, "inflight": inflight,
                    "wall_s": wall, "requested": sheets, "completed": success,
                    "errors": sheets - success, "sheets_per_s": success / wall,
                    "comparison_valid": success == sheets}
                trials.append(result)
                print(json.dumps(result), flush=True)
            group("warmup", -1, 1, args.warmup)
            for trial in range(args.trials):
                order = args.inflight if trial % 2 == 0 else list(reversed(args.inflight))
                for inflight in order:
                    group("measured", trial, inflight, args.sheets)
    finally:
        gpu.stop.set()
        gpu.thread.join(timeout=6)
    if args.server_log:
        safe_server_timings(args.server_log, offset, args.out / "server-phases.jsonl")
    summaries = []
    for inflight in args.inflight:
        values = [t["sheets_per_s"] for t in trials if t["phase"] == "measured" and t["inflight"] == inflight]
        summaries.append({"inflight": inflight, "median_sheets_per_s": statistics.median(values),
                          "range_sheets_per_s": [min(values), max(values)]})
    summary = {"requested": len(rows), "completed": sum(r["ok"] for r in rows),
        "errors": sum(not r["ok"] for r in rows), "gpu_samples": gpu.samples,
        "gpu_errors": gpu.errors, "comparison_valid": all(r["ok"] for r in rows),
        "trials": trials, "summaries": summaries}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["comparison_valid"] else 1


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--prompts", type=Path, required=True, help="one complete unique sheet prompt per line")
    ap.add_argument("--blind50", type=Path, required=True)
    ap.add_argument("--eval50", type=Path, required=True)
    ap.add_argument("--lora", required=True, help="operator-provided server-side LoRA identifier")
    ap.add_argument("--lora-mult", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sd-server", type=Path, help="start this supplied executable as a dedicated benchmark child")
    ap.add_argument("--server-args", type=Path, help="JSON argv array for the dedicated server's model/startup settings")
    ap.add_argument("--startup-timeout", type=float, default=180)
    ap.add_argument("--flash-attn", choices=("on", "off", "auto"), help="real startup flag for --sd-server")
    ap.add_argument("--flash-attn-declared", choices=("on", "off", "auto", "unknown"), default=None,
                    help="unverified label for an existing server; does not change that server")
    ap.add_argument("--conditioning-cache", choices=("off", "fresh-server", "unknown"), required=True)
    ap.add_argument("--inflight", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--sheets", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--extract-v2", action="store_true")
    ap.add_argument("--server-log", type=Path)
    return ap


if __name__ == "__main__":
    try:
        raise SystemExit(run(parser().parse_args()))
    except Exception as error:
        # Server messages, user paths and prompts can contain sensitive data.
        print(f"benchmark failed: {type(error).__name__}")
        raise SystemExit(1)
