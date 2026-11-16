# Measure Krea sheet throughput

`krea_unique.py` measures a loopback-only sd.cpp server. It can start a separate
benchmark server and pass flash attention as a real startup argument, or attach
to an operator-started server with explicitly unverified settings.
It leaves the running generation pipeline and production configuration unchanged.
No GPU performance gains have been measured for this change. CPU mock tests
verify the protocol, accounting, holdout guard, and output files.

## Run a short comparison

Use a separate idle GPU and dedicated server, with the same model, LoRA, quantization,
5-step Euler sigmas, dimensions and backend placement as the baseline. Do not
benchmark on a GPU running training or the chat worker. Use the project venv;
the base benchmark requires Pillow. Optional extraction also requires the
existing NumPy/SciPy extraction dependencies.

The protocol benchmark needs only Pillow. Install extraction extras only in the
project venv when running `--extract-v2`, for example
`"$PY" -m pip install numpy scipy`. The single real-extractor CPU test skips when
these optional dependencies are absent; the protocol and server tests still run.

Prepare a local JSON array containing the server's model arguments. Use the same
weights and at-runtime LoRA settings as the baseline. The file is an argv array,
so paths with spaces do not require shell quoting. The benchmark does not expand
environment-variable strings inside JSON. For example, build the file from
operator-supplied environment variables with the project Python:

```sh
"$PY" - <<'PY'
import json, os
args = ["--diffusion-model", os.environ["KREA_DIFFUSION"],
        "--llm", os.environ["KREA_LLM"], "--vae", os.environ["KREA_VAE"],
        "--lora-model-dir", os.environ["KREA_LORA_DIR"],
        "--lora-apply-mode", "at_runtime", "--conditioning-cache-size", "64",
        "--disable-image-metadata"]
with open(os.environ["KREA_BENCH_ARGS"], "x", encoding="utf-8") as output:
    json.dump(args, output)
PY
```

Do not include `--flash-attn`, legacy flash flags or `--listen-ip`/`--listen-port`
in that file. The benchmark owns those arguments and rejects conflicting values.

Supply a text file containing **92 distinct complete sheet prompts**, one per
line, for the default 2 warmup sheets + 5 trials × 3 concurrency settings × 6
sheets. Use representative subjects and the normal magenta-background sheet
template. Prompts are checked against both frozen holdouts, including punctuation
normalization; the benchmark refuses a corpus containing a held-out phrase.
Prompt texts and local input paths are omitted from benchmark reports.

```sh
"$PY" data-pipeline/benchmarks/krea_unique.py \
  --server http://127.0.0.1:17890 --gpu 0 \
  --prompts "$DATA_ROOT/bench/sheet-prompts.txt" \
  --blind50 holdout/blind-50.json --eval50 "$DATA_ROOT/holdout/eval-50.json" \
  --lora "$KREA_LORA" --size 512 \
  --sd-server "$SD_SERVER" --server-args "$KREA_BENCH_ARGS" \
  --flash-attn off --conditioning-cache fresh-server \
  --out "$RUNS/krea-fa-off-01"
```

This starts the supplied executable with `--flash-attn off` on an idle explicitly
selected physical GPU, waits for its model to be ready, runs the benchmark and
stops only that child process. Unset `CUDA_VISIBLE_DEVICES` before running. The
child receives the selected GPU through its own environment; the parent and any
existing process keep their environment. The loopback port must be unused. The
output folder must be new. Model loading is excluded from throughput timing.
Startup is bounded by `--startup-timeout`, default 180 seconds and maximum 600.
That same deadline includes waiting for the loopback port to become free after
the previous server stops. Address-in-use errors retry every 0.2 seconds. Other
socket errors fail immediately. The benchmark never reuses an occupied port,
stops another process or extends the configured deadline.

For an already-started dedicated server, omit `--sd-server` and `--server-args`,
and use `--flash-attn-declared unknown` or the setting you independently observed.
This mode does not select the server's GPU, alter its settings or stop it.
`--flash-attn` is rejected in this mode, so the old declaration cannot silently
look like an applied setting. Only this mode accepts `--server-log` for numeric
phase parsing. Dedicated-child startup output is discarded because raw server
messages contain local asset paths; argv values are never copied into reports.

For flash on/off comparisons, repeat the dedicated-child command with
`--flash-attn on` and another new output folder. Keep the JSON args file unchanged.
Use `--size 576 --extract-v2` to additionally time the locked 3×3 extractor and
write all nine RGBA cells per sheet, recording the number accepted by its current
gates. The 2×2 benchmark does **not** time extraction: `extract_v2` was calibrated
for 3×3 at 576. No gate is changed.

The script persists:

- `requests.jsonl`: successful and failed requests; server round-trip (including
  queueing and polling), PNG decode, lossless PNG compression/write, and optional
  extract/write timing. Each output is decoded and dimension-checked.
- `summary.json`: end-to-end sheets/s, counts, five-trial median and range for
  each in-flight setting. Warmup is separate; trial order reverses each round.
  Any request failure invalidates the comparison and returns a nonzero exit.
- `gpu.csv`: selected-GPU utilization, memory MiB, watts, temperature, SM/memory
  clocks MHz and P-state sampled every two seconds. Missing telemetry is counted
  separately; a throughput run without it cannot establish the GPU limiter.
  Clock power/thermal/slowdown reasons are queried only when the installed
  `nvidia-smi --help-query-gpu` advertises them. `Active` is stored as 1 and
  `Not Active` as 0. Unsupported reasons remain absent or `N/A`; a failed optional
  query falls back to the original GPU fields instead of losing all telemetry.
- `server-phases.jsonl`, if a log is supplied: numeric text/sampling/VAE seconds
  parsed only from new log content. No raw server lines or paths are copied.
  Empty output means the log format was not recognized; phase timings remain
  unmeasured. The server must be dedicated: other jobs would contaminate phases.

Run the same corpus and seed on each GPU. Each request inside an invocation uses
a different prompt; all settings therefore see different prompt blocks. Reversed
order reduces drift but does not equalize content. Confirm any small difference
with separate `--inflight 1` / `2` / `3` invocations using the same corpus and
fresh servers. With one concurrency setting the first 32 prompts are used.
Repeating these matched runs at least five times permits a fair content comparison.
Record the exact binary/model file hashes separately using sanitized labels.

Request `start_s`, `request_end_s` and `end_s` now share the monotonic origin of
`gpu.csv`. Use [the slow-streak guide](krea-telemetry.md) to compare request
latency windows with clocks, power and slowdown reasons, or to monitor an
already-running generation job without restarting it.

## Flash attention and the 5070 Ti gap

The original benchmark treated `--flash-attn` as a declaration, so an operator
observed a declared `off` run whose server actually chose `auto` then `on`. Such
a run does not establish a flash-on/off comparison. The option now requires
dedicated-child mode and reaches the server's startup argv. Reports distinguish
`flash_attn_startup_argument` from `flash_attn_operator_declared`; they do not
claim that a backend selected a particular kernel without runtime evidence.
`--conditioning-cache` remains an operator declaration. A new child starts a
fresh cache; to disable caching, put a supported conditioning-cache startup flag
in the JSON args file. A unique
string is not proof of a cache miss after tokenizer normalization/truncation;
reports explicitly mark cache-hit status unknown. Never compare a cached repeated
prompt result to the unique-prompt production workload.

The stack notes report 4.805 s/sheet on the 5070 Ti and 3.07 s/sheet on the
3080 Ti at 2×2. Those measurements used different concurrency/OS conditions and
lack GPU telemetry, so they do not identify a cause. Test these in order:

| Priority | Test | Evidence that would explain the gap | Expected gain |
|---|---|---|---|
| 1 | Match in-flight 1/2/3, prompts, seeds and model/build hashes | Wall throughput vs request latency separates queueing from useful overlap | Untested; may help or hurt |
| 2 | Flash on/off with fresh server, unchanged other flags | Sampling phase changes with steady clocks and memory | Untested; architecture/backend dependent |
| 3 | Verify CUDA build supports the GPU and compare backend placement | Low GPU utilization plus long text/VAE phase or CPU buffers indicates fallback | Untested; no fallback has been demonstrated |
| 4 | Compare clocks/power/P-state and free VRAM during steady state | Throttling or shared-memory pressure correlates with slow requests | Untested; do not change power limits blindly |
| 5 | Time compression/extraction and inspect CPU scheduling, storage, other load | GPU idle gaps and CPU/write phases grow with concurrency | Untested; existing notes already found Windows priority starvation |
| 6 | Test direct convolution on/off, or at-runtime LoRA alternatives supported by the pinned build | Isolated VAE/sampling changes with identical quality checks | Untested; needs additional memory and compatibility checks |

Keep the current IQ4_XS diffusion, Q4_K_M text encoder, LoRA multiplier, guidance,
resolution, steps and VAE tiling fixed initially. Moving the text encoder to CPU
or enabling VAE tiling can help memory limits while adding work; the 3060 Ti's
documented fit settings are not a fair default for the 16 GB card.

The inspected public sd.cpp revision is `984eb06`. Its
[CLI documentation](https://github.com/leejet/stable-diffusion.cpp/blob/984eb06/examples/cli/README.md)
documents `--flash-attn on|off|auto`; its
[performance guide](https://github.com/leejet/stable-diffusion.cpp/blob/984eb06/docs/performance.md)
says both flash attention and direct convolution depend on architecture/backend
and recommends repeated comparisons separating startup from steady state. Check
the installed binary's `--help` before changing flags: older builds differ.
Its [server documentation](https://github.com/leejet/stable-diffusion.cpp/blob/984eb06/examples/server/README.md)
places backend and memory options at startup, where they apply to model context
creation. Flash attention is not a per-image generation argument in this build.

## CPU checks

```sh
"$PY" -m unittest discover -s data-pipeline/benchmarks -p test_krea_unique.py -v
```

These use a local mock HTTP server, a real child Python process standing in for
sd-server, and tiny synthetic PNGs. The child verifies that `off` actually
reaches its argv and that cleanup stops only the child, including after an error.
The tests also verify refusal of conflicting args and occupied/unknown GPUs.
Their execution
times are not estimates of generation speed or expected hardware gains.
