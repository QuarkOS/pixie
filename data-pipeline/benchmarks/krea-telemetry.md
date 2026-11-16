# Investigate intermittent generation slowdowns

The operator reports that about 40% of sheets on the 5070 Ti ran at roughly
half speed. This is a supplied observation, not a local measurement or a
demonstrated thermal, memory or CPU cause. Collect evidence before changing
power limits, scheduling, quantization or generation settings.

## Record an existing generation job

Start this read-only observer beside the runner, preferably before generation
starts. It never launches or stops a server, changes GPU settings or checks
that the GPU is idle. A busy GPU is the workload being observed.

```sh
"$PY" data-pipeline/benchmarks/krea_telemetry.py record \
  --gpu 0 --inflight 3 --seconds 1800 \
  --events "$RUNS/generation/events.jsonl" \
  --out "$RUNS/krea-telemetry-001"
```

Supply the existing runner's `events.jsonl` and its actual concurrency. The
observer copies only new complete event lines after startup. It persists
relative generation/extraction timings and numeric yield counts, without
subjects, cell filenames, error messages or input paths. Events for sheets
already in flight at startup can have negative start times and only partial
telemetry coverage. Do not mix changing concurrency/layout/model settings in
one recording. Truncating the source event file fails the observation.
Runner events use wall-clock timestamps, which are converted using the
observer's startup anchor. A system clock adjustment can misalign those events;
use the benchmark's monotonic request timestamps for a controlled comparison.

`gpu.csv` records utilization percent, used MiB, watts, GPU temperature Celsius,
SM/memory clocks MHz and P-state every two seconds. When supported, it also
records active clock power-cap, thermal, hardware-slowdown and power-brake
reasons. Field discovery uses the installed driver's help; unsupported reason
queries fall back to the basic fields. The summary counts telemetry errors.
No device names, UUIDs, host information or raw command output are persisted.

The observer has a duration bound and writes a fresh folder. Ctrl-C stops only
the observer and flushes its summary. `--events` can be omitted to record GPU
data alone. Generation continues normally when the observer exits. Observation
overhead has not been measured; compare a control run if a small effect matters.

## Analyze a recorded run or benchmark

```sh
"$PY" data-pipeline/benchmarks/krea_telemetry.py analyze \
  --run "$RUNS/krea-telemetry-001" --slow-ratio 1.75 \
  --out "$RUNS/krea-streak-analysis-001.json"
```

The same command accepts a fresh `krea_unique.py` benchmark output folder.
Older benchmark outputs without aligned request timestamps are refused.
Analysis excludes warmup and separates each declared in-flight setting.

The default baseline is the lower quartile of successful request durations.
It is a heuristic estimate of a fast population, not a measured ideal latency.
The default threshold labels durations at least 1.75 times that estimate as
slow. Supply `--baseline-seconds` from a controlled same-setting run when
available. If most of a run is slow, its own lower quartile may hide the issue.
Repeat with a few thresholds to check that the finding is robust. The script
does not assert that a bimodal distribution exists.

The output contains counts, slow fractions, consecutive slow completions and
normal/slow GPU summaries. Runner generation durations include server queueing
and PNG recompression, with extraction recorded separately. Benchmark request
durations include queueing and polling but exclude decode/compression/extraction.
Compare like scopes. A request under concurrency is **not** wall seconds per
sheet, and overlapping requests can share GPU samples in both populations.

Look for evidence in this order:

1. Lower SM/memory clocks with active power/thermal/slowdown flags during slow
   windows supports investigating clock limits. A power cap can also be normal;
   a flag alone does not prove it caused the latency.
2. Higher used VRAM and low utilization supports inspecting backend placement
   and Windows shared GPU memory. These logs do not measure offload directly.
3. Stable clocks with GPU idle gaps supports inspecting CPU text encoding,
   PNG work, process scheduling and storage. CPU utilization/RSS/I/O are not
   measured here, so use an operator-selected process profiler next.
4. Separate fixed-concurrency runs distinguish queueing changes from kernel
   slowdowns. Keep unique prompts, seeds, layout, steps, LoRA and flash settings
   fixed. Use numeric server-phase logs where available to separate text,
   diffusion and VAE time.

The two-second sampling interval can miss short events. Correlation is not a
cause, and no GPU speed improvement is claimed for this instrumentation.

## Verify on CPU

```sh
"$PY" -m unittest discover -s data-pipeline/benchmarks -p 'test_krea*.py' -v
```

Synthetic fixtures check aligned request fields, privacy allowlists, streak
classification, concurrency grouping, error counts, optional-reason fallback,
and recording only new runner events. They do not measure GPU performance.

NVIDIA documents [clock-event reasons and their limits](https://docs.nvidia.com/deploy/nvidia-smi/index.html#clocks-event-reasons)
and warns that `nvidia-smi` output may vary across releases. Unsupported values
are kept unknown rather than interpreted as inactive.
