# Compare the same offline worker on two GPUs

There is no current 3060 Ti worker profile. The old approximately 1.6-second
setup figure has unknown timing boundaries: it is not established as worker-only
or browser latency. Do not compare it with the measured 628 ms offline worker
result on the 3080 Ti or infer a network gap from the difference.

Run this same checkout, local asset set, configuration and command once on each
idle GPU. Keep caches, graph mode, PNG mode, precision, scorer and candidate
budget identical. The command makes no app requests, needs no worker key and
does not touch any running worker or training.

```bash
PIXEL_CACHE_BLANK=1 PIXEL_REWARD_TEXT_CACHE=0 PIXEL_PNG_MODE=legacy \
  "$PY" worker/benchmark_worker.py --config "$WORKER_CONFIG" --device cuda:0 \
  --batch 64 --show 4 --candidates 20 --warmup 5 --trials 5 \
  --out "$RUNS/worker-3080-new"

# On the 3060 Ti setup, with the same assets/configuration and checkout:
PIXEL_CACHE_BLANK=1 PIXEL_REWARD_TEXT_CACHE=0 PIXEL_PNG_MODE=legacy \
  "$PY" worker/benchmark_worker.py --config "$WORKER_CONFIG" --device cuda:0 \
  --batch 64 --show 4 --candidates 20 --warmup 5 --trials 5 \
  --out "$RUNS/worker-3060-new"

"$PY" worker/compare_latency.py \
  --reference "$RUNS/worker-3080-new/worker-benchmark.json" \
  --candidate "$RUNS/worker-3060-new/worker-benchmark.json" \
  --out "$RUNS/worker-gpu-comparison-new"
```

`cuda:0` selects an explicit physical GPU on each setup; retarget it if needed.
The benchmark refuses another compute process on the selected GPU. Transfer
the reports yourself if needed; the comparator reads only the supplied files.
All outputs require new folders.

New benchmark reports bind the workload with a versioned identity containing:

- Ordered prompt hashes, seed schedule, trials/warmup, batch and display/storage budgets.
- Checkpoint, CLIP, tokenizer and palette hashes; sampler math and cache settings.
- Exact loaded reward model tensors, recipe/PCA/head tensors and reward token hashes.
- PNG mode/scale, actual render size, loaded model/scorer/CPU-feature code and benchmark source hashes.
- Python, Torch/CUDA/cuDNN, NumPy/Pillow and relevant package versions.

GPU model names are recorded separately, without hosts, GPU UUIDs or paths.
Prompts and local asset filenames are not copied into the report. Hashes are
identifiers, not encryption of low-entropy prompts. Reward model hashing reads
each loaded tensor to CPU once outside all timed calls. It may add noticeable
startup time and CPU memory up to the largest tensor; this is not generation
latency and is not included in reported throughput.
Only Python source hashes normalize CRLF to LF so matching Git checkouts can
compare across operating systems. Binary model/assets and frozen holdout bytes
are never normalized. Imported model/scorer sources are hashed from their
actual loaded modules, including configured code overrides. Importing the GPU
idle-check helper no longer preloads the repository's model before the worker
can apply its explicit code configuration.

The comparator requires matching workload/runtime identities, completed five
paired trials per prompt and local exact-output parity. It verifies every
summary against raw trial medians/min/max and checks exclusive phase accounting.
Missing old provenance, changed settings, stale summaries, incomplete trials
or nonfinite timing produce an **inconclusive** result with no latency ratio.
This conservative policy deliberately refuses different software versions;
those would compare whole environments rather than isolate hardware.

Read `latency-comparison.json`. The clean total is the complete offline worker
latency on each GPU. Each profiled phase gets a separate candidate/reference
median ratio and range overlap. Profiling synchronizes CUDA and can perturb
timing; do not sum phase medians or treat those ratios as causal proof. Output
equality is tested between clean/profiled calls on each GPU, not promised across
different GPUs. The comparison excludes queue, polling, JSON serialization and
HTTP; it cannot identify browser latency.

Only after obtaining the 3060 Ti offline baseline should transport observations
be compared with it. Existing `PIXEL_PROFILE_PHASES=1` worker events measure job
receipt to result acknowledgement and post/retry duration. Long-poll wait is not
user queue latency. Actual browser/request timestamps would be needed to locate
queue or client overhead; this tool does not invent them.

```bash
"$PY" -m unittest discover -s worker -p 'test_latency_compare.py'
```

CPU fixtures test real hash changes, matching/mismatched identities, nonfinite
or stale evidence, exclusive accounting, actual CLI output and overwrite refusal.
No GPU measurements or 3060 Ti slowdown diagnosis were made in this workspace.
