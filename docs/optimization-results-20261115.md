# Optimization results and next experiments

The operator supplied these results on 2026-11-15. They ran PRs #7 through #10
on the 12 GB RTX 3080 Ti with compile, CE chunks 16 and five trials per mode.
No local GPU measurements were made by the code author. Raw trial ranges and
limiter profiles were not supplied, so the figures below are reported results
rather than independently reproduced measurements. They do not establish
3060 Ti or 5070 Ti performance.

## Accepted cache and generation changes

| Experiment | Supplied result | Decision |
|---|---|---|
| P32 CLIP RAM cache, PR #10 | 335 to 347 images/s, about +3.5%; loss identical and VRAM unchanged | Retain the opt-in cache. |
| Cache construction | 86,906 rows, 2.43 GB, 52 seconds | Reuse this verified cache for the proposed branch. |
| Krea concurrency, PR #7 | 62/62 sheets succeeded; in-flight 1 took 4.22 s/sheet and 2 took 3.30 s/sheet, about +28% wall throughput | Keep the benchmark and test settings independently per GPU. |

Krea's benchmark declared flash attention off, but the server reported auto
resolving to on. Those results do **not** measure flash-off performance.
The original client flag only recorded an operator declaration. A follow-up
must pass the setting to a benchmark-owned server at startup or label it as
unverified when using an existing server.

## Rejected changes

| Experiment | Baseline | Proposed change | Decision |
|---|---:|---:|---|
| Head checkpointing, batch 64, PR #9 | 334 images/s, 9.22 GB | 304 images/s, 9.72 GB | Drop head checkpointing. |
| Head checkpointing, batch 80 | 345 images/s, 11.18 GB | 317 images/s, 11.90 GB | Batch 80 fits without it. |
| Head checkpointing, batch 96 | OOM | OOM | Do not use batch 96. |
| Chat encoding, PR #8 | 15.2 ms | 7.6 ms | Drop single-row CLIP encoding because parity failed. |
| Chat benchmark encode plus sampling | 287.5 ms | 280 ms | About 2.6% lower time did not justify changing outputs. |

Head loss parity passed, but speed and VRAM were worse. The memory figures are
as supplied in GB. PR #8 had maximum absolute feature error 1.0 and only
85/960 exact sprite matches, about 8.9%. Its benchmark excludes reward scoring,
PNG work and network latency, so its timings do not establish full worker
latency. BF16 kernels at batch one versus batch 64 are a suspected cause of
the output mismatch, not a demonstrated root cause. Keep the original batched
worker encoding path.

## Next training branch

The proposed configuration uses the paused **full P32 checkpoint at step
10,602**, the verified RAM cache, batch 80, compile, CE chunks 16, bf16 and
no block or head checkpointing. The supplied head benchmark measured batch 80
about 3.3% faster than batch 64 without head checkpointing. The operator's
approximately +8% combined target is an **estimate**, not a measurement of
batch 80 plus the cache together. Re-run that combination in a new folder.

Use an explicit P32 branch operation. Carry the saved weights, AdamW state,
global update count and saved `images_seen`. Never derive old views as
`step * 80`. Start a new batch plan and record the source digest and reset
sampler position. Keep exact resume strict about batch size and cache identity.

Keep the existing LR of `1e-4` for the first comparison. This is a conservative
experimental recommendation, not a measured optimal LR for batch 80. Carry
the completed warmup and keep decay tied to the original image budget.
Increasing batch size by 25% means 20% fewer updates over the same remaining
views. AdamW moments and per-update weight decay therefore have different
effects per image even with the same LR. A speed improvement alone does not
prove equal training quality. Compare IID/probe NLL at matched image counts.

## Remaining bottlenecks

1. Measure wall time before rewriting the loader. The reported training rate
   excludes evaluation, sample grids and checkpoint writes. Record those phases
   and segment wall images/s separately.
2. Remove duplicate final evaluation and rendering only when they already ran
   at the same model step. The current trainer repeats them at exact periodic
   boundaries. Expected total savings are unmeasured.
3. Profile batch preparation, RAM feature lookup and host-to-device transfer
   on the real machine. The input arrays fit in RAM; no measured loader limiter
   has been established.
4. Sampling recomputes the growing trunk prefix for each of 256 patches.
   A KV cache may reduce this work, but first measure its share of wall time.
   Any implementation must include GPU numerical/output checks after the chat
   parity failure. Do not replace it based on CPU parity alone.

Run comparisons on an idle explicitly selected GPU, in new output folders,
with the same checkpoint and input cache. Keep both frozen holdouts excluded.
Report all completed work and errors, five alternating trials, medians/ranges,
peak VRAM and evaluation quality. A gap below the trial spread is inconclusive.
