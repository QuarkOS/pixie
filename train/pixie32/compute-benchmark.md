# Measure a P32 compute change

Run one comparison at a time on an idle GPU. This tool reads a paused full P32
checkpoint and verified data/CLIP caches. It never saves trained weights or edits
a run. Choose a new output directory for every command.

```bash
$PY -m train.pixie32.bench_compute \
  --data "$DATA_ROOT/p32-cache" --clip-cache "$DATA_ROOT/p32-clip-cache" \
  --checkpoint "$RUNS/paused/last.pt" \
  --clip "$CLIP_DIR/clip_l14_text_l11_f32.pt" \
  --blind50 holdout/blind-50.json --eval50 holdout/eval-50.json \
  --device cuda:0 --batch 80 --ce-chunks 16 \
  --steps 30 --warmup 15 --trials 5 \
  --experiment fused-adamw --out "$RUNS/bench-p32-fused-NEW"
```

Repeat with `sdpa-flash`, `sdpa-efficient`, `compile-max-autotune-no-cudagraphs`,
`compile-reduce-overhead`, `compile-max-autotune`, or `tf32`. Each command compares
one setting with the same baseline. `sdpa-math` is available as a diagnostic and
may exceed memory at batch 80. No setting is enabled in the trainer by this PR.

To run the six requested variants separately at batch 80 with the CLIP cache
and five alternating trials per side:

```bash
"$PY" -m train.pixie32.bench_compute_suite --out "$RUNS/compute-suite-NEW" -- \
  --data "$DATA_ROOT/p32-cache" --clip-cache "$DATA_ROOT/p32-clip-cache" \
  --checkpoint "$RUNS/paused/last.pt" \
  --clip "$CLIP_DIR/clip_l14_text_l11_f32.pt" \
  --blind50 holdout/blind-50.json --eval50 holdout/eval-50.json \
  --device cuda:0 --ce-chunks 16 --steps 30 --warmup 15
```

The suite order is fused AdamW, SDPA flash, SDPA efficient, TF32, compile
reduce-overhead, compile max-autotune. It continues after a failed experiment
so unsupported/OOM configurations do not hide the remaining results; `suite.json`
and the command's nonzero exit preserve those failures. Results are independent
comparisons, not a tested combination. The suite uses the same bounded hardware
benchmark; no local GPU or real checkpoint was available to run it here.

Read `benchmark.json`. Adopt a result only when `comparison_valid` is true, all
five trials completed on each side, loss/gradient/update parity passed, the gain
exceeds trial variability, and reserved memory fits. A failed parity gate stops
before timing. A positive median timing result also triggers the short trajectory
check below before adoption. An OOM or unsupported kernel invalidates the comparison and returns
exit code 1. The result includes the error class; kernel warnings remain in the
local console. Do not interpret failed or partial trials as a speedup.

The baseline uses BF16 autocast, automatic causal SDPA, `torch.compile` default
mode, non-fused AdamW, CE chunks 16 and no block checkpointing. These match the
reported compute setup. CLIP stays on the selected GPU even though features come
from RAM, so its memory is included. TF32 is explicitly off in the baseline.
Each candidate changes exactly one field. Confirm that this baseline matches the
machine's real run before comparing its throughput with a previous benchmark.

Each timed side runs in a fresh process. This prevents retained CUDA graph pools
and allocator state from favoring the next side. Trial order alternates A/B and
B/A. Both sides restore the same checkpoint weights and AdamW moments, draw the
same prepared batches and reset RNGs. Timing includes augmentation, cache lookup
and transfer, forward/loss, backward, gradient clipping and completed AdamW steps.
It synchronizes after every update, as the trainer does. Startup, asset loading,
warmup compilation, evaluation, sampling and checkpoint writes are excluded.
The report keeps the throughput range, losses, warmup time including compilation,
and peak allocated/reserved VRAM
for every trial. It records asset hashes and software versions without paths.

The separate parity gate uses the full requested batch, disables dropout on both
sides, and compares loss, NLL, every gradient tensor, and parameters after one
clipped AdamW update from saved moments. Loss/NLL and gradient gates retain
their existing elementwise and relative tolerances. Updated parameters now gate
on **finite values and global relative L2 error ≤ `1e-5`**. The L2 numerator and
reference norm aggregate all named parameter tensors in FP64. Max absolute error
is diagnostic only; an individual small/zero bias tensor does not introduce a
separate denominator or absolute rejection threshold.

The operator's **measured, supplied** one-step absolute errors were `7.8e-6`
to `3.1e-5`, with global relative L2 errors `3e-7` to `2.2e-6`; loss and gradients
passed, and reported holdout NLL agreed or differed by at most `1e-4`. The old
elementwise parameter `atol=2e-6` gate blocked timing despite that evidence. This
change accepts that scale of rounding while rejecting nonfinite or materially
larger relative drift. No new GPU agreement is claimed by the CPU tests.

Timed training retains dropout. Different compiled
dropout implementations can consume RNG differently, so timed loss sequences
are recorded for inspection and are not asserted bit-identical. This gate checks
local numeric agreement; it does not replace a later quality evaluation.

## Validate a timing winner over 200 updates

Any valid comparison with a positive candidate median throughput change runs
two additional fresh processes, each starting from the same checkpoint, AdamW
moments and RNG seed. These perform **200** matched updates by default, retaining
the original training dropout and augmentation plan. They do not write model
weights. These trajectories are separate from the timing trials; a positive
median alone still does not establish a statistically distinguishable speedup.

`multi_step_parity` reports paired 20-update loss-block means. Its default
tolerance is `1e-3 + 1e-3 * abs(baseline mean) + 3 * paired standard error`.
The standard error uses differences between corresponding candidate/baseline
training losses. This is an explicit **heuristic noise allowance**, not a
formal independence assumption or proof of long-run quality. Inspect the raw
`curve-0.json` and `curve-1.json` loss traces and the reported block errors; do
not widen tolerances just to make a candidate pass. Flags `--curve-atol`,
`--curve-rtol` and `--curve-noise-sigma` expose these assumptions in the report.

At update 0 and every 20 updates (plus the last partial block), both trajectories
evaluate deterministic conditioned NLL on the first batch of the IID split and
the first batch of the probe split. These rows never participate in gradients.
The NLL gate uses absolute `1e-4` plus relative `1e-4` tolerance per split/check.
This bounded sample is not a full held-out evaluation; Blind-50 and Eval-50
remain excluded by the existing dataset/byte-manifest checks. A later quality
evaluation remains necessary before a long run.

`adoption_gate_passed` requires valid five-trial timing, the entire candidate
throughput range above the baseline range, and a passing multi-step check.
The automatic curve trigger after a positive median treats it only as a
provisional speed candidate; overlapping ranges remain inconclusive. If the multi-step check fails, the timing
numbers remain measured but the command exits 1 and says not to adopt. A
missing/partial/nonfinite curve or changed asset identity fails validation.
`--validate-steps` can select 1..200 updates for debugging; use the default 200
for an adoption decision. The learning rate remains the checkpoint's saved LR
on both sides, isolating numerical drift rather than reinterpreting the WSD
schedule for an experimental continuation.

To validate a candidate independently without another timing run, repeat the
single-experiment command with `--validate-only --validate-steps 200` and a new
output folder. That mode sets `validation_valid`; it supplies no timing claim.

## Priority and limits

The operator measured a median update of 221 ms at batch 80. The operator's CUDA
phase measurements were 141.8 ms backward, 66.6 ms forward/loss, 9.0 ms AdamW,
1.7 ms data and 1.4 ms cache transfer. Reserved memory was measured at 11.66 GB.
The phase total is not assumed to equal the median wall update exactly.

1. Test fused AdamW first. It changes no model kernel and has a narrow parity
   test. Its gain is untested. Removing the entire measured 9 ms phase would cap
   the estimated throughput gain at `221 / (221 - 9) - 1`, about 4.2%.
2. Test attention backend selection. The model already calls SDPA, so merely
   enabling SDPA is not a new optimization. The gain is untested and depends on
   which kernel automatic dispatch already selects. Forced backends have no
   fallback. A backend that cannot run the real training dropout fails clearly.
3. Test compile modes, starting with max-autotune without CUDA graphs. The gain
   is untested. Graph-enabled modes can retain additional memory, which matters
   at the measured 11.66 GB reserved. Warmup cost is excluded from steady-state
   throughput and still matters when deciding whether to adopt a mode.
4. Test TF32 last. BF16 autocast already handles most matrix multiplications.
   The gain is untested; this is a check for residual FP32 work, not a BF16 switch.

An estimated scenario illustrates the compute opportunity. Cutting the combined
measured forward/backward time by 10% while leaving other costs fixed would save
20.84 ms and improve throughput by about 10.4%. This is arithmetic, not a measured
gain or a prediction for any candidate. Gains from separate experiments cannot
be added; benchmark any accepted combination again.

The alpha loss already needs the vocabulary logsumexp. Replacing NLL alone with
`cross_entropy` can duplicate that reduction. A fused NLL/alpha kernel needs its
own gradient comparison and memory evidence before it earns a training flag.

PyTorch documents [SDPA backend selection](https://docs.pytorch.org/docs/2.9/generated/torch.nn.attention.sdpa_kernel.html),
[compile modes and CUDA graph memory](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html),
and [fused AdamW](https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html).

## Local verification

```bash
$PY -m unittest train.pixie32.test_bench_compute -v
```

CPU tests exercise a real small P32, compare fused/non-fused AdamW loss,
gradients and updated parameters, check saved-moment isolation, count completed
updates, and verify A/B ordering and failure handling. These tests provide no
RTX 3080 Ti speed or VRAM result. The real checkpoint and GPU are supplied only
on the training machine.
