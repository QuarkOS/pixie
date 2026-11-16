# Pixie stack investigation

The goal is useful throughput and better sprites on the existing GPUs. Keep
the accepted b80/cache P32 branch, the existing batched chat numerics and locked
extraction gates. The next changes target seed provenance, model compute,
long-run reporting, data-quality measurement and complete worker latency.

The latest follow-up uses merged main at `8e622cc` and operator feedback after
PRs #20–#24. Earlier sections retain the historical #14–#19 investigation at
`6041ee5`; the original code review used `25ed637`. This work does not launch training,
contact a production worker or use private checkpoint paths. The operator runs
the hardware experiments. The [workflow](pixie-investigation-plan.md) and
[decision trail](pixie-investigation-decisions.tsv) describe the work sequence.

## Evidence and current baselines

**Measured, supplied** means a result reported by the operator or the earlier
[stack handoff](training-stack.md). **Calculated** means arithmetic from those
results or from the checked-in model shapes. **Estimated** means a projection,
recommendation or untested effect. Configuration values describe code, not
measurements. No new GPU performance was measured in this workspace.

The latest operator results supersede earlier smoke figures. Unless stated
otherwise, the new hardware comparisons used the 3080 Ti, compile and five
trials. Raw trial ranges and traces were not supplied, so they cannot establish
the limiter or significance of a small difference independently.

| Workload | Baseline and result | Evidence status |
|---|---|---|
| P32 cached vs online CLIP | 345.5 vs 334.2 images/s, about +3.4%; same loss and VRAM; cache 2.43 GB | Measured, supplied. Supersedes earlier rounded 347/335. |
| P32 b80/cache continuation | 200 updates to step 10,802 and 694,528 views; loss about 0.52; 11.62 GB reserved | Measured, supplied smoke. About 357 images/s is the operator's steady-rate estimate, not a five-trial result. |
| P32 head recomputation | About 9% slower and more VRAM; b80 fits without it | Measured, supplied rejection of #9. Keep it absent. |
| Short P32 segments | Eval, sampling and checkpoint writes consumed about 73% of 100-step segment wall time | Measured, supplied. This is not the reporting share of a long run. |
| Final reporting deduplication | Saves about 50 seconds once in a real run | Measured, supplied approximation. Main value of #13 is the phase clocks. |
| Krea flash, 3080 Ti | On 3.27 vs off 3.71 seconds/sheet, about +13.5% sheets/s | Measured, supplied after startup forwarding was fixed. |
| Krea concurrency, 3080 Ti | In-flight two 3.30 vs one 4.22 seconds/sheet, about +28% sheets/s; 62/62 successes | Measured, supplied earlier comparison. Do not multiply this gain by the separate flash result. |
| Chat CLIP batch-one reuse | Feature max error 1.0; 85/960 exact sprites, about 8.9%; encode 15.2 to 7.6 ms | Measured, supplied rejection of #8. |
| Same rejected chat benchmark | 287.5 to 280 ms, about 2.6% lower time | Measured, supplied encode-plus-sampling scope. Excludes reward, PNG and network. |
| Historical 3060 Ti setup figure | About 1.6 seconds, with unknown timing boundaries | Operator-supplied approximate figure. The operator clarified that no 3060 Ti benchmark/profile exists; it is not established as worker-only or browser latency. |
| 5070 Ti slow streaks | About 40% of sheets ran at roughly half speed | Measured, supplied approximate observation. Cause and grouping by concurrency are unknown. |

Latest **measured, supplied** results supersede the instrumentation proposals
below: b80/cache on the 3080 Ti had median update time 0.221 s across 50
profiled updates. CUDA intervals were backward 141.8 ms, forward/loss 66.6 ms,
AdamW 9.0 ms, data 1.7 ms and cached CLIP 1.4 ms. Reserved VRAM was 11.66 GB.
CUDA intervals and the CPU wall median are different statistics; do not add
them or treat their ratio as a precisely measured phase share. They identify
model compute as the first target. Data-loader work has little demonstrated
headroom. Attention already calls SDPA, and the trainer already uses BF16.

Exact full-batch blank CLIP caching passed 2,880/2,880 sprite matches with
feature error 0.0. Encode time fell 15.5 to 8.3 ms; encode plus sampling fell
293.1 to 285.6 ms (about 2.5% lower latency). These are **measured, supplied**
3080 Ti results, not full-worker or 3060 Ti results. The earlier rejected
batch-one reuse remains rejected. Restart retries passed two back-to-back
benches without errors. Holdout preflight passed with the original bytes and
rejected reformatted copies. New telemetry works; the older 5070 Ti run has
no GPU telemetry, so its slow-streak cause remains unknown. Record the next
run rather than infer GPU behavior from the old aggregate events.

### Latest feedback after #20–#24 (merged)

All five PRs merged at `8e622cc`. The following **measured, supplied** 3080 Ti
offline-worker results supersede speculation about the complete generation cost:

| Metric | Operator result | Scope |
|---|---|---|
| Complete offline generation | About 628 ms | Generation, reward and PNG; excludes queue/HTTP/browser timing. |
| Prepare / CLIP / sampler phase medians | 0.17 / 8.68 / 279.6 ms | Instrumented phases; do not sum medians or treat them as the clean total. |
| Readback/RGBA / reward / PNG phase medians | 1.67 / 258.6 / 77.8 ms | Instrumented phases on the 3080 Ti. |
| Compute parity experiments | Loss/gradients passed; holdout NLL identical or within `1e-4`; only updated-parameter absolute tolerance failed | Absolute differences `7.8e-6`–`3.1e-5`, relative L2 `3e-7`–`2.2e-6`, supplied. No timed winner established. |
| Human review of 400 R1 sprites | Approximately 87% good / 10% borderline / 3% junk | Supplied visual review. Rounded shares, not an exact label export or calibrated caption-mismatch rate. |

There is **no measured 3060 Ti worker profile**. The old roughly 1.6-second
figure has unknown timing boundaries and cannot establish a hardware or network
gap against 628 ms. [Matched offline comparison](compare-worker-gpus.md) first
requires the same prompts, settings, loaded weights/code and software on both
cards. Its identity checks refuse old or mismatched reports rather than explain
a gap from missing data.

Current follow-ups are separate, open PRs; none changes the running recipe:

| PR | Change and acceptance | Runbook and measured CPU verification |
|---|---|---|
| #25 | Reviewable simple-tile geometry signals and optional new weights; no automatic claim that a rectangle is junk. | `docs/r1-quality-weights.md`; 21 tests passed. |
| #26 | Global relative-L2 updated-parameter gate plus loss/NLL/gradient parity; separate bounded multi-step validation and six isolated five-trial compute comparisons. GPU timings remain pending. | `train/pixie32/compute-benchmark.md`; 24 tests passed. |
| #27 | Default-off exact reward-text cache and complete-worker cold/warm benchmark. Benefits repeated prompts only; new-prompt gains remain unknown. | `docs/reward-text-cache.md`; 30 tests passed. |
| #28 | Plan from supplied steady `0.221 s/update`; measure completed checkpoint writes separately. No target-disk measurement is inferred from CPU fixtures. | `docs/pixie32-long-run-cadence.md`; 10 tests passed. |
| #29 | Caption-mismatch review against explicit local semantic-model assets; scores require calibration and are not automatic semantic truth. | `docs/caption-mismatch-check.md`; 4 tests passed. |
| #30 | Opt-in lossless PNG compression experiment; exact pixels and full-worker parity required, payload growth reported. | `docs/worker-png-speed.md`; 25 tests passed. |

Test counts overlap across branches and must not be summed as unique tests.

The PNG experiment's final level-3 candidate **measured locally** 45.39 to
12.64 ms median for 20 sample PNGs over five alternating CPU pairs, with ranges
44.40–48.00 and 12.41–14.02 ms. Signature payload grew 40,401 to 49,369 bytes
(8,968 bytes, about 22.2%, calculated). Other work was active (37% CPU
load observed before the run). These are sample encoder results, not GPU or
browser gains. Full-worker and transport measurements must decide adoption.

The updated comparison tool and its [runbook](compare-worker-gpus.md) bind all
settings/assets before publishing clean-total or separate phase ratios. CPU
behavior tests provide no GPU speed, cross-GPU sprite parity or latency cause.
Its 29 worker CPU tests passed, including nine new identity/comparison checks.

### Historical follow-up order after #14–#19 (now merged)

This order supersedes the original priority table below. Benefits are
**estimates / untested** unless explicitly marked measured.

| Order | Change | Expected benefit | Cost / risk | Acceptance |
|---|---|---|---|---|
| 1 | Honor job seeds and persist honest batch/candidate provenance | Reproducible quality comparisons; no speed gain claimed | Small/medium; exported seed meaning must be versioned | Same whole-batch replay settings and original candidate indices survive ranking/export. Preserve existing RNG; do not claim independent candidate or chunk/reorder replay. |
| 2a | Compare fused AdamW alone | Unknown. Eliminating all 9 ms would cap update throughput improvement at roughly 4.2%, calculated from 0.221/(0.221-0.009)-1, using mismatched timing statistics only as a rough ceiling | Small; optimizer numerical drift | Five alternating b80 trials, completed updates, loss/gradient/update parity, peak memory. |
| 2b | Compare SDPA backend and compile modes separately | Unknown; forward/backward dominate, but neither speedup nor fit is established | Medium; BF16 parity, graph pools/autotune near the VRAM limit | Same parity gate and fixed real cached batches; unsupported/OOM configurations are failed experiments. Never stack knobs before attributing each. |
| 2c | TF32 comparison alone; defer a custom CE kernel until profiler evidence warrants it | Unknown; BF16 limits TF32's scope, and manual CE also supplies unweighted NLL | Medium; quality drift and implementation cost | Gate on numerical results, not the option's name. Keep BF16 baseline and training defaults. |
| 3 | Full eval every 250k views; sample grid every 500k; recovery saves stay at 100k initially | Estimated 60% fewer periodic evals and 80% fewer periodic grids versus 100k cadence; overall speed gain needs measured phase durations | Small; slower quality feedback. Checkpoint disk growth persists | Use [cadence analysis](pixie32-long-run-cadence.md), separate segment measurements from projections, branch to a fresh run only when the operator chooses. |
| 4 | Audit the complete R1 population and a uniform human-labelled sample | Unknown useful-yield gain; determines what to gather or reject next | Medium; missing labels and selection bias | Report actual coverage, completed-label denominator, uncertainty, duplicate definition and weighting scope. Do not call metadata heuristics a measured junk rate. |
| 5 | Instrument complete worker generation, scoring, PNG and transport phases | Historical proposal; full 3080 Ti profile now supplied. The old 3060 Ti figure has unknown boundaries | Small/medium; instrumentation synchronizes phases | Obtain a matched 3060 Ti offline profile before comparing hardware or transport. Preserve images, scores and candidate budget. |

Keep the accepted b80 LR/WSD recipe while comparing compute changes. Twelve
million views remains a budget rather than a measured quality optimum. Use
the same held-out quality checks and matched replayable candidates to decide
whether to extend or shorten a later run. Never optimize those decisions on
Blind-50 or Eval-50 training exposure.

### Historical #20–#24 delivery (now merged)

These PRs are now merged. Hardware commands are included with each change;
reported CPU test counts overlap and must not be summed as unique tests.

| PR | Runbook | Local verification, measured |
|---|---|---|
| #20 seed provenance | `docs/worker-seed-provenance.md`; full-batch GPU replay gate | 11 Python CPU tests and Node queue/archive checks passed. Existing Windows HTTP integration test could not start because of an ESM import error. |
| #21 isolated compute experiments | `train/pixie32/compute-benchmark.md`; start fused AdamW, then forced SDPA and compile modes separately | 17 CPU tests passed; independent review passed. GPU gains/parity/VRAM remain untested here. |
| #22 cadence analysis | [Long-run cadence](pixie32-long-run-cadence.md); read a completed segment | 5 CPU tests passed; independent review corrected final partial-batch event counting. No long-run performance measurement. |
| #23 R1 quality audit | `docs/r1-quality-audit.md`; scan 133,688 rows, uniformly sample 400 and summarize human labels | 7 CPU tests passed. Sample200 had 198 subjects, 200 normalized captions and 28 edge-heavy rows; zero exact visible-pixel duplicate excess or empty/nonbinary-alpha images. Full-corpus junk rate unmeasured. |
| #24 complete worker timing | `docs/worker-latency-profile.md`; offline full-generation pairs plus opt-in transport observations | 20 CPU tests passed. Depends on #20 and targets its branch. Real 3060 Ti timings and exact GPU parity remain unmeasured here. |

No running training, dataset inputs or frozen holdout bytes were changed. The
old 5070 Ti aggregate events are retained as incomplete evidence; use the new
telemetry on the next run rather than infer a GPU cause from missing counters.

## Original ranked improvements (before #14–#19 measurements)

Effort and benefit entries are estimates. No numeric speed forecast is offered
where a phase share or hardware trace is missing. Instrumentation gains are
diagnostic; they do not themselves make kernels faster.

| Priority | Change or experiment | Expected gain, estimated unless labeled | Effort | Risk and acceptance gate |
|---|---|---|---|---|
| P0 | Preserve frozen holdout bytes and preflight the original files | Avoids invalid resumes and accidental policy drift; no speed claim | Small | Strict raw hashes remain mandatory. Formatting-only changes must fail. |
| P0 | Bound server port-release retries | Reliability on immediate restart; no throughput claim | Small | Retry address-in-use only, keep one deadline, never stop another process. |
| P1 | Correct experiment seed provenance and honor requested comparison budgets | Reproducible quality comparisons; no speed claim | Medium | Current seeds do not reproduce individual sprites. Version any changed RNG scheme and test batch reorder/chunk invariance. |
| P1 | Attribute real P32 update phases and long-run reporting costs | Finds the limiting phase; no direct gain | Small | Profiling perturbs execution. Do not compare instrumented throughput with a clean speed run. |
| P1 | Correlate Krea slow streaks with persistent telemetry | If 40% of equal-cost sheets truly take twice as long and all recover to the fast rate, the arithmetic ceiling is 40% more sheets/s | Small/medium | This is a conditional upper bound, not a forecast. Verify work, cache state, concurrency and usable yield. |
| P1 | Reduce expensive grid/full-eval frequency after measuring long-run share | If a phase takes fraction f of wall time and its frequency is halved, ideal speedup is 1/(1-f/2) | Small | A formula, not a measured gain. Keep timely quality checks and recovery checkpoints. |
| P1 | Measure junk and coverage by source/subject/edge bucket before more bulk generation | Higher useful sprites per GPU-hour; magnitude unknown | Medium | Blind annotation, random audit sample and false-reject checks; do not tune on Blind-50. |
| P2 | Cache the complete blank CLIP batch at the original batch shape | Saves one blank encode on cache hits; numeric latency gain unknown | Small | Default off. Exact feature and sprite equality on the GPU is required. |
| P2 | Separate permanent model snapshots from full recovery cadence | Lower disk growth and write pauses; magnitude unknown | Small | Keep full optimizer/RNG recovery often enough for the operator's recovery budget. |
| P2 | Test compiler modes and fused AdamW at the accepted recipe | Unknown; may be negative or OOM | Medium | Same weights/data/precision, completed updates, five alternating trials and memory peaks. |
| P3 | Pinned-memory prefetch or precomputed row indices | At most the measured transfer/lookup bottleneck share | Medium | Need a demonstrated loader limiter, 16 GB RAM budget and exact augmentation/resume checks. |
| P3 | P32 sampling KV cache | Potentially reduces repeated-prefix grid work; magnitude unknown | Large | Requires GPU feature/output parity and a profile showing grids matter over a long run. |

## Training on the 3080 Ti

### Separate update throughput from time to finish

[`train_p32.py`](../train/pixie32/train_p32.py) already measures whole training
updates and separate evaluation, sampling and checkpoint phases. Training
images/s excludes the latter phases. Segment wall images/s includes them and
first-step compilation, but excludes asset loading. Short smoke finalization
runs a full evaluation/grid regardless of whether the regular interval occurred.
This explains why its overhead share cannot be extrapolated to a long run.

Add bounded update profiling to distinguish:

1. Array indexing/conversion, augmentation, host-to-device transfers and LR setup.
2. Online CLIP or RAM feature lookup/transfer.
3. Trunk, causal head and CE forward/loss.
4. Backward, including CE recomputation.
5. Gradient clipping and AdamW.

Use CUDA-event intervals plus host dispatch/blocking time. Neither is a pure
kernel profile: GPU intervals can include dispatch gaps, and host time overlaps
device work. Never add the two totals. Use a separate profiler trace if a
particular kernel is suspected. Report warmup/recompile counts and successful
updates before interpreting throughput.

### Keep b80 and the current LR for the first controlled comparison

The accepted branch carries model weights, AdamW moments, RNG, actual saved
image count and WSD progress. It resets a deterministic batch plan explicitly.
An exact resume still rejects changes to batch/cache identity. The supplied
post-smoke counts are consistent with the saved parent views plus the new
updates; they must never be reconstructed as global step times the new batch.

Keep LR `1e-4`, warmup 1,500 updates and decay over the final 2M of the total
12M views for the current comparison. These are configuration choices, not a
measured optimal b80 recipe. b80 makes 20% fewer updates per remaining view
than b64, calculated from the batch ratio. AdamW moment horizons and per-update
weight decay therefore change per image. Do not automatically raise LR by 25%.
Research on Adam-style optimizers does not support a universal linear scaling
rule; this does not establish Pixie's optimum either. See the
[NeurIPS study](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ef74413c7bf1d915c3e45c72e19a5d32-Abstract-Conference.html).

Measured reserved memory of 11.62 GB leaves little practical room for autotune,
extra graph pools, evaluation batches or desktop load. It is not a guarantee
that another compiler mode fits. b96 already OOMed in the earlier comparisons.
Keep one heavy workload on this GPU and measure peak memory across evaluation
and rendering, not only the clean update loop.

### Twelve million views is a budget, not a demonstrated optimum

Calculated from the documented R1 corpus, 12M/133,688 is about 89.8 whole-R1
view-equivalents. The training split is smaller, and weighted replacement draws
mean these are not epochs. The current 694,528 views are about 5.2 whole-R1
equivalents, calculated. No supplied quality curve establishes convergence or
the best stopping point.

For current run planning use the supplied steady **0.221 seconds/update** at b80.
The full 12M budget is 150,000 updates, about **9.21 hours**, calculated estimates
excluding reporting, startup and recompiles. The earlier 357 images/s smoke
estimate is superseded for planning. The 694,528-view checkpoint count above is
historical; supply the actual current count to the read-only cadence tool in #28.

Keep matched-view IID/probe NLL curves plus a frozen non-holdout development
generation panel. Measure subject correctness, clean outlines/transparency,
palette/style compliance, diversity and usability. Use Blind-50/Eval-50 for
frozen evaluation only, never training or prompt generation. Do not repeatedly
select hyperparameters on the blind set. Later, compare explicit anneal branches
from selected full checkpoints using matched remaining budgets. The current
branch API intentionally forbids schedule changes; a decay experiment needs a
separate reviewed mode. Do not silently alter the running recipe or extend it
because a budget remains.

### Compiler, loader and memory

The accepted run already uses bf16, compile and CE chunks 16. Inspect graph
break/recompile counts before replacing kernels. Fixed batch and sequence shapes
help reuse; a final partial batch can compile separately. Test default,
reduce-overhead and max-autotune in isolation. PyTorch documents extra graph-pool
memory for reduced-overhead modes and more tuning work for autotune, so include
compile time and peak reserved memory. See
[PyTorch 2.9 compile modes](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html).
Fused AdamW is another isolated experiment, not an assumed upgrade. Compare
completed optimizer updates and loss/gradient checks; see
[AdamW implementation options](https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html).

The arrays are memory-mapped; the supplied handoff estimates their footprint
around 0.3 GB. The accepted CLIP cache preloads 2.43 GB, measured by the operator.
There is no established HDD-loader bottleneck. Batch preparation converts
selected NumPy values to int64, copies masks/conditions and flips images.
[`FeatureCache`](../train/pixie32/clip_cache.py) maps token rows through a Python
dictionary, gathers FP16 host features and copies them to the GPU. Profiling
should determine whether CPU lookup, transfers or compute limits the update.
Only then test row-index precomputation or bounded double-buffered pinned copies.
Count process RSS, compiler/checkpoint transients and OS headroom within 16 GB;
do not pin the whole cache or multiply copies with many loader workers.

### Evaluation, sampling and storage

Evaluation runs both conditioned and unconditioned NLL, with separate model
forwards. It is eager even when training is compiled. Avoid changing those
metrics while measuring a compiler or loader experiment. A cheap fixed IID
subset more often and full IID/probe less often is a possible future split, but
must retain enough examples for rare subjects and edge/source diagnostics.

P32's grid sampler performs 256 growing-prefix trunk forwards per prompt group
and four slot draws per patch. Unlike the chat sampler it has no KV cache.
Measure its long-run share first. Increasing the grid interval is a small,
reviewable cost/visibility tradeoff; a KV-cache rewrite is a larger numerical
and implementation risk. The accepted duplicate skip avoids only repeated
final-step work, not periodic cost throughout the run.

CPU shape inspection calculated 66,424,320 trainable parameters and 265,697,280
bytes of FP32 state tensors. A model plus two FP32 AdamW moments is approximately
797,091,840 bytes of tensor payload, an estimate excluding serialization/RNG
overhead. These are not measured checkpoint file sizes. See
[the reproducible output](evidence/pixie-stack-facts.json).

At the default 100k-view save interval, a 12M run creates up to 120 permanent
model snapshots, calculated. Their state payload alone is approximately
31.88 GB; overwritten full `last.pt` and local `best.pt` add write traffic.
100k views take roughly 280 seconds at the supplied 357 images/s estimate,
before reporting. Decide a recovery-time budget, then decouple latest full
recovery saves from sparse permanent milestones. Do not delete old checkpoints
to implement this policy. Asynchronous saving adds immutable-copy RAM demand
and crash semantics, so first test lower frequency and measure actual write
time and bytes.

### Quality levers need controlled data experiments

The builder combines rare-subject frequency to power -0.35, a cap of 6 and an
edge-heavy multiplier of 0.5, then normalizes training weights. The edge factor
changes sampling probability, not per-pixel loss. A cropped object and a valid
full-frame object can share that bucket. Compare blinded junk/usefulness labels
and model performance by bucket before changing the multiplier. Audit source
mass, subject mass, effective sample size `(sum w)^2/sum(w^2)` and expected
unique coverage under replacement sampling. R1 retention is not a quality score.

The P32 builder removes exact quantized pixel duplicates. First-winner ordering
can discard useful alternate captions for identical pixels. Audit caption
disagreement before choosing a canonical caption or a multi-caption policy.
Near-duplicate sheet/family variants can still dominate; group them before
splitting experiments. IID is sheet-disjoint. Probe subjects are unseen, but
other subjects from probe sheets can remain in training after the accepted fix;
probe metrics do not establish entirely unseen sheet style.

Caption strings must match sidecars, metadata and provenance, but that does not
prove semantic correctness. Audit generated-sheet subject swaps, truncation at
the CLIP limit, template boilerplate and directional mistakes. Preserve original
and corrected captions with provenance. P32 currently draws 10% blank, 15%
subject-only and otherwise full captions; it does not implement v3's synonym
and typo mixture. Treat those as separate ablations, not omitted defaults.

Palette conditioning currently draws 30% top-one, 50% top-three and 20% absent.
Style is retained with probability 0.70. Blank text can still carry conditioning.
Conditioned CFG must therefore compare blank/full text with the same requested
condition. Evaluate CFG, temperature, top-k and conditioning compliance on a
development set at fixed sampling budgets. Never use reward score alone as the
quality metric or assume stronger CFG always improves sprites.

The supplied loss near 0.52 is a training minibatch value, not evidence of held-out
sprite quality. Training uses alpha-weighted loss while evaluation also reports
unweighted NLL. Compare the same metric on the same split at matched views.

## Inference on the 3060 Ti

The existing chat [FastSampler](../worker/fast_sample.py) already pre-fills a
static KV cache and uses CUDA graphs for decode. It generates a batch of
candidates with CFG doubling, then the worker ranks and returns a subset.
Building another generic KV cache is not a justified optimization.

The graph covers trunk decode. Slot heads and Gumbel draws remain outside it.
Graph preparation clears KV storage and prefill clears the used slice again;
that redundant work is a small profiling candidate with unknown benefit.
The current sampler still builds both branches at CFG 1, so changing that number
alone does not establish half the compute.

The old approximately 1.6-second figure has unknown timing boundaries; no
3060 Ti profile establishes which phases it includes. In
[`Gen.run`](../worker/worker_v3.py), `t_gen` includes text encoding,
sampler work, transfer to CPU and RGBA conversion. The broader worker `ms`
includes scoring/sorting/PNG work, while queue and HTTP are outside it. Existing
labels must not be read as a clean sampling-only profile.

Record text, sampler, reward, transfer/render, PNG, queue and result-upload times
separately on representative jobs, with p50/p95, request errors and candidate
counts. CUDA-event timing can isolate device work but must not turn every phase
into a synchronization in the default path. Compare full worker wall latency
after any isolated benchmark; the rejected #8 measurement excluded several of
these costs.

Reward scoring also deserves attribution. The fast reward path computes connected
components on CPU, transfers indices back to GPU, renders two backgrounds per
candidate and encodes image/text inputs for its reward models. Cache only inputs
whose identity and numerical behavior can be verified. Reducing scorer work or
candidate count can change the selected sprite even when generation is identical.

Keep the current candidate count, CFG, temperature, top-k, ranking and precision
for a behavior-preserving test. Fewer candidates can save compute but change
best-of-N quality, diversity and reward selection. A small controlled panel
can compare candidate budgets, with blinded usefulness and none-good rate.
Batching unrelated user jobs may improve throughput while worsening first-job
latency and memory use; measure queue tails before adding it to an interactive
worker on 8 GB.

The next CLIP experiment caches the entire blank candidate batch at the original
shape. It must preserve every row, not expand one row, and key by batch size,
tokens, encoder state, device and dtype. Keep it off by default until a GPU run
passes `torch.equal` for both feature tensors and exact sprite equality using
the same job seeds and conditions. Its possible gain is bounded by one blank
encode, not the factor by which duplicate input rows disappear.

BF16 versus FP32, graph versus eager, and changed batch shapes are separate
experiments. Close features are insufficient for seeded discrete sampling;
small logit differences can change subsequent autoregressive pixels. Report
feature/logit error and exact sprite match, then perform a separate quality
test if an intentional numerical change is acceptable. Never call CPU parity
proof of GPU parity. Default numeric behavior stays unchanged in these PRs.

There is a separate provenance problem. The default Gumbel sampler seeds one
job generator from `seeds[0]`; later returned per-candidate seeds do not drive
independent streams. A candidate seed alone cannot reproduce that sprite under
a new batch/order. Record honest job seed, candidate index, batch shape and
sampler version immediately in experimental artifacts. A future per-row stream
fix must test reorder/chunk invariance and intentionally version changed outputs.
The grid path also needs controlled requested seed/candidate-budget handling,
as noted in the earlier [hardware review](../reviews/pipeline-hardware-review.md).

## Data gathering and useful yield

### Compare like-for-like generation workloads

These historical baselines are supplied measurements from the stack handoff.
The inverse rates below are calculated and rounded; raw cells do not mean
accepted training samples.

| GPU/layout | Supplied measured wall time | Calculated sheets/s and raw cells/s |
|---|---:|---:|
| 3080 Ti, 3x3 at 576, in-flight two | 3.648 s/sheet | 0.274 sheets/s; 2.467 cells/s |
| 3080 Ti, 2x2 at 512, earlier configuration | 3.07 s/sheet | 0.326 sheets/s; 1.303 cells/s |
| 5070 Ti, 2x2 at 512, in-flight three | 4.805 s/sheet | 0.208 sheets/s; 0.832 cells/s |
| 3060 Ti, 3x3 at 576, CPU text encoder | 12.50 s/sheet | 0.080 sheets/s; 0.720 cells/s |

The newer 3080 flash-on result is 3.27 seconds/sheet, supplied measured,
approximately 0.306 sheets/s calculated. Keep its exact workload/settings next
to it rather than treating every row as one controlled card comparison. Unique
prompts, model/build hashes, LoRA, dimensions, seed, steps, device placement,
cache state and in-flight count must match. Keep queue-inclusive request latency
separate from wall sheets/s.

The 3060 Ti's CPU text encoding previously took about 2.37 seconds/request,
supplied measured. Repeated-prompt tests hid it behind the conditioning cache.
Moving a smaller text encoder quantization to GPU is an untested quality/memory
tradeoff, not free speed. Desktop use and the 8 GB memory constraint matter;
do not run generation alongside chat on that card.

### Diagnose the 5070 Ti streaks instead of guessing

Persist a common relative monotonic clock for request begin/end, GPU samples,
decode/compress/extract work and safe server-phase timings. Include utilization,
dedicated VRAM, SM/memory clocks, temperature, power, P-state and supported
clock-event/throttle reasons. Unsupported fields must stay explicitly missing,
not turn the entire GPU log into an empty file. Keep raw asset paths, hostnames,
GPU UUIDs and server text out of report artifacts.

Group requests by concurrency and workload before declaring a slow streak.
At in-flight three a request's latency includes competing jobs. A long request
can straddle a fast and a slow GPU interval. Report overlap and sample coverage;
do not infer GPU causality from a wall-time correlation alone.

Hypotheses to test are power/thermal clock limiting, memory pressure/offload,
quantized-kernel or flash dispatch, CPU scheduling, text-cache misses and
submission/PNG/storage backpressure. Native Blackwell support and a different
sd.cpp build are separate controlled comparisons. The current data does not
select a cause. Keep dimensions, quality settings and model hashes fixed first.
The pinned [sd.cpp performance guide](https://github.com/leejet/stable-diffusion.cpp/blob/984eb06/docs/performance.md)
also calls for architecture-dependent flash/convolution comparisons.

### Optimize accepted useful sprites per GPU-hour

Three-by-three produces more cells per request, but compare accepted clean
subjects per hour after extraction, R1, semantic review and deduplication.
Different LoRAs and layouts are quality confounders. Keep the baseline five-step
Turbo schedule initially. Test fewer steps or another LoRA only on matched
prompts/seeds with human acceptability, subject correctness, extraction/R1 yield
and diversity. A faster sheet with more unusable cells is not a training win.

The locked extractor source uses seed tolerance 90, near-magenta flood growth
120 over six screen pixels, and two outline-fringe rounds at tolerance 100.
Its module header and portions of the dataset prose still mention older 110
tolerance. Trust executable constants and the locked AGENTS recipe; correct
documentation separately rather than changing gates to match stale text.
The 3x3-at-576 calibration is not proof of equal behavior on 2x2 sheets.

The combined re-extractor already memoizes a full-sheet grown mask. Proposing
that optimization again would duplicate existing work; benchmark the actual
combined path before changing extraction kernels.

R1 rejects excessive magenta outline pixels and tiny components. It does not
measure wrong subjects, malformed anatomy, duplicate semantics or asset utility.
AGENTS describes an intended SigLIP so400m semantic filter, rejecting negative
cosine similarity to the subject phrase, but that script is absent from this
code copy. Its application to the current corpus has not been verified here.
Recover its version, model/preprocessing identity, scores and admission record
before assuming it ran. Calibrate any future semantic threshold against blinded
labels rather than treating the intended zero cutoff as a proven optimum.
The historical visual checks are explicitly small-sample estimates, not a
validated junk rate. Keep a stratified, blinded random audit across both layouts,
each source, edge buckets and accepted/rejected threshold bands. Label semantic
correctness, cropping, key/fringe damage, disconnected intentional details and
usability. Estimate precision, false rejection and uncertainty per bucket before
tuning gates. Do not loosen them based on a higher automatic acceptance count.

Useful prompt coverage should be measured by subject families, visual attributes,
palette/style combinations and failure history. Avoid repeating easy prompts
solely because their ranker score is high. Preserve a random coverage component
beside targeted hard cases and record selection probabilities. Normalize and
guard all subject/caption/trimmed-token inputs against both frozen holdouts.
The benchmark does this, but legacy generation queues and flywheel exports have
guard gaps documented in the earlier reviews. Recheck them before training
admission; do not assume arbitrary supplied queues are safe.

For future reward models, retain explicit comparison groups, display exposure,
none-good/tie labels, verified edit lineage and training permission. Candidate
rank, unselected archive entries and downloads alone are not human preference
labels. Keep reward data split by prompt family/source/session and prevent
near-duplicate train/eval leakage. The earlier
[data collection review](../reviews/data-flywheel-review.md) lists concrete
export/linkage issues that remain separate work from GPU throughput.

## Hardware handoff and acceptance

Run each new branch on one idle explicit GPU with the project venv and a new
output folder. Supply paths only through arguments/environment variables.
Use the original byte-identical Blind-50 and Eval-50 files used by the dataset
manifest. Never repair a mismatch by reformatting them or changing hashes.

For timing changes, first run the CPU behavior test and one bounded hardware
correctness check. For speed claims, use at least five alternating trials with
matched work, warmup, successful update/sheet/candidate counts, median/range,
peak VRAM and quality/parity checks. Save a separate profile for attribution.
If variability exceeds the effect, call it inconclusive. Do not run paid cloud
GPUs, install packages globally or change the current live run.

## Original delivery (now merged)

Each code PR contains one correction or experiment and its own runnable checks.
The operator merged #14–#19 and supplied hardware results. The telemetry PR was
stacked on the restart fix; the other code PRs targeted main independently.
The historical follow-up table above links #20–#24, which are now merged; the
latest feedback section links the current changes.

| PR | Change | Hardware/test entry point | CPU verification, measured |
|---|---|---|---|
| #15 | Original holdout-byte preflight and actionable mismatch errors | `python -m train.pixie32.check_holdouts --data "$DATASET_OUT" --blind50 "$BLIND50" --eval50 "$EVAL50"` | Full P32 suite passed 62 tests after malformed-manifest correction. |
| #16 | Bounded immediate server restart retry | Repeat the dedicated server command in `data-pipeline/benchmarks/krea-unique.md`; CPU `test_krea_unique.py` | 14 tests passed on the restart branch. Optional extraction test skips without its dependencies. |
| #17 | Opt-in update phase profiling | `python -m train.pixie32.profile_steps`; full command in `train/pixie32/step-profile.md` | Full P32 suite passed 63 tests; final focused checks passed 4. |
| #18 | Default-off full-batch blank CLIP cache | `worker/benchmark_blank_cache.py`; exact-match command in `docs/chat-blank-cache.md` | 6 tests passed, including real synthetic CLIP and sampler parity. |
| #19 | Krea read-only telemetry and offline slow-streak analysis | `data-pipeline/benchmarks/krea_telemetry.py record` / `analyze`; see `krea-telemetry.md` | Combined Krea suite passed 21 tests. |

Use `$PY`, the project venv executable, in place of `python` above. Run the
holdout check first, then capture a bounded training profile and a representative
generation telemetry window. Test the CLIP experiment with exact GPU equality
before enabling it. No new GPU speed gain, GPU output identity, semantic-filter
application or improved junk rate is established by these CPU checks.

Independent review found one malformed-manifest boundary case in the preflight;
it is fixed and regression-tested. Report review also identified the missing
semantic-filter provenance noted above. The latest measurements establish the
broad update limiter and exact blank-cache parity. Remaining uncertainties are
the best compute setting at b80, long-run reporting share, full-corpus junk and
coverage, full worker attribution on the 3060 Ti, and the 5070 Ti slowdown cause.
