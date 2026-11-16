# Native 32×32 Pixie

This folder implements a separate P32 training path. It imports the v3 CLIP
encoder and causal slot head without changing the live v3 trainer or worker.
Only the operator launches the full run.

P32 uses 256 raster-ordered 2×2 patches and a sequence length of 279. The
23-token prefix contains BOS, 18 CLIP positions, three palette slots and one
style slot. SEP predicts patch zero; each following position predicts the next
patch. Half of each attention head uses axial 2D-RoPE, divided equally between
row and column; the prefix has zero coordinates. The shifted image position
uses its **target** patch coordinates. The existing causal slot head predicts
TL, TR, BL and BR in order.

Warm-start loading requires a compatible v3 P50 configuration. It copies all
shape-compatible tensors, retains the learned prefix positions, and expands
the learned 8×8 patch positions and row/column tables to 16×16 by nearest
neighbour repetition. RoPE adds no learned tensors. Loading reports copied,
remapped and new keys and refuses incompatible architectures or tensors.

## Prepare data

Use a project venv with Python and the packages in `requirements.txt` (PyTorch
2.9.0, CUDA 12.8). All paths below are operator-supplied placeholders.

```bash
"$PY" -m pip install -r train/pixie32/requirements.txt
"$PY" -m train.pixie32.build_data_p32 \
  --data-root "$DATA_ROOT" --r1 "$DATA_ROOT/r1_list.txt" \
  --edge-list "$DATA_ROOT/edge_px_ge32_list.txt" \
  --palette "$PALETTE" --tokenizer "$TOKENIZER" --out "$DATASET_OUT"
```

The builder requires `manifest.jsonl`, `final/metadata.jsonl`, PNGs and caption
sidecars. It admits reviewed R1 rows, verifies their provenance and caption
agreement, removes frozen holdout matches from caption and subject before and
after CLIP truncation, and deduplicates quantized images. Holdout matching uses
case-folded text with Unicode punctuation replaced by spaces and phrase
boundaries, including occurrences within longer captions. The defaults are
the frozen repository Blind-50 and Eval-50 JSON
files. Each must contain exactly 50 distinct, nonempty lowercase phrases.
Missing or malformed holdouts are errors; `--blind50` and `--eval50` can
override their locations.

IID evaluation holds out entire source sheets. The probe holds out subjects.
Other subjects sharing probe sheets stay in training, so the probe measures
unseen subjects and can share sheet style with training. IID sheets never
overlap either split. Exact evaluation captions are removed from training.
Rare-subject sampling uses
alpha 0.35 with a relative cap of 6; edge-list rows receive a further 0.5
multiplier. Training weights are normalized to mean one.

Fill thresholds are 320 and 720 opaque pixels. Colour-count thresholds are
the training split's nearest 1/3 and 2/3 quantiles; their actual values are
written to the dataset manifest. No production thresholds have been measured
from the supplied sample. Every array carries the ordered-ID digest and has
its own file digest. The trainer verifies all digests, dimensions, IDs,
holdouts and split separation before training.

## Smoke, launch and resume

Choose one idle physical GPU with `GPU`; the launcher checks only that device,
then exposes it as `cuda:0`. Set `PY`, `DATASET_OUT`, `RUNS`, `V3_CKPT`,
`CLIP_DIR`, and `TOKENIZER` to local inputs. `CLIP_DIR` holds
`clip_l14_text_l11_f32.pt`. Direct CLI flags take precedence over the matching
asset environment variables. The warm start loads model weights and creates
a fresh optimizer; the source v3 checkpoint has no optimizer state.

```bash
export RUN_NAME=p32-smoke-001
bash train/pixie32/run_p32.sh --smoke --smoke-steps 200 --batch 32
```

The smoke uses the full model and real CLIP inputs. It prints loss, training
step/s, peak allocated and reserved VRAM, evaluates IID/probe NLL and writes a
native 32×32 sample grid. It saves a resumable checkpoint and `paused.json`.
Training throughput includes batch preparation, online CLIP, forward,
backward and optimizer steps; evaluation, sampling and checkpoint I/O are
excluded. Peak VRAM includes those later phases. No 12 GB fit is claimed until
this smoke runs on the target GPU. Reduce batch size for a **new** smoke if
needed; a checkpoint cannot be resumed with a changed batch size.

Grids default to up to 12 everyday prompts, four columns and 4× nearest-neighbour
enlargement. Each cell still comes from a native 32×32 sample. Configure them
with a UTF-8 prompt file, one prompt per line:

```bash
bash train/pixie32/run_p32.sh --smoke --sample-prompts "$SAMPLE_PROMPTS" \
  --sample-scale 6 --sample-columns 4 --sample-batch 2
```

The trainer filters original and CLIP-trimmed sample prompts against both
holdouts before creating the output folder. Empty files and files with no
safe prompts are errors. `--sample-batch` limits simultaneous prompts before
CFG doubles the model batch. Larger grids therefore need not increase the
sample batch. Each PNG has a JSON sidecar recording its ordered prompts,
layout, scale, batch size and seed. `sampling.json` records the run settings.
Display options can change on resume without changing the training recipe.

`--compile` optionally applies `torch.compile` with its default Inductor
backend to the training loss. Evaluation and sampling remain eager, and
checkpoints retain the original tensor names. Compilation is off by default.
Its first-use cost is included in reported training time. CUDA/Inductor
throughput and memory use have not been measured here. Run a fresh smoke
with `--compile` before choosing it for a new run. Exact resume refuses a
change between eager and compiled execution; existing eager checkpoints from
`06dafb0` keep their fingerprint format.

Unexpected errors, including CUDA OOMs, print the complete traceback and
exception message to stderr and return exit code 1.

Proposed first full recipe: 12 million image views, including a final 2 million
views of linear decay, batch 32, 1,500 warmup steps, peak LR `1e-4`, minimum
LR `1e-5`. The lower peak is a conservative starting point for transferred
weights rather than v3's from-scratch `3e-4`; compare IID NLL and samples before
committing to a longer run. Twelve million views are about 90 passes over a
134k-image corpus (arithmetic estimate, weighted sampling changes coverage).
At batch 32 this is 375,000 steps; wall time in hours is
`375000 / measured_steps_per_second / 3600`, plus evaluation and I/O.

```bash
export RUN_NAME=p32-first-001
bash train/pixie32/run_p32.sh --batch 32 --images 12000000 \
  --decay-images 2000000 --warmup 1500 --lr 0.0001 --lr-min 0.00001 \
  --gradient-checkpointing --ce-chunks 16
```

Do not run this full command until the smoke establishes memory headroom on
the selected GPU (target: below approximately 11 GB reserved). Checkpointed
transformer blocks and chunked CE are enabled by default. Arrays are memory
mapped on the host, and only the current batch moves to the GPU. The token
cache alone is about 274 MB for 134k rows (estimate, decimal units).

`best.pt` tracks the best IID NLL within the current output folder. On resume,
the inherited global best metric is tracked separately, and the first local
evaluation always produces a local best checkpoint. Periodic `last.pt` contains optimizer state, RNG,
epoch/offset and the recipe fingerprint; `img_*.pt` files are model-only.
Evaluation and grids run every 100k image views by default. A time limit or
SIGINT saves `last.pt` and `paused.json`; only a completed image budget writes
`train_summary.json`.

```bash
bash train/pixie32/stop_p32.sh
export RESUME="$RUNS/p32-first-001/last.pt"
export RUN_NAME=p32-resumed-001
bash train/pixie32/run_p32.sh --batch 32 --images 12000000 \
  --decay-images 2000000 --warmup 1500 --lr 0.0001 --lr-min 0.00001
```

Resume always writes into a **new** output folder. Keep the original recipe
arguments. Changed batch size, dataset, seed, model or learning-rate schedule
is refused. Epoch draws use a local seed, and full checkpoints restore the
training RNG rather than deriving progress from `step × batch`.

The probe change affects newly built dataset folders. Rebuild to a new
`$DATASET_OUT` for a future run to retain the previously dropped rows.
Existing cache and checkpoint fingerprints remain unchanged. Resuming a
checkpoint against a rebuilt dataset is still refused; the current run can
continue with its original inputs. Startup uses the stronger punctuation
holdout check even for an existing cache and refuses any newly detected match.

## Optional frozen CLIP cache

Online CLIP remains the default. For a future run, encode every unique caption,
subject caption and blank token row once, then keep features in host RAM. The
trainer uses the same deterministic caption draws and image augmentations;
evaluation also uses cached features. Arbitrary sample prompts still use online
CLIP, so its GPU weights stay resident. This change does not free their VRAM.

```bash
"$PY" -m train.pixie32.build_clip_cache \
  --data "$DATASET_OUT" --clip-dir "$CLIP_DIR" --out "$CLIP_CACHE" \
  --device cuda:0 --batch 128 --clip-cache-max-gb 4

# Add these flags to a future new-folder training invocation:
# --clip-cache "$CLIP_CACHE" --clip-cache-max-gb 4

"$PY" -m train.pixie32.bench_clip_cache \
  --data "$DATASET_OUT" --clip-cache "$CLIP_CACHE" \
  --checkpoint "$P32_CKPT" --clip "$CLIP_DIR/clip_l14_text_l11_f32.pt" \
  --device cuda:0 --batch 64 --no-gradient-checkpointing --ce-chunks 16 \
  --steps 30 --warmup 10 --trials 5 --out "$RUNS/clip-cache-bench-001"
```

Run the builder/benchmark only on an idle GPU with `CUDA_VISIBLE_DEVICES` unset;
their explicit index refers to the physical GPU. They refuse an occupied GPU.
Never stop the current training to run this check. Outputs must be new folders.
The benchmark reads a P32 checkpoint, performs bounded training updates without
saving model weights, alternates five trials per mode and reports medians,
ranges, finite losses, CLIP feature parity and peak VRAM in `benchmark.json`.
Its timing includes augmentation, feature transfer/encoding and completed
optimizer updates; startup, evaluation and sample rendering are excluded.
Keep the current recipe flags when comparing; add `--compile` only if both sides
should be compiled. A limiter/profile check is still needed to interpret results.

Each feature row occupies 27,648 bytes: 89,000 unique rows would require about
2.46 GB before subjects, index overhead and other process memory (estimate).
`--clip-cache-max-gb` caps estimated cache RAM, not total process RAM. Loading
verifies dataset/ordered IDs, tokenizer, CLIP weights, token coverage, shapes,
finite features and file hashes, then reads features into RAM once. The feature
file may live on the HDD; training performs no random reads from it. CUDA builds
use the trainer's BF16 CLIP path with FP16 storage and finite/range checks. Very
small BF16 values can round in FP16; the builder records maximum storage error
and the benchmark checks online/cache feature error. CPU builds use FP32 encoding
rounded to FP16 and are accepted for CPU training only. Caches must be rebuilt
for a different device type or dataset.

Cached mode and its cache digest are part of the resume fingerprint. Existing
online checkpoints retain their original fingerprint and cannot silently switch
to cached training on resume. Use the original flags/cache for exact continuation.

The operator measured **335 to 347 images/s**, about **3.5%**, on the 12 GB
3080 Ti with compile, CE chunks 16 and five trials per mode. Loss was identical
in that comparison and VRAM unchanged. The cache held 86,906 rows in 2.43 GB
and built in 52 seconds. These are supplied hardware results, not local GPU
measurements. Trial ranges and a limiter profile were not supplied. See
[`docs/optimization-results-20261115.md`](../../docs/optimization-results-20261115.md)
for the accepted and rejected experiments. Test batch 80 plus the cache together
before treating their separate gains as a measured combined improvement.

## Continue a paused P32 run with a new batch and cache

Use `--branch-from "$P32_CKPT"` to start a **new output folder** from a full
`pixie32-full-v1` checkpoint. This carries the model, AdamW moments, saved
optimizer step, total image views and training RNG. It preserves WSD progress
and does not restart warmup. It differs from the v3 warm start, which creates
a fresh optimizer. Model-only `best.pt` and `img_*.pt` files are refused.

Batch size, cache, compile, CE chunks, block checkpointing and save/eval/sample
intervals can change through this explicit branch operation. Dataset, tokenizer,
CLIP weights, seed, model, loss, augmentation and the entire WSD schedule must
match the parent. The total `--images` budget includes the parent's saved views.
Progress always comes from `images_seen`, never `step * new_batch`.

The branch starts a new deterministic epoch plan at epoch zero and offset zero
using the recipe seed. It may revisit samples already seen by the parent.
It retains the parent's global best IID NLL and resets the new folder's best.
Save/eval/sample boundaries start at the next image-aligned multiple of each
interval after the parent's saved views. `branch.json` and full checkpoints
record the parent checksum, progress, recipe digests and allowed changes.
They contain no input paths. Exact resumes of this new branch keep that record
and restore its batch plan rather than restarting it.

For the paused batch-64 run, keep LR at `1e-4`, warmup at 1500 steps and decay
over the final 2 million of 12 million total views. This is a conservative
continuation choice, **not a measured optimum for batch 80**. No automatic
linear or square-root LR scaling is applied. Batch 80 makes fewer optimizer
updates for the remaining views, so its optimization trajectory differs.

Set `P32_CKPT` to the paused full checkpoint, and `CLIP_CACHE` to the verified
cache. The operator reported batch-80 fit without block checkpointing. A smoke
must still check the combined cache/batch configuration, evaluation and grids.
The launcher checks only the selected GPU and refuses a busy device.

```bash
unset RESUME
export RUN_NAME=p32-cache-b80-smoke-001
bash train/pixie32/run_p32.sh \
  --clip-cache "$CLIP_CACHE" --batch 80 --compile --ce-chunks 16 \
  --no-gradient-checkpointing --lr 1e-4 --lr-min 1e-5 \
  --images 12000000 --decay-images 2000000 --warmup 1500 \
  --smoke --smoke-steps 20
```

Keep the original seed and all immutable parent recipe flags. If they differ
from the example, use the original values. The launcher supplies
`--branch-from "$P32_CKPT"`. Its `--resume` source instead comes from `RESUME`;
setting both source variables is an error. Direct CLI use is also supported:

```bash
"$PY" -m train.pixie32.train_p32 \
  --data "$DATASET_OUT" --clip-dir "$CLIP_DIR" --tokenizer "$TOKENIZER" \
  --branch-from "$P32_CKPT" --clip-cache "$CLIP_CACHE" \
  --out "$RUNS/p32-cache-b80-smoke-002" --device cuda:0 \
  --batch 80 --compile --ce-chunks 16 --no-gradient-checkpointing \
  --lr 1e-4 --lr-min 1e-5 --images 12000000 --decay-images 2000000 --warmup 1500 \
  --smoke --smoke-steps 20
```

Check loss, VRAM, the grid and `branch.json` before the operator launches a
long run. A passing smoke writes `last.pt`. To continue those exact smoke
updates, unset `P32_CKPT`, set `RESUME` to that checkpoint, choose another new
`RUN_NAME`, and rerun the same flags without `--smoke`. Do not switch batch or
cache through `--resume`; it still refuses any fingerprint change.

`smoke_p32_branch.sh` supplies the batch-80/cache/compile settings and bounds the
check to 20 updates through the guarded launcher. Pass the parent's immutable
recipe flags as additional arguments. It reports loss, update throughput and
peak VRAM and writes the grid. This short correctness/fit check includes compile
startup in trainer timing; it is not a five-trial steady-state speed comparison.

```bash
export RUN_NAME=p32-cache-b80-smoke-003
bash train/pixie32/smoke_p32_branch.sh \
  --lr 1e-4 --lr-min 1e-5 --images 12000000 --decay-images 2000000 --warmup 1500
```

## Verification

For evaluation, sampling and checkpoint overhead, see
[the reporting benchmark](reporting-benchmark.md). Segment wall throughput
includes those phases; the existing training-update throughput remains available.
For data, CLIP/cache, forward, backward and optimizer intervals within updates,
see [the opt-in step profile](step-profile.md).

```bash
"$PY" -m unittest discover -s train/pixie32 -p 'test_*.py' -v
"$PY" -m train.pixie32.verify_sample \
  --sample "$SAMPLE_DIR" --out "$RUNS/sample-check-001"
"$PY" -m train.pixie32.verify_sample_cache \
  --sample "$SAMPLE_DIR" --out "$RUNS/sample-cache-check-001"
```

The supplied sample is explicitly for loader tests. Its manifests were added
in handoff update `b49864e`. Read-only image/caption validation, a contact sheet,
and loader results using a synthetic palette and tokenizer are included under
`evidence/`. Synthetic fixtures exercise full-shape input loading and the
trainer without admitting sample images to training or fabricating provenance.
See `evidence/verification.md` for measured checks and remaining limitations.
