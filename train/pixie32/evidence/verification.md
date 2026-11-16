# Native32 verification

All results below are CPU checks on handoff `b49864e` plus this implementation.
No training run or GPU benchmark was launched. The operator will run the real
warm-start smoke with private local assets. The 12 GB memory target and
production throughput remain unmeasured.

Environment: Python 3.13.7, torch 2.9.0+cu128, NumPy 2.5.4, Pillow 13.0.0,
tokenizers 0.23.3, installed only in the project venv.

## Reproduce

```bash
"$PY" -m unittest discover -s train/pixie32 -p 'test_*.py' -v
"$PY" -m train.pixie32.verify_sample \
  --sample "$SAMPLE_DIR" --out "$RUNS/sample-check-001"
"$PY" -m train.pixie32.verify_sample_cache \
  --sample "$SAMPLE_DIR" --out "$RUNS/sample-cache-check-001"
bash -n train/pixie32/run_p32.sh train/pixie32/stop_p32.sh
```

The final suite passed **34 tests in 12.385 seconds**. The unit-test output is
in [unit-tests.txt](unit-tests.txt). Tests exercise
patch/slot order, patch and slot causality, prefix rotary identity, chunked
loss and gradients, full-shape warm-start loading, frozen CLIP loading,
manifest corruption and shuffled identities, strict holdouts, deduplication,
sheet splits, weights, CLI/env resolution, launcher guards, and exact resume.

The full v3 fixture has 224 float32 tensors and 66,309,632 parameters. Its
model-only checkpoint copies 219 tensors and remaps five spatial tables into
P32, which has 66,424,320 parameters. The full CLIP fixture has 178 float32
tensors and 115,971,072 elements; the actual loader produces finite frozen
features of shape `[1,18,768]`. Fixture weights are synthetic.

The trainer-loop test compares uninterrupted updates with both smoke-paused
and SIGINT-paused continuation into fresh output folders. Model weights,
optimizer tensors and torch RNG match exactly on CPU. Changed batch size is
refused. Time-limited runs write `paused.json` and no completion summary.
These tests use a small test-only model and CLIP fixture; they do not establish
CUDA bitwise reproducibility or convergence.

## Supplied 200-image sample

The read-only check found 200 unique native 32×32 images, 200 nonempty captions,
alpha values `{0,255}`, and 28 edge-list intersections. The report is
[sample-validation.json](sample-200-b49864e/sample-validation.json). The sample
itself and its contact sheet are not part of the public tree; `$SAMPLE_DIR` in
the commands below is any local folder with the same layout.

The actual builder and verified loader admitted all 200 rows using supplied
provenance and an explicitly synthetic RGB-cube palette and 49,408-entry BPE
tokenizer. Splits contained 180 training-loader rows, 10 IID rows, and 10 probe
rows. Holdout matches were zero. Token and caption shapes were `[200,1024]`
and `[200,18]`; 28 rows received the edge multiplier. Mean training sampling
weight was 0.9999998211860657 after normalization.

The fixture's colour-count thresholds were 11 and 17. These are loader-test
measurements with a synthetic palette, **not production calibration values**.
The real dataset build recalculates them with the operator's actual palette.
No sample row was used in an optimizer update.

See [sample-cache-validation.json](sample-cache-200-v2/sample-cache-validation.json)
and the generated [dataset manifest](sample-cache-200-v2/dataset/manifest.json).
Generated cache arrays, synthetic assets and temporary checkpoint fixtures
are omitted from the PR; the scripts reproduce them in a new folder.

## Operator smoke

With `PY`, `DATASET_OUT`, `RUNS`, `GPU`, `V3_CKPT`, `CLIP_DIR`, and `TOKENIZER`
set, run:

```bash
export RUN_NAME=p32-smoke-001
bash train/pixie32/run_p32.sh --smoke --smoke-steps 200 --batch 32 \
  --gradient-checkpointing --ce-chunks 16
```

Retain the printed loss, training step/s, and peak allocated/reserved VRAM,
plus `paused.json` and `samples_*.png`. Confirm headroom below approximately
11 GB reserved before launching the proposed full recipe in the README.
Throughput is an average including the first update; evaluation, sample
generation and checkpoint time are excluded. Peak VRAM includes evaluation
and sample generation. No speed or memory claim is based on fixture timing.
