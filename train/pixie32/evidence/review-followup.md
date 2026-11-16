# PR #4 follow-up verification

This change addresses the review of launched revision `06dafb0`. It changes
code for future builds and runs. No existing dataset, checkpoint or running
training process was modified.

The full CPU suite passed **43 tests in 16.309 seconds** on torch 2.9.0+cu128.
The output is in [review-followup-tests.txt](review-followup-tests.txt).

```bash
"$PY" -m unittest discover -s train/pixie32 -p 'test_*.py' -v
"$PY" -m train.pixie32.verify_sample_cache \
  --sample "$SAMPLE_DIR" --out "$RUNS/review-sample-check-001"
```

The mixed-subject sheet fixture retains all 48 rows, with train/IID/probe
counts 40/4/4. Running the `06dafb0` builder on the same fixture admitted
44 rows and dropped four non-probe partner rows.
The loader accepts train/probe sheet overlap and still rejects IID sheet
overlap and probe subjects appearing in training. This fixture does not
measure recovered rows in the full production corpus.

Holdout tests cover hyphens, underscores, em dashes, CJK punctuation and
quoted words. Word boundaries still reject partial-word false positives.
The builder checks captions, subjects and decoded trimmed captions. Startup
also rejects newly recognized matches in existing caches.

An OOM exception injected into the CLI reproduced the old missing-traceback
failure. With the fix, stderr contains the complete traceback and original
exception message, and the command exits with code 1. No GPU OOM was induced.

The configured-grid test uses five fixture sprites, two columns, scale three
and sample batch two. The PNG measures 192×288 pixels, preserves transparent
pixels and nearest-neighbour blocks, and records prompts in row-major order.
Sampler calls use batches 2/2/1. Global training RNG and the prior model mode
are preserved. File prompts are filtered before creating a run folder.

The optional compiled path uses the bound training loss, leaving model state
keys unchanged. A real tiny P32 loss with gradient checkpointing and 16 CE
chunks passes Dynamo's full-graph capture using the CPU `eager` backend.
Loss and gradients match uncompiled execution. The trainer-loop fixture also
resumes compiled CPU updates with matching model, optimizer and RNG tensors.
Switching compiled execution on or off is refused for exact resume. Default
eager fingerprints retain the `06dafb0` format.

CUDA/Inductor execution, compilation latency, production throughput and GPU
memory were not measured. No speedup is claimed. `--compile` stays opt-in.

The supplied 200-image sample still passes the actual builder and loader
with explicitly synthetic palette/tokenizer inputs: 200 admitted rows,
splits 180/10/10, 28 edge rows and zero holdout matches. The report is
[sample-cache-validation.json](review-followup-sample/sample-cache-validation.json).
No sample row was trained on. The 11/17 colour thresholds are fixture values,
not production calibration. Generated caches and synthetic assets are
excluded from the PR.

Model the Domain kept probe assignment based on subject while IID isolation
remains based on sheet. Prove It Works led to the mixed-sheet, traceback,
grid-pixel and compiled-gradient behavior checks.
