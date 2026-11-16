# Measure P32 training and reporting time

The existing `images_per_second` measures augmentation, CLIP encoding or cached
feature transfer, and completed training updates. It excludes evaluation,
sampling and checkpoint writes. Use the new segment clock to find how much
those phases affect the time needed to finish a run.

`train_summary.json` and `paused.json` now contain these fields:

| Field | Scope |
|---|---|
| `segment_images` | Image views completed in this invocation, excluding inherited checkpoint progress |
| `segment_wall_seconds` | Time after asset loading through final evaluation, grid rendering and checkpoint writes |
| `wall_images_per_second` | Segment image views divided by segment wall seconds |
| `phase_seconds`, `phase_counts` | Separate training, evaluation, sampling and checkpoint totals and invocation counts |
| `unclassified_seconds` | Remaining segment time, including epoch-plan creation, logging and phase dispatch |

The segment clock includes first-step compilation. It excludes model/cache
loading and the final summary write. GPU reporting phases synchronize at their
boundaries. Training retains its existing per-update synchronization. Best-model
writes count as checkpoint time, outside evaluation time. The phase totals do
not overlap.

Finalization reuses evaluation metrics and the sample grid only when they
already ran at the same optimizer step in this invocation. If training changed
the weights afterward, finalization evaluates and renders again. It still
writes the full resumable checkpoint. At an exact periodic boundary this removes
one redundant evaluation and one redundant grid. Actual GPU time savings are
untested and depend on how often a run ends at such a boundary.

## Run five short segments

Use the full `last.pt` from the new b80/cache branch. Do not use its model-only
`best.pt`, and do not resume the old b64/online checkpoint with changed arguments.
The branch-from PR prepares the changed recipe in a fresh folder first.

The benchmark requires the exact recipe saved in the full checkpoint, including
the total image target, schedule, seed, reporting intervals and compile/cache
flags. Retain enough remaining image budget for all requested steps in each
independent segment. Replace the values below only to match that saved recipe.

```bash
"$PY" -m train.pixie32.bench_reporting \
  --out "$RUNS/reporting-bench-001" --steps 100 --trials 5 -- \
  --data "$DATASET_OUT" --resume "$P32_CKPT" \
  --clip-dir "$CLIP_DIR" --tokenizer "$TOKENIZER" \
  --clip-cache "$CLIP_CACHE" --clip-cache-max-gb 4 \
  --device cuda:0 --batch 80 --compile --ce-chunks 16 \
  --no-gradient-checkpointing --images 12000000 --warmup 1500 \
  --decay-images 2000000 --lr 1e-4 --lr-min 1e-5 --seed 42 \
  --save-every-images 100000 --eval-every-images 100000 \
  --sample-every-images 100000
```

Select an idle GPU and unset `CUDA_VISIBLE_DEVICES`. The wrapper checks the
explicit physical GPU before each child process and refuses occupied devices.
Each trial resumes the same checkpoint in a new `trial-NN` folder. It retains
those full output artifacts and writes an aggregate `benchmark.json` containing
phase summaries, counts, medians/ranges and failures. Input arguments, paths,
prompts and child output are omitted from that aggregate.

This is a phase-profile benchmark, not an A/B speedup claim. No warmup steps
are excluded because it measures whole bounded segments, including the first
compiled update and final reporting. Repeated short segments exaggerate final
reporting relative to a long run. A 100-step segment can miss periodic work;
inspect phase counts and the real run's reporting intervals before changing
cadence. Do not infer steady-state training speed from this measurement.

The trainer always renders/evaluates at smoke completion. It skips the final
duplicate only if a periodic invocation already handled the final step. No
reporting interval, prompt count, extraction gate or training recipe changes.

## CPU verification

```bash
"$PY" -m unittest train.pixie32.test_reporting -v
```

Synthetic trainer fixtures check single invocation at the final periodic
boundary, normal final reporting, unchanged model/RNG/metrics, exclusive phase
accounting, segment rather than inherited image counts, and five fresh benchmark
outputs. They do not measure hardware throughput or sample rendering speed.
