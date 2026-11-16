# Replaying a candidate batch

The worker now honors a comparison job's `seed` and its requested candidate
budget. Ordinary chat keeps the configured reward best-of-N budget. Seeds must
be integers from 0 through 2147483647; missing seeds are generated once per job.

The existing fast Gumbel sampler uses one PyTorch generator for the complete
batch. The `sampling` record names this scheme `torch-gumbel-batch-v1` and stores
the job seed, actual generated batch size, sampler settings, graph/cache flags,
PyTorch/CUDA versions, dtype and TF32 setting. Existing model metadata supplies
the checkpoint hash. Each result's `idx` is its original row before reward
sorting. The compatibility `seed` field now repeats the actual job seed for
every candidate. It is never an independent candidate seed.

Replay the complete batch with the same prompt, conditions, checkpoint, CLIP
weights/tokenizer, hardware, software and math settings. Then select `idx` from
that batch. Changing the batch size or row order changes RNG consumption and
can change GPU numerics. No individual-seed, chunking, row-reorder or
cross-runtime sprite equality is promised. Older records without `sampling`
retain unknown provenance. Existing archives and holdouts are not rewritten.

This supersedes the report's proposed per-row RNG change: preserving the
existing fast sampling stream keeps this correction separate from an unmeasured
kernel/quality change. Grid requests now use their explicit small budget;
their quality is no longer best-of-64 when requesting three candidates.

The queue preserves `sampling` and original row indices, comparison grids save
it on each cell, candidate archives save it in their metadata, and chat votes
and JSONL exports retain it. Legacy CSV seed columns remain batch seeds and do
not contain the full replay record; use JSONL for replay provenance.

Run the CPU behavior checks without any model assets:

```sh
(cd worker && "$PY" -m unittest test_sampling_provenance)
node web/tests/seed_provenance_test.mjs
```

On an idle, explicitly selected GPU, run five full-batch replay trials using
local assets. This does not poll the app or load a worker key. Use a new output
file and the same math/cache/graph settings as the intended comparison:

```sh
"$PY" worker/benchmark_blank_cache.py --device cuda:0 --batch 64 \
  --checkpoint "$V3_CKPT" --clip-weights "$CLIP_WEIGHTS" --tokenizer "$TOKENIZER" \
  --seed 123 --warmup 5 --trials 5 --verify-seed-replay \
  --out "$RUNS/seed-replay-001.json"
```

The additional check requires exact sprites on repeated complete-batch draws
and when changing the unused tail seed entries. Any mismatch exits nonzero.
Run again with `--batch 3` to check the comparison budget; comparisons across
those two batch sizes are deliberately not an equality test. The benchmark
times CLIP plus sampling only, excluding reward, PNG, queue and HTTP. GPU
replay and performance remain **unmeasured** until the operator runs it.
