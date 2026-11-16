# Test full-batch blank CLIP reuse

PR #8 changed CLIP's candidate batch to one row and expanded its features. The
operator measured maximum absolute feature error 1.0 and 85/960 exact sprite
matches on the 3080 Ti. That change was rejected. This experiment retains the
original batch size for **both** CLIP calls and caches the entire blank result.
It never expands one row or substitutes the first row for the others.

`PIXEL_CACHE_BLANK=0` is the default. An operator can set it to `1` on a future
worker invocation only after its hardware benchmark passes. No running worker
or production configuration is changed by this PR.

The cache holds one `[candidates, text_len, 768]` tensor. A batch of 64 at text
length 18 uses an estimated 1.69 MiB in BF16, excluding tensor metadata. It is
recomputed when the candidate count, blank token row, encoder, weight versions,
device, dtype or relevant math-backend settings change. Normal
`load_state_dict` and in-place parameter updates invalidate it. Call `clear()`
after mutations that bypass tensor version counters, such as direct `.data`
writes. Consumers treat the cached tensor as read-only.

The prompt still encodes all candidate rows on every job. A warm job skips one
blank CLIP call. Candidate count, CFG, temperature, top-k, conditions, sampler,
random seed path and reward ranking remain unchanged. Expected hardware gain
is **unmeasured** and bounded by the removed blank encoding. The previous
benchmark's CLIP timing cannot be transferred to the 3060 Ti or interpreted as
full worker latency.

## Run the exact-parity hardware benchmark

Use an idle GPU with the chat worker and other compute users paused. The
benchmark checks the selected physical index before CUDA setup. Unset
`CUDA_VISIBLE_DEVICES`. Supply explicit local assets through arguments or
`V3_CKPT`, `CLIP_WEIGHTS` and `TOKENIZER`. No worker config, key, reward-model
asset, network endpoint or palette file is read.

```sh
"$PY" worker/benchmark_blank_cache.py \
  --checkpoint "$V3_CKPT" --clip-weights "$CLIP_WEIGHTS" --tokenizer "$TOKENIZER" \
  --device cuda:0 --batch 64 --warmup 5 --trials 15 \
  --out "$RUNS/blank-cache-b64-001.json"
```

Use a new output filename. Repeat with
`--palette-ids 1 2 4097 --style 3 1 2`, with batch 16 and with `--no-graph`
as separate comparisons. Test every candidate batch used by the worker. Keep
the same precision and graph mode on both sides. Each measured pair uses the
same prompt and seeds; execution order alternates after warmup. The benchmark
reports synchronized CLIP, sampling and combined times, median/range and peak
allocated VRAM. The cache is warm, and startup/graph capture are excluded.

Every measured pair must satisfy `torch.equal` for both complete CLIP tensors
and for all sampled palette-index sprites. There is no tolerance flag. The
JSON report is written even when parity fails, marks `comparison_valid=false`
and the command exits nonzero. Do not enable the worker option if it fails.

This benchmark excludes reward scoring, RGBA/PNG work, queueing, polling and
HTTP. It establishes neither full chat latency nor reward ranking throughput.
CPU tests establish behavior but cannot prove GPU BF16 parity. No measured
GPU speedup or GPU output identity is claimed in this PR.

## Run CPU behavior tests

```sh
"$PY" -m unittest discover -s worker -p test_blank_features.py -v
```

Tests preserve distinct per-row outputs, cache invalidation and candidate-batch
shape. A real synthetic-weight CLIP layer and FastSampler check exact features
and seeded sprites with default and explicit palette/style conditions. Report
tests verify that even a one-ULP feature difference fails and is saved as an
invalid comparison.
