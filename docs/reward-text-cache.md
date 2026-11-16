# Measure exact reward-text reuse

The operator measured **259 ms reward scoring** within a **628 ms complete
worker generation** on the 3080 Ti. Those are supplied measurements, with no
new GPU result in this PR. Removing the entire reward phase would cap latency
reduction at **41.2%**, calculated from 259/628. This cache removes only the
still-unmeasured text-encoding subset, and only when a prompt repeats. It does
not imply a 259 ms saving or predict the 3060 Ti result.

`FastRewardV7` can cache the exact two existing batch-one reward-text embeddings
for a model/prompt. Both full image batches (light then dark), image precision,
normalization, PCA, feature order and ranking arithmetic stay the same. The
cache key includes actual tokenizer output, parameter/buffer versions, model
identity, device, dtype and attention/FP16/TF32 math settings. Cached models must
be in eval mode. A bounded LRU holds at most 256 model/prompt entries; the worker
default is zero, which disables caching.

Run a five-trial comparison on an idle GPU:

```bash
PIXEL_CACHE_BLANK=1 "$PY" worker/benchmark_reward_cache.py \
  --config "$WORKER_CONFIG" --device cuda:0 \
  --batch 64 --show 4 --candidates 20 --warmup 5 --trials 5 \
  --out "$RUNS/reward-cache-new"
```

Run the same command separately on the 3060 Ti and 3080 Ti. Supply the actual
worker's local config and keep the same checkpoint, candidate budget, scorer,
graph mode and blank-cache setting. The script loads local assets, takes no
worker lock, needs no worker key, and makes no app requests. Choose a fresh
output folder. Config paths, URLs, prompts and startup logs are not copied to
the result. Scorer fallback and load errors invalidate the comparison.

`reward-cache-benchmark.json` reports complete generation and completed reward
scoring for three workloads, with median/range and five trials per prompt:

- `uncached`: current text encoding on every request.
- `cold`: cache enabled, cleared before each request, including lookup overhead.
- `warm`: the same prompt's text features primed before timing. This is the
  repeated-prompt best case, not a representative chat hit-rate claim.

The three sides rotate order each trial and use the same full-batch seed,
prompt and candidate budget. Exact raw reward scores, rankings, shown PNGs,
stored PNGs and sampling provenance must match. The benchmark verifies actual
cache hits/misses; a skipped encode must not masquerade as faster scoring.
It reports cache capacity, entries, hits/misses and resident feature-tensor
bytes. It never caches generated images or reward scores. A mismatch makes
`comparison_valid` false and exits nonzero. Compare full-worker latency and
cold overhead as well as warm scoring time before deciding to enable it.

Separate, synchronized reward profiles on the first trial report:

| Phase | Work |
|---|---|
| `hand_features` | Stacking/transfer, fill/color counts, CPU connected components |
| `text_encode_or_cache` | Tokenization, cache lookup or the original two text encodes |
| `render` | Both background renders and image concatenation |
| `image_encode` | The original full 2N image batch per model |
| `pca_linear_head` | Similarities, hand features, PCA and linear score |
| `score_readback` | Completed score transfer to CPU |

These profiles are outside timed trials; their synchronizations perturb
execution. Do not sum phase medians or use the instrumented call as a throughput
measurement. If image encoding dominates, this cache cannot remove that work.
Changing image batch size, merging text batches, dropping candidates, or changing
precision needs a separate exact GPU parity experiment.

After exact GPU parity and a useful real repeated-prompt hit rate are established,
the operator can start a worker with `PIXEL_REWARD_TEXT_CACHE=64`. This PR changes
no production config. A one-off new prompt still pays the original text encodes
plus lookup overhead; it has no promised speedup.

## Distinguish GPU generation from chat latency

The approximately **1.6 s setup figure on the 3060 Ti** has unverified timing
boundaries. The operator clarified that it is not established as worker-only or
browser latency. Do not compare it with **628 ms offline generation on the
3080 Ti** or infer a network gap from their difference. First run the same
complete worker benchmark and this reward breakdown on the 3060 Ti. Then
inspect the existing `PIXEL_PROFILE_PHASES=1` events for time from
job receipt to result acknowledgement and result-post time. Long-poll waiting
is not user queue latency. Until the same-GPU generation time and app/client
timestamps are measured, slower kernels, reward models, queueing, payload and
polling remain hypotheses; none is established as the cause.

## CPU verification

```bash
"$PY" -m unittest discover -s worker -p 'test_reward_cache.py'
```

Tests use a small actual Torch encoder and real `FastRewardV7` scoring to check
exact scores with cold/warm caches, unchanged image-encode calls, weight/token/
model/math invalidation, LRU eviction, eval-mode enforcement, exclusive scorer
phases, completed benchmark budgets, mismatch/fallback rejection and safe error
reports. These tests provide no GPU performance or real-model parity evidence.
