# Measure complete worker latency

This change adds opt-in phase timing and an offline benchmark of the real
worker generation path. It changes no sampler, candidate budget, scorer,
precision or PNG settings. It depends on the versioned batch-seed contract in
PR #20. The previous 293.1 to 285.6 ms result was **measured on the 3080 Ti**
for CLIP plus sampling. It excludes reward, PNG and HTTP and does not explain
the approximately 1.6-second **operator-reported 3060 Ti** worker latency.

Run this benchmark on an idle GPU with an explicit local worker config:

```bash
PIXEL_CACHE_BLANK=1 "$PY" worker/benchmark_worker.py --config "$WORKER_CONFIG" --device cuda:0 \
  --batch 64 --show 4 --candidates 20 --warmup 5 --trials 5 \
  --out "$RUNS/worker-phases-new"
```

Use the 3060 Ti to investigate the actual worker, or the 3080 Ti for a separate
hardware baseline. The script selects exactly one physical GPU, refuses another
compute process, loads local assets from the supplied config and makes no app
requests. It needs no worker key and acquires no worker lock. Choose a new
output directory. Paths, URLs, prompts and startup logs are not copied into
`worker-benchmark.json`. Errors produce an invalid result with an error class.
The config must point at the actual intended reward assets: scorer fallback
invalidates the benchmark rather than measuring the cheaper fallback.

The benchmark alternates clean and instrumented calls for five trials, with the
same full batch seed, candidate count and prompt on each side. Exact shown PNGs,
stored PNGs, rankings, scores, indices and sampling provenance must match.
`comparison_valid` must be true. Warmup calls are excluded; the blank cache,
if enabled, is warm. Profiling is opt-in and synchronizes CUDA phase boundaries,
so compare clean full-generation latency with instrumented full-generation
latency before interpreting attribution. Do not sum phase medians or treat
instrumented latency as an optimization.

The exclusive generation phases are:

| Phase | Work |
|---|---|
| `prepare` | Seed/provenance, tokenization, seed and condition tensors |
| `clip` | Both text branches, including the exact blank cache when enabled |
| `sample` | Full CFG sampler with existing KV cache and optional CUDA graphs |
| `readback_rgba` | GPU-to-CPU transfer and palette/RGBA construction |
| `score_rank` | Feature/reward scoring for all candidates and sorting |
| `png` | Resize, PNG compression and base64 for displayed/stored candidates |

The offline total excludes polling, queue wait, JSON serialization and HTTP.
For transport observations, the operator can set `PIXEL_PROFILE_PHASES=1` when
starting a worker. This does not change any checked-in production config. Each
completed request adds a numeric `WORKER_PROFILE` log event containing generation
phases, the next-request duration, result serialization/HTTP/retry duration,
attempt count, acknowledgement state and time from job receipt to result ack.
The next-request duration includes long-poll waiting; it is not a measurement of
user queue latency. No new prompt, URL, job ID, key or host is added to the event.
When disabled, the phase timer reads no clocks and performs no CUDA synchronizations.
Queue-mode authentication is still required before its lock/poll loop starts.

## Decide what to change after measuring

These are **untested hypotheses**, not speed forecasts:

1. If reward dominates, split the existing scorer into CPU feature preparation,
   GPU embeddings and head evaluation before changing its batch size. Fewer
   candidates changes best-of-N quality; compare quality at the same total budget
   and preserve exact features/scores for reuse experiments.
2. If PNG dominates, separately benchmark compression level and return payload
   size. Pixel equality is required, and a larger payload can undo encode savings
   over HTTP. Keep displayed and stored candidate counts fixed for attribution.
3. If sampling dominates, profile trunk versus slot-head/RNG. KV caching and
   trunk CUDA graphs already exist. A new graph strategy or BF16/FP32 change
   must pass exact-match checks before enabling it on the live path.
4. If transport dominates, measure payload bytes and actual retry causes on the
   operator's worker. The offline benchmark cannot establish network savings.
5. Keep the accepted exact full-batch blank cache. Its **measured 7.5 ms saving**
   on the 3080 Ti is useful but supplies no measured 3060 Ti total-worker gain.

No GPU timings were measured in this workspace. CPU behavior tests exercise
the actual `Gen.run` method, unchanged outputs with profiling on/off, exclusive
phase accounting, disabled instrumentation, alternating paired seeds/budgets,
scorer fallback rejection, invalid output parity, and key-free offline import.

```bash
"$PY" -m unittest discover -s worker -p 'test_*.py'
```
