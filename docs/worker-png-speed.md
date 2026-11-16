# Compare cheaper lossless PNG encoding

`PIXEL_PNG_MODE=fast` uses Pillow compression level 3 without PNG optimization.
The default `legacy` mode preserves existing encoded bytes. Both modes preserve
the exact resized RGBA pixels, output dimensions, rankings and candidate counts.
Pillow documents that `optimize=True` overrides the compression level and uses
level 9: [PNG save options](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#png).

This is an opt-in experiment. Faster compression usually produces larger PNGs;
the benchmark reports encoded JSON payload bytes beside time. Enabling it on a
worker requires an operator decision after measuring the actual transport cost.
This PR does not change production config, images, scores or rendering scale.

## Run the comparisons

Encoding-only CPU comparison, five alternating trials of 20 PNGs at 256 square:

```bash
"$PY" worker/benchmark_png.py --sprites "$SAMPLE_DIR/final" \
  --count 20 --scale 16 --trials 5 --out "$RUNS/png-cpu-new"
```

It loads local sprites recursively and resizes them to native 16 square before
timing. That makes the encoder workload explicit; it does not measure generated
v3 sprite quality or GPU speed. Every decoded output is checked for exact pixels.

Complete offline worker comparison on the 3060 Ti or 3080 Ti, with actual assets:

```bash
PIXEL_CACHE_BLANK=1 "$PY" worker/benchmark_png.py \
  --worker-config "$WORKER_CONFIG" --device cuda:0 --trials 5 \
  --out "$RUNS/png-worker-new"
```

This command compares the real worker at 64 candidates, show 4 / store 20, with
identical complete-batch seeds. It requires an idle explicitly selected GPU,
the configured reward scorer, exact decoded pixels and unchanged metadata.
Warmup, asset loading, parity decoding and JSON encoding are outside timing.
Generation, reward, resizing, PNG and base64 are inside. It makes no app requests
and excludes HTTP, queue wait and actual network transfer. A faster offline
worker does not prove a faster browser result. JSON payload size counts the
comparison signature, including duplicated shown/stored images, rather than
claiming to be an exact on-the-wire result body. Compare payload differences and
operator-observed result-post time before enabling the mode.

## Evidence and limits

The operator measured **78 ms PNG time** and **628 ms total worker time** on
the 3080 Ti. Eliminating all PNG work would cap the calculated latency saving
at roughly **12.4%**, or throughput improvement at roughly **14.2%**. These
are scale checks using separate medians, not exact measured phase shares or a
prediction for level 3 compression.

**Measured locally**, five alternating CPU trials on 20 checked-in sample sprites
resized to 16 square and rendered at 256 square, Pillow 13.0.0:

| Metric | Legacy | Fast |
|---|---|---|
| Encoding batch median | 45.39 ms | 12.64 ms |
| Encoding batch range | 44.40–48.00 ms | 12.41–14.02 ms |
| JSON signature bytes | 40,401 | 49,369 |
| Decoded pixel equality | Reference | All outputs exact |

The run data is in [png-cpu-sample20.json](evidence/png-cpu-sample20.json).
Encoding was lower in this microbenchmark, while payload grew by **8,968
bytes**, about **22.2%**, calculated. An earlier level-1 comparison measured
48.87 to 14.79 ms but produced 110,421 payload bytes. A local compression-level
probe identified level 3 as a better size/time tradeoff, so the final five-pair
comparison above uses level 3. CPU load was observed at 37% before
the initial invocation, other work was active throughout, and no controlled idle-host or
full-worker result was measured here. The initial invocation failed to find
nested sprite paths, reported invalid and was excluded; the fixed recursive
input run completed all five pairs. This is not an end-to-end speed claim or
evidence about either RTX GPU.

At the measured median difference of roughly 33 ms, the added bytes alone need
about **2.2 Mbit/s** of effective transfer bandwidth to avoid erasing that
saving, calculated as `8968 * 8 / 0.0327545`. Actual requests contain different
images, protocol overhead and response storage; this is only a scale check.

Tests verify legacy byte identity, fast decoded pixels including transparency,
metadata drift rejection, alternating seeds, CLI output and overwrite refusal:

```bash
"$PY" -m unittest discover -s worker -p 'test_*.py'
```

No GPU or browser latency was measured here. The operator clarified that the
earlier roughly 1.6-second 3060 Ti setup figure has **unverified timing
boundaries**. It is not established as worker-only or browser latency and must
not be compared with the 628 ms offline 3080 Ti result. Run the same full-worker
benchmark on both cards with identical assets/settings, then use
`WORKER_PROFILE` separately for transport observations. There is currently no
measured 3060 Ti phase breakdown or established latency gap to explain.
