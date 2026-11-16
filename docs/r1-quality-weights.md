# Downweight reviewed simple tiles in a fresh cache

The operator's review of 400 sprites was approximately **87% good, 10%
borderline and 3% junk**. Those are supplied approximate human judgments, not
exact counts or a calibrated simple-tile detector. Plain squares and mats can
be valid game assets. This tool flags visual simplicity for review and proposes
a sampling multiplier; it does not classify every rectangle as junk.

```bash
"$PY" -m train.pixie32.quality_weights \
  --data-root "$DATA_ROOT" --r1 "$DATA_ROOT/r1_list.txt" \
  --blind50 holdout/blind-50.json --eval50 holdout/eval-50.json \
  --expect-rows 133688 --simple-weight 0.25 --out "$RUNS/r1-simple-review-new"
```

The conservative signal requires a foreground bounding box of at least 144
pixels, both sides at least 8 pixels, and at least 98% rectangular coverage.
After ignoring a two-pixel outline border, at least 92% of visible interior
pixels must share the same exact RGB and at most 10% of adjacent pairs can
change RGB. These are **untested heuristic thresholds**. Detailed/textured
rectangles are deliberately retained. Inspect `simple-tiles-*.png` and the
corresponding indices/full captions in `review.json`. If no rows are flagged,
there is no contact sheet. None of the holdout matches enter the review packet.

`sample_weights.json` contains every admitted R1 row's image/caption hashes,
ID, relative file name, measured signal and proposed multiplier. Only flagged
rows receive 0.25; others receive 1.0. The 0.25 value is an **untested policy
choice**, not a measured optimum. Review false positives before using it. The
summary reports counts and criteria, not a precision/recall or junk-rate claim.
No input file or active run changes when this command runs.

Use the reviewed file only when building a **new** dataset cache:

```bash
"$PY" -m train.pixie32.build_data_p32 \
  --data-root "$DATA_ROOT" --r1 "$DATA_ROOT/r1_list.txt" \
  --palette "$PALETTE" --tokenizer "$TOKENIZER" \
  --blind50 holdout/blind-50.json --eval50 holdout/eval-50.json \
  --sample-weights "$RUNS/r1-simple-review-new/sample_weights.json" \
  --out "$DATA_ROOT/p32-quality-cache-new"
```

Retain the existing build's seed, edge-list and split flags when comparing
caches. Quality multipliers do not affect admission or split assignment; they
multiply the existing balance/edge sampling weights and are normalized with
the train weights. Eval/probe still use their unchanged examples. The optional
file is bound to the original metadata, provenance, R1 and holdout bytes and
each image/caption. Changed inputs, duplicate IDs, unrecognized IDs or invalid
multipliers fail. Its hash is included in the new cache manifest/fingerprint.
An existing run cannot silently resume against the changed dataset.

The narrow geometric check is not a semantic caption-mismatch check. Do not
infer that an unflagged image matches its caption or is usable. Continue the
uniform human audit and its `caption_match` labels; automated semantic ranking
requires a separate calibrated check.

Measured locally: **21 CPU tests passed** (five quality tests plus 16 data
tests). They check plain/outlined squares, textured rectangles and irregular
icons; actual multiplier consumption without split changes; hash mismatch
rejection; fresh-output refusal and unchanged inputs. The actual checked-in
sample200 produced **0 flags out of 200**, and both holdouts were unchanged.
The full 133,688 set, detector quality and training effect were not measured.

```bash
"$PY" -m unittest train.pixie32.test_quality_weights train.pixie32.test_data
```
