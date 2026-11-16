# Measure the combined R1 set before generating more

`train/pixie32/audit_r1.py` audits the existing combined layout without changing
images, keep-lists, extraction gates, or training. It needs only the project
venv's NumPy and Pillow. No GPU is used.

Run a complete scan and create a uniform annotation packet in a fresh folder:

```sh
"$PY" -m train.pixie32.audit_r1 prepare \
  --data-root "$DATA_ROOT" \
  --r1 "$DATA_ROOT/spotcheck/cand_R1_mag_outline<=2_and_no_tiny_islands.txt" \
  --blind50 "$DATA_ROOT/holdout/blind-50.json" \
  --eval50 "$DATA_ROOT/holdout/eval-50.json" \
  --expect-rows 133688 --sample 400 --seed 20261115 \
  --training-cache "$DATA_ROOT/p32-cache" \
  --out "$RUNS/r1-quality-01"
```

Retarget `--training-cache` to the existing prepared P32 cache. With this flag,
the usual cache integrity checks run, including the original holdout **byte**
hashes. Supply the original files; do not reserialize them. The tool also
reports the actual train split's sampling weights, edge probability, and
effective row count. Omit this optional flag when auditing raw R1 independently;
then the report records supplied holdout hashes without claiming they match a
training cache.

`audit.json` contains measured source, sheet, subject, and normalized-caption
coverage; exact visible-pixel duplicates; alpha integrity; and recomputed
edge-heavy counts. Duplicate comparison ignores RGB under fully transparent
pixels and preserves the alpha values. Near duplicates and semantic quality
need separate assessment. `coverage.csv` lists every subject's row count,
distinct-caption count, and source counts. This describes coverage of the
existing subject vocabulary; it cannot find subjects missing from that vocabulary.

The full-R1 weighting scenario applies the trainer's balance formula and 0.5
edge multiplier **before** probe/IID selection. It is explicitly separate from
the actual split-specific cache weights. Effective rows is the sampling
concentration statistic `(sum weights)^2 / sum(weights^2)`, not a count of
independent training examples.

Holdout matches use the trainer's punctuation-normalized, word-boundary policy.
Matches are excluded from the annotation population and counted. The expected
row-count guard makes any such exclusion or wrong keep-list fail the command
above. The tool never edits either holdout or any dataset input.

## Label the random packet

Open `contact-*.png`; use `sample.json` for each index's full caption and ID,
and `images/` for enlarged individual sprites. The packet is sampled uniformly
without replacement from sorted IDs. Source and edge metadata do not influence
selection. Complete every row of `labels.csv` with this rubric:

| Field | Values | Meaning |
|---|---|---|
| `junk` | `yes`, `no`, `borderline` | `yes`: unusable sprite (severe cropping, broken extraction, unrecognizable object, sheet contamination); `borderline`: usable only after repair or ambiguous; `no`: usable sprite |
| `caption_match` | `yes`, `no`, `uncertain` | Does the visible sprite match the full caption, including requested attributes? |
| `defect` | optional text | Use consistent tags, for example `crop`, `magenta`, `islands`, `wrong-subject`, `wrong-attribute`, `duplicate-like` |

Review sprites before reading their source/edge metadata to reduce expectation
bias. Consider having a second reviewer label the same packet independently
before reconciling disagreements. Keep their label files separate.

```sh
"$PY" -m train.pixie32.audit_r1 summarize \
  --packet "$RUNS/r1-quality-01" \
  --labels "$RUNS/r1-quality-01/labels.csv" \
  --out "$RUNS/r1-quality-01-result.json"
```

The summary reports clear junk, junk plus borderline, caption mismatch, and
caption uncertainty separately, with labeled/missing denominators and 95%
Wilson intervals. It marks incomplete metrics as invalid for population
inference: missing labels may be systematically harder examples. Source and
edge subgroup intervals are also included; small subgroups will be uncertain.
Intervals describe sampling uncertainty, not reviewer error. They are
conservative binomial approximations for sampling without replacement.

**Design estimate:** a complete uniform packet of 400 gives a worst-case
95% half-width of about 4.9 percentage points. This is a sample-size calculation,
not a measured junk rate. Use the actual returned intervals when deciding
whether an apparent edge/source difference supports changing a filter or weight.
Do not loosen extraction gates on the basis of an unlabeled audit.

## Local verification

```sh
"$PY" -m unittest train.pixie32.test_audit_r1
"$PY" -m train.pixie32.audit_r1 prepare \
  --data-root "$SAMPLE_DIR" \
  --r1 "$SAMPLE_DIR/r1_list.txt" \
  --blind50 "$PIXIE_HOLDOUT_DIR/blind-50.json" --eval50 "$PIXIE_HOLDOUT_DIR/eval-50.json" \
  --expect-rows 200 --sample 40 --seed 20261115 \
  --out "$RUNS/r1-sample200-check"
```

**Measured on the checked-in 200-sprite sample:** 19 / 133 / 48 rows from
3060 Ti / 3080 Ti / 5070 Ti sources, 198 subjects, 200 normalized captions,
28 edge-heavy rows, zero exact visible-pixel duplicate excess, zero empty
images, and zero nonbinary-alpha images. A 40-row packet and contact sheet
were generated and visually inspected. These are sample integrity/coverage
counts, not measurements of junk in the full 133,688-row R1 set. No human
junk labels or full-corpus access were available for this PR.
