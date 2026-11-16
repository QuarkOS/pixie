# Animation consistency review

`score.py` reads a normalized collection and writes a fresh review folder. It
emits independent, uncalibrated metrics. It does not accept or reject training
examples and has no composite quality score.

```powershell
$PY -m animation.score --input $DATA_ROOT/animation/normalized --output $RUNS/animation-review-<new-id>
```

Run the command from `data-pipeline` so `animation` is importable. The output
contains `score.jsonl`, `previews/<sequence_id>.contact.png`, and GIF playback
previews. Intended loops repeat indefinitely. Non-looping actions play once,
without a GIF loop extension. The output `manifest.json` hashes every generated
file and binds the source collection manifest hash. The input masters are never
rewritten. An output nested inside the input collection is refused.

Identity drift is unavailable unless you provide one of these local-only
options:

- `--features features.npz` loads `features`, `sequence_ids`, `frame_paths`,
  and `frame_sha256` arrays. Every sidecar row must match a collection frame
  path and its recorded file hash. This is the preferred repeatable interface.
- `--vision-model <local-model-directory>` loads a local Transformers image
  model with `local_files_only=True`. The model runs on `--device cpu` by
  default. No model is downloaded. Frames are composited over fixed neutral
  gray before embedding, so hidden RGB under transparent pixels cannot change
  identity scores. The output records a hash of the local model file inventory.

The options are mutually exclusive. A precomputed sidecar's own hash identifies
the supplied feature artifact. Its underlying model identity is not inferred.
Generate sidecars with your chosen local image encoder and retain that model's
provenance alongside them. Feature fixtures exercise the interface without
claiming human-calibrated semantic accuracy.

`features` is a finite, nonzero matrix with one row per frame. The identity
columns are one-dimensional NumPy Unicode arrays. Rows bind the sequence ID,
relative frame path and byte hash. Every input frame must appear exactly once.

The reported identity values are cosine similarities between adjacent frames
and from the first frame. Geometry reports bounding boxes, anchor offsets,
center span, and a jitter proxy. Jitter is the RMS residual displacement from
the median adjacent step. It includes intended deformation and cannot track a
ground-contact landmark. Palette metrics ignore RGB under transparent pixels
and never refit a palette. Motion and loop comparisons also ignore hidden RGB
only where alpha is zero. Loop metrics apply only to intended loops;
they compare endpoints and do not judge apparent velocity. Static repetition
is reported separately. Human labels must calibrate all thresholds before use.

The feature sidecar schema is deliberately hash-bound. Do not reuse features
after changing, resizing, or re-encoding a frame. The scorer rejects missing,
duplicate, extra, escaped, zero, or non-finite feature rows.

The contact sheet composites frames over checkerboard, light and dark
backgrounds. It preserves the canvas aspect ratio and uses nearest-neighbor
display scaling. GIF
durations are quantized down to the format's 10 ms units. The output records
source durations, requested quantized durations and durations read back from
the actual encoded GIF. Identical frames can merge during GIF encoding, while
their combined display time is retained. GIF palettes and alpha cannot preserve
all RGBA detail. The preview is a review artifact rather than a training target.

The approximately 500 curated sequences are a collection goal, not a measured
yield or a scorer acceptance threshold. No automatic admission decision is
implemented. Compare the independent metrics with human defect labels before
choosing thresholds.

## Compare palette metrics at the same resolution

Metrics use the stored collection frames, before the contact-sheet or GIF
display resize. They cannot recover colors discarded by an earlier ingest
downscale. `measurement_space` records the measured canvas, normalization
recipe, original canvas when available, and whether ingest resampled it.

For a native-size pilot diagnostic, split a 512x512 2x2 sheet into its 256x256
cells and ingest with `--canvas 256 256 --key-rgb 255 0 255`, explicit cell
order and annotated durations. Score that collection. Keep it as a diagnostic
collection, separate from the normalized 32x32 training targets. The sheet
builder deliberately does not accept a 256x256 logical training canvas.

Alternatively, compare both baseline and candidate after the exact same
32x32 normalization, key and crop policy. Label those as normalized-target
metrics. Native-cell palette sizes and normalized-target palette sizes are
not comparable: resampling can discard rare colors, remove thin details and
change palette Jaccard. Exact magenta keying also does not remove generated
near-magenta fringes. Review these artifacts rather than interpreting a lower
palette count as improved consistency. Existing saved scores do not become
native-size measurements after this update.

From the repository root, run the CPU fixtures:

```sh
$PY -m unittest discover -s data-pipeline/animation/tests -t data-pipeline -p test_score.py
```
