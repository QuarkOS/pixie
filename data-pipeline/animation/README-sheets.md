# Build Krea animation LoRA sheets

Build from reviewed sequences in new animation collections. The first-round
goal is approximately 500 curated four-frame sequences, not a measured yield.
The default is a 2x2 sheet with fixed-size cells and a flat magenta background.
The builder does not train or change the P32 corpus.

Run commands from `data-pipeline` with the project venv interpreter in `$PY`:

```sh
$PY -m animation.sheets --input "$DATA_ROOT/animation-real" --reviews "$DATA_ROOT/animation-reviews.json" --output "$DATA_ROOT/animation-sheets-2x2" --grid 2 --seed 42
```

Use the manifest-bound review format in `README-ingest.md`. Only accepted
sequences enter a sheet. Default source kind is artist-made. To mix reviewed
derivatives later, repeat `--input` and `--reviews`, then explicitly enable
`--include-procedural`, `--include-generated`, or both. The manifest retains
each source kind rather than treating derivatives as artist-made motion.

Each sequence must contain exactly the requested frame count. The builder
does not select, truncate, pad or repeat frames. Curate longer source actions
with explicit ingest order/timing and review the selected sequence first.
Frames must share the selected logical canvas and a known common anchor.
They enlarge by one fixed integer nearest-neighbor factor. No frame is recentered and no
palette is refitted. The initial format uses binary-alpha cutouts with ordinary
alpha compositing. Partial-alpha and additive examples are excluded from this
opaque-sprite LoRA dataset while their original collection remains intact.

The default logical canvas is 32x32. For larger source art, ingest with
`--canvas 64 64` and build a separate dataset with `--logical-px 64`. A cell
still defaults to 256 screen pixels, so 64x64 art enlarges by four rather than
eight. `--cell-px` must be a multiple of the selected logical size. Mixing
32x32 and 64x64 inputs in one build is refused through canvas exclusions.
This option preserves more detail but does not establish compatibility with
the current 32x32 LoRA or P32. Measure that separate recipe before adopting it.

Grid options and default output sizes are:

| Option | Frames in chronological row-major order | Default sheet size |
|---|---|---|
| `--grid 2` | 4 | 512x512 |
| `--grid 3` | 9 | 768x768 |
| `--grid 4` | 16 | 1024x1024 |

These are supported formats, not measured quality, memory-fit or generation
claims. The [proposal](../../docs/animation-alpha-data-plan.md) defines the
untested quality targets for progressing to longer layouts. `--cell-px` must
be a multiple of the selected logical size. CPU fixtures use smaller cells
to keep checks short.

Each PNG has a matching text caption, including subject, action, direction,
frame count, layout, loop intent and reading order. An illustrative caption
begins `pixie_anim, a copper beetle, walk, right, 4-frame animation sheet`.
The manifest retains source URLs and hashes, parent collection hash, frame
order/durations, common anchor, provenance, review-file hashes and exclusion
reasons. Original RGBA masters stay in the input collection.

Training sheets and captions are under `train/`. Evaluation sheets are under
`eval/`. Point a future training config only at `train/`, with image bucketing
and random cropping disabled so the grid survives. This PR does not select
LoRA rank, learning rate, training steps or a model checkpoint.

## Keep related sequences in the same split

The builder groups common lineage IDs, normalized subjects and identical visible
frames across collections. Empty transparent frames do not join unrelated
subjects. Exact duplicate sequences, including durations, emit a single sheet.
Hidden RGB under zero alpha does not create a distinct visible example.
Source archive membership alone does not join unrelated characters in a pack.
Curators must assign shared lineage IDs to recolors and near-duplicates that
are not pixel-identical; this is not a semantic deduplicator.

The default evaluation fraction is 0.1, selected by a stable hash of the split
seed and connected lineage group. Existing validation groups stay in eval.
Existing test/audit-only examples are excluded entirely. Blind-50 and Eval-50
matches are excluded using the existing normalized subject/caption policy.
Original holdout bytes and hashes remain unchanged.

Use `--eval-subjects` with a frozen JSON subject array to force those subject
groups into eval and retain its file hash. Use the same file for a later LoRA
comparison. The hash split also evaluates other groups unless
`--eval-fraction 0` is supplied. Split membership is frozen in each output
manifest. Adding data that joins formerly distinct groups requires another
split audit. A positive evaluation fraction with no evaluable group fails
rather than silently delivering an empty evaluation set.

From the repository root, run CPU fixtures:

```sh
$PY -m unittest discover -s data-pipeline/animation/tests -t data-pipeline -p test_sheets.py
```

Training composition is capped by default: procedural and generated sheets may
make up at most 50% of training records (`--max-procedural-fraction 0.5`). The
cap is deterministic by seed and emits `procedural_training_cap` exclusions.
Artist records are provenance kinds `artist` or `real`; generated and
procedural records are the capped group. Evaluation records are not consumed
by this training cap. Use a stricter fraction when the run needs more artist
motion, and retain that value in the manifest.

For a 64x64 build with a nonzero evaluation fraction, at least two evaluation
lineage groups are required by default. Pass `--eval-min-lineages` explicitly
when a different review design is justified. A single-lineage evaluation is
too fragile for the larger-canvas corpus.
