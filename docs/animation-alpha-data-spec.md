# Proposed animation and alpha collection records

Status: collection records are implemented by the
[pack importer and integrity helpers](../data-pipeline/animation/README-ingest.md).
The animation model and LoRA training remain future work. Examples and collection
counts in the proposal are illustrative targets, not collected data.

## Files and identities

Each collection occupies a new directory under `$DATA_ROOT`. The immutable
source archive and any retained license metadata are separate from normalized working files.
Only relative paths appear in exported records. Secrets, operator identities,
hostnames and private absolute paths have no place in these records.

| Artifact | Contents |
|---|---|
| `sources.jsonl` | Required source URLs and asset hashes, optional license metadata, generation or transformation provenance |
| `sequences.jsonl` | One record per ordered animation or one-frame alpha example |
| `frames/<sequence_id>/` | Full-canvas RGBA PNG masters with explicit ordered frame references |
| `reviews.jsonl` | Verdicts, defect dimensions and repair lineage |
| `preferences.jsonl` | Sequence A/B choices, ties and unusable-both judgments |
| `manifest.json` | Schema version, input/file hashes, collection policy, split assignments, holdout byte hashes and aggregate counts |

`sequence_id` identifies a specific sequence/version. `lineage_group` groups
the original asset and all related actions, directions, recolors, edits,
procedural derivatives and generated variants. `source_id` identifies the
immutable source record. Content hashes supplement IDs; filenames alone do
not establish identity.

## Sequence fields

| Field | Meaning |
|---|---|
| `schema_version` | Proposed value `pixie-animation-alpha-v1` |
| `sequence_id`, `source_id`, `lineage_group` | Stable collection identifiers |
| `subject`, `caption`, `action`, `direction` | Subject identity and sequence-level intent; unknown values are explicit |
| `loop` | Whether the final-to-first transition is intended |
| `canvas_px` | Shared full-frame width and height; normalized working target is 32x32 |
| `anchor_px` | Shared reference point, with top-left origin, x right and y down; unknown anchors remain null |
| `alpha_mode` | `binary` or `partial`, derived from actual channel values |
| `alpha_association` | `straight` in normalized masters; original convention retained in source metadata |
| `color_space`, `blend_mode` | Declared color convention and `alpha_over` or separately handled `additive` rendering |
| `frames` | Ordered relative PNG paths, decoded RGBA hashes, file hashes and positive integer `duration_ms` |
| `timing_origin` | `source_metadata`, `procedural` or `annotated`; inferred timing is never called source metadata |
| `normalization` | Original size, common crop/scale/offset transform and tool/version |
| `palette_derivation` | Null for original masters, or shared mapping/version for a derived palette target |
| `review_status` | `unreviewed`, `accepted`, `repair` or `rejected` |
| `split` | `unassigned`, `train`, `validation` or `test`, assigned by lineage group |

A one-frame alpha example uses the same record, `action=static`, `loop=false`
and a documented display duration with `timing_origin=annotated`. Its duration
has no meaning as motion supervision. Original-size masters remain available
when normalization would lose pixel detail. Arbitrary resampling is not
evidence that a source was native 32x32 pixel art.

## Source records

An imported source records the original archive hash, asset/license URLs,
license identifier/text hash and exact selected frames/tags. A Krea source
records the sheet hash, complete prompt and seed semantics, model/LoRA hashes,
settings, cell order and extraction version. A seed without its batch context
does not establish individual-frame replay.

A procedural source records parent sequence/sprite IDs, transformation/render
code version, parameters, seed and original alpha/blend conventions. The
source record distinguishes deliberate geometry movement from incidental
camera movement. A repair records its parent and edit operation without
replacing the original files.

For the [proposed Krea sprite-sheet LoRA](animation-alpha-data-plan.md#proposed-krea-sprite-sheet-lora),
a derived training-sheet record also references the parent sequence/version,
ordered source frame indices, frame durations, grid rows and columns, cell
size, common canvas/anchor transform, background composite, sheet hash,
caption/trigger version and assembly tool version. The caption's layout,
action and frame count must agree with the sheet and sequence metadata.
Sheet derivatives inherit their parent's lineage and split. Record whether
each example is real, procedural, generated or repaired, including all parents
for mixed provenance. Generated examples also retain the producing model and
LoRA identifiers and the whole-sequence review that admitted them.

## Integrity and split requirements

- Frame references have explicit order and reconstruct the original frame
  durations, including intentional repeated poses. Deduplication must not
  silently remove a held pose or change playback duration.
- Every sequence uses a common canvas and anchor convention. Trimmed sheet
  cells require offsets; packed sheet coordinates are not frame positions.
- Alpha classifications agree with actual values. Partial-alpha masters retain
  values strictly between 0 and 255. Additive examples stay a separate task.
- Visible-image hashes ignore hidden RGB only where alpha is zero. Original
  file hashes remain byte-exact. Sequence dedup also includes order and timing.
- Every derivative, sibling animation, recolor and near-duplicate lineage stays
  in one split. Random per-frame splitting is invalid. Cross-source duplicates
  join a shared lineage group before allocation.
- Blind-50 and Eval-50 are excluded before datagen, admission and augmentation.
  Use the repository's punctuation-normalized word-boundary subject/caption
  policy, including frame captions and any eventual trimmed model text. Keep
  original holdout bytes and record their hashes; never rewrite the lists.
- New animation evaluation subjects are frozen separately and excluded from
  positive training and reward-model training examples. Preference train/eval
  assignment follows the same lineage groups as generation data.

## Review and preference records

Reviews reference a sequence/version and distinguish individual frame defects
from sequence-level defects. Fields cover identity drift, motion correctness,
unintended flicker, jitter, crop, alpha halos and loop quality. A dimension can
be `pass`, `fail`, `uncertain` or `not_applicable`. Loop quality is not applicable
to a non-looping action. The whole-record verdict is distinct from these fields.

A preference record references the request, both sequence versions, their common
preview conditions, selected dimension and `a`, `b`, `tie` or `both_unusable`.
Pair order is randomized and recorded. Anonymous review-session IDs can support
repeatability without recording a person's identity. Automated scores retain
the scorer/version and are distinguishable from human labels.

Reported acceptance and preference statistics name their denominator, source
mix, review completeness and sampling design. Ranked defect packets support
repair; a separate random review sample supports population quality estimates.
Counts of augmented frames are not counts of independent motions or subjects.
