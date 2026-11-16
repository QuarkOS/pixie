# Animation and transparency data proposal

Status: proposed future collection, not part of the approved current P32 run.
This document authorizes no generation, asset acquisition, training or deployment.
No dataset yield, animation quality, GPU throughput or training improvement has
been measured for this proposal. Counts below are proposed pilot budgets.

## Start with coherent sequences and original alpha

Collect two related datasets: ordered animation sequences and sprites with
true partial alpha. Some examples should belong to both. Preserve original
masters so future models and reward models can use the same evidence.

The [combined dataset](dataset-krea2-combined.md) already contains binary RGBA
transparency. The current [P32 plan](pixie32-plan.md) maps alpha below 128 to
palette index 0 and treats the remaining pixels as opaque. That representation
cannot retain smoke, translucent glass or soft shadows with intermediate alpha.

Animation frames and external asset packs are outside the current P32 recipe.
Keep proposed collections in new folders under `$DATA_ROOT`, with reports under
`$RUNS`. External assets stay in a separate research collection until their use
in a future training corpus is explicitly approved. The existing synthetic-only
P32 policy, R1 list, extractor gates and frozen holdouts remain unchanged.

## Animation sources in collection order

| Source | Useful examples | Limitation and retained provenance |
|---|---|---|
| Original animated asset packs | Artist-authored idle, walk, attack and effect sequences with original alpha | Style and resolution vary. Preserve source archive, content hashes, asset/license URLs and license text. Prefer CC0 sources for initial research. |
| Procedural derivatives of accepted sprites | Bob, blink, pulse, recoil, spin and fade with known timing | Transformations provide limited motion. Record parent ID, algorithm/version, parameters and seed. Keep them distinguishable from artist-authored motion. |
| Krea sheets | Consecutive poses in the existing visual style | Current LoRAs were trained for sprites, not demonstrated temporal consistency. Retain complete sheet, prompt, model/LoRA hashes, generation settings and sequence-level verdict. |
| Human repairs and preferences | Corrected timing, stable identities, repaired loops, paired alternatives | Record parent/candidate lineage, edits and defect labels. Retain rejected alternatives for reward-model research, outside positive training examples. |

[Kenney's support page](https://kenney.nl/support) describes its asset-page
downloads as CC0; [Mini Dungeon](https://kenney.nl/assets/mini-dungeon) is an
example pack advertised with animation. Verify and retain each downloaded
pack's actual license. Source selection here is a research proposal, not a
claim that a pack matches Pixie's style or supplies native 32x32 frames.

Use original PNG frames or sprite sheets with metadata. Aseprite supports
[PNG-sheet and JSON export](https://aseprite.org/docs/cli/). Preserve source
frame durations, frame tags and offsets. Reconstruct trimmed frames on their
original canvas before normalization. Do not infer timing from sheet order
when the source provides explicit animation metadata.

## Test Krea sequences before scaling generation

The first synthetic experiment uses the existing 2x2 sheet layout as four
ordered animation frames, read left-to-right then top-to-bottom. A proposed
prompt template asks for the same subject, fixed camera/scale, one action,
four consecutive poses and a magenta background. Subject and action values
must pass the holdout exclusion policy before generation.

This is an untested prompt/layout experiment. Four related pictures do not
establish an animation. Review playback at intended speed and inspect each
frame. Record whether identity, pose progression and the closing transition
are usable. Generated sheets have no intrinsic frame durations; any assigned
duration is an explicit annotation, not recovered ground truth.

Use one canvas and anchor for the whole sequence. Any crop, scale or alignment
transform applies consistently across frames. Independent centering creates
jitter. Ground contact, intended translation and intentional squash/stretch
must remain visible rather than being normalized away. Use one palette mapping
across frames; never fit independent per-frame palettes.

The existing extractor can inform opaque frame extraction, but its independent
cell verdicts cannot admit a sequence. A damaged or missing required frame
rejects the sequence from positive animation training. Smoke/effects with
multiple components need separate, measured extraction criteria; do not relax
the existing R1/extract_v2 rules to admit them.

If Krea sequence acceptance is poor, prioritize original sequences and human
repairs. The dedicated LoRA proposal below starts from curated real animations.
Do not use a large unreviewed generated set to declare temporal consistency.

## Proposed Krea sprite-sheet LoRA

The goal is consistent animations with a target range of 4 to 16 frames per
sheet. Start with a target 2x2 layout containing four ordered frames. These
layout sizes, dataset counts, resolutions and acceptance thresholds are proposed
targets, not measurements. A sheet LoRA learns an image layout and motion poses;
it does not provide a temporal guarantee or generate playback timing by itself.

### Start with curated real sequences

Target 300 curated training sequences from licensed real animated packs, with
additional lineage-disjoint validation examples reserved before training.
This is a few-hundred-sequence starting target within the animation pilot,
not evidence that the amount is sufficient. Record actual train, validation
and audit counts separately. Favor complete, readable actions over many
recolors or windows from the same character. Preserve source licenses,
original frames, durations and all normalization decisions.

An optional later round can add manually reviewed Krea sequences and repairs.
Retain their parent model, LoRA, sheet, seed context and review provenance.
Compare a real-only baseline against the mixed corpus on the same frozen
evaluation set. Record the real/generated sampling mix and cap each lineage's
contribution. Admit generated sequences only after playback review of every
required frame; a frame-level aesthetic score cannot establish motion quality.
Never recycle evaluation outputs into either round's training corpus.

### Train on complete, grid-aligned sheets

Each training example pairs a complete sheet PNG with a matching text caption
and a sequence record from [the collection specification](animation-alpha-data-spec.md).
The proposed initial format is:

- Target 2x2 equal cells, with four chronological frames in row-major order.
  The order is top-left, top-right, bottom-left, bottom-right.
- Target 32x32 logical pixels per frame, nearest-neighbor enlarged to target
  256x256 cells in a target 512x512 sheet. Use the same enlargement for every
  frame. Keep the unmodified RGBA sequence masters separately.
- One shared canvas, camera, scale and anchor across the sequence. Preserve
  intended movement relative to that anchor. Restore trimmed source offsets
  before assembly. Do not crop or center frames independently.
- A flat magenta background for the initial opaque-sprite LoRA. No borders,
  frame numbers, text or checkerboards inside the sheet. Partial-alpha masters
  remain separate; magenta sheets do not supply true partial-alpha supervision.
- Chronological poses with real motion. Prefer complete short source actions.
  If selecting phases from a longer cycle, record selected indices and timing,
  then review the resulting playback. Do not pad with repeated poses merely to
  satisfy a frame-count target or label an incomplete action as looping.
- Captions with a stable trigger, subject, action, direction, layout, frame
  count, loop intent and reading order. Use consistent vocabulary across packs.
  Source timing stays in metadata; a sheet alone cannot encode variable durations.

An illustrative caption for the initial target layout is:

> pixie_anim, pixel art sprite sheet, 2x2 grid, 4 frames, a small armored beetle,
> walking right, looping, fixed camera and scale, frames in chronological order
> from left to right then top to bottom, magenta background

Apply augmentation to the sequence before sheet assembly. A flip changes the
direction caption and preserves frame order. Independent frame transforms,
random sheet crops and augmentations that permute grid cells invalidate the
format. Record the base model, LoRA configuration, caption version, data
manifest and assembly version. Training settings and memory fit require a
later bounded experiment; this proposal launches no run.

### Measure whole-sequence consistency on unseen subjects

Freeze an animation audit set with a target of 20 unseen subjects, varied
actions and a fixed seed list before comparing models. Keep checkpoint-selection
validation separate from that audit set. Subject, asset and derivative lineages
cannot cross training, validation or audit splits.

Exclude Blind-50 and Eval-50 from training, generation and the new animation
audit prompts using the existing normalized matching policy. Retain their
original bytes and hashes. The animation audit is an additional evaluation
set, not a replacement for either frozen holdout.

Compare the current static-sheet LoRA with the proposed animation LoRA using
matched prompts, generation settings, sample counts and recorded seed context.
Randomize model order for playback review. Report complete-sheet usability and
per-subject results, with denominators and uncertainty grouped by subject.
Do not report only the best selected sheet for each prompt.

| Measure | Review method |
|---|---|
| Identity drift | Check shape, palette, markings, equipment and anatomy across all frames. Distinguish intended deformation from a changed character. Embedding similarity can assist review but cannot replace it. |
| Jitter | Inspect unintended camera, scale or anchor displacement during playback. Measure annotated stable landmarks where applicable. Foreground-centroid changes alone are not failures because limb motion can move the centroid. |
| Motion correctness | Check action, direction and readable pose progression. Static copies fail an action that requires motion, even if they score well on identity and jitter. |
| Loop quality | Review the closing transition at documented playback timing. Compare its pose change and apparent velocity with internal transitions. Identical endpoint images alone do not establish a good loop. Non-looping actions are not scored as loops. |
| Sheet integrity | Check requested frame count, grid alignment, complete extraction, clipping and shared palette behavior. Reject a sequence with a missing required frame. |

Proposed promotion targets are at least 90% structurally valid sheets, at
least 85% passing identity and jitter review, and at least 80% usable whole
sequences with correct motion. For intended loops, target at least 80% passing
loop review. All percentages are untested acceptance targets. Count incomplete
or failed generations in whole-sequence denominators. Report loop results
separately with their applicable denominator. Review failures by subject and
action so aggregate success cannot hide a broken category.

### Increase frame count only after the shorter layout passes

Move to a target 3x3 layout with nine chronological frames only after the
four-frame model meets the proposed promotion targets on the frozen audit,
improves usable-animation rate over the static-sheet baseline, and has no
unresolved systematic identity, jitter or loop failure. Report uncertainty;
an inconclusive comparison calls for more review, not promotion.

Curate genuinely longer sequences for this stage. Do not repeat the shorter
sequence to fill the grid. Reuse the same caption, provenance and evaluation
rules with the new layout and frame count. A target 768x768 sheet preserves
the proposed cell scale. Resolution support, extraction correctness, VRAM and
throughput must be tested independently before committing to that target.

Move to a target 4x4 layout with sixteen chronological frames only after the
nine-frame stage passes those same quality gates without degrading its
shorter-layout control. A target 1024x1024 sheet preserves the cell scale.
Higher frame counts require enough distinct poses and a reviewed timing plan;
more cells are not evidence of better motion. Treat each expansion as a
separate experiment, with new output folders and measured quality, memory
and generation cost before adopting it.

## Gather partial alpha from ground truth

| Source | What to preserve | What the examples teach |
|---|---|---|
| Original RGBA assets | Unmodified image or sequence plus documented alpha/blend conventions | Artist-authored smoke, glass, water, ghosts, shadows and particles |
| Procedural renderers | Foreground color, alpha, render settings and deterministic seed | Controlled transparency, moving particles and opacity changes |
| Sprite derivatives | Original sprite and explicit opacity/layer/fade operation | Basic alpha behavior; not a substitute for realistic material structure |

Magenta keying supplies useful opaque cutouts. It cannot uniquely recover both
foreground color and alpha from one flattened translucent image. A rendered
checkerboard is also RGB content, not an alpha channel. Do not label either
source as ground-truth partial alpha.

Retain straight-alpha RGBA masters with alpha values 0 through 255, without
thresholding or background baking. Record any conversion from premultiplied
alpha, its color-space convention and the original master. Preserve RGB under
partially transparent pixels. RGB under zero alpha does not contribute visible
color and must not inflate visible-image dedup counts.

Preview every alpha example over light, dark and saturated backgrounds to
inspect halos, fringes, lost detail and unintended opacity. Keep those previews
as review artifacts rather than independent copies of the training target.
Distinguish ordinary alpha compositing from additive emission; an additive
glow cannot be treated as the same blend operation without an explicit model
and rendering decision.

## Pilot budget and acceptance evidence

Proposed targets are 500 accepted short animation sequences and 500 reviewed
partial-alpha examples. Animated alpha examples can count toward both targets;
report that overlap. These are collection targets, not required sample sizes,
expected yields or proof that a model will train successfully.

Begin with a review checkpoint after 50 animation candidates and 50 alpha
candidates before attempting the full pilot. Count rejected candidates too.
No duration, cost or GPU-hour budget is inferred from the current static-sheet
benchmarks. Use the existing local machines only when idle; no paid cloud GPU.

Coverage should include character motion and effects, multiple subject shapes,
directions, action types and alpha behaviors. Report source and action coverage,
not just total frames. A long loop or many recolors must not dominate the
sampling budget. Keep binary-alpha examples as controls alongside partial-alpha
targets. Tiny palette or alpha differences do not create independent examples.

For each review checkpoint, report:

- Candidate/accepted/rejected sequence counts, distinct lineage groups, actions
  and frame counts; explicit denominators and reviewed/unreviewed status.
- Identity drift, unintended flicker, jitter, crop, motion correctness and loop
  transition verdicts. Non-looping actions have no loop-closure requirement.
- Alpha defects across backgrounds, partial-alpha pixel coverage and blend-mode
  coverage. A high partial-alpha fraction alone does not prove a useful target.
- Source/license completeness, original-byte holdout checks, duplicate groups
  and lineage-disjoint train/eval allocation.

No universal numerical quality threshold is established. Defect heuristics
rank examples for review; evaluate false rejects on a random labeled sample
before turning a heuristic into an admission rule.

## Reward-model data and future training decisions

Store whole-sequence A/B preferences with identical subject/action requests.
Collect separate labels for appearance, motion correctness, identity stability,
loop quality and alpha cleanliness. Allow ties and unusable-both judgments.
Frame-only scores can miss broken timing or a poor closing transition.

Keep edits, rejected generations and preview-background variants linked to
their original sequence. A rejection is useful preference evidence but is not
automatically a negative for every quality dimension. The proposed record
format is in [the collection specification](animation-alpha-data-spec.md).

The current static model is not an animation model. A later design must choose
temporal conditioning, frame order, timing representation and a partial-alpha
output. A separate alpha output is one candidate, not a decision in this PR.
Preserve continuous alpha masters while comparing representations later.
Do not quantize all masters into a new palette or expand the vocabulary now.

The 12 GB training GPU constrains temporal model size and sequence length.
Collect short clips first without promising that concatenating all frames will
fit. A later bounded memory/parity experiment must precede any real training.
