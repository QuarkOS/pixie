# Animation and transparency proposal

Extend Pixie with useful transparency and animation features while building reusable datasets for future generators, refinement models, and reward models. Use the existing 32x32 Krea 2 LoRA as a candidate-data generator and preserve its existing static-sprite workflow.

This is a design proposal based on the sanitized repository. It implements no features, changes no extraction gates, and starts no generation or training. Native 32x32 pretraining requires separate explicit operator authorization. Dataset sizes, quality gains, GPU memory fit, and throughput remain unmeasured.

## Existing capabilities and boundaries

| Capability | Current evidence | Boundary |
| --- | --- | --- |
| Transparent sprite backgrounds | [worker/common.py](../worker/common.py#L16) renders palette index 0 with alpha 0 and all other indices with alpha 255. | Binary opacity only. |
| Extracted RGBA sprites | [extract_v2.py](../data-pipeline/extract_v2.py#L287) derives binary alpha from keyed sheets. | Intermediate opacity is not recovered. |
| Alpha-aware generator loss | [v3_model.py](../train/v3_model.py#L402) adds transparency classification loss to palette-token NLL. | Existing Pixie remains a single-image 16x16 model. |
| Krea LoRA training sheets | [sheets.py](../data-pipeline/scripts/sheets.py#L16) composites assets onto magenta and saves RGB. | Original partial alpha is discarded by compositing. |
| Human corrections | [flywheel.mjs](../web/lib/flywheel.mjs#L388) stores original and edited pixels with edit metadata. | The schema assumes a single 256-pixel image. |

The operator reports an existing trained 32x32 Krea 2 LoRA. Its appearance prior is a starting point for experiments, not evidence of temporal consistency or native alpha output. No checkpoint was inspected.

## Transparency

### Improve binary transparency first

Retain the current locked extraction behavior as the baseline. Review candidate sprites against light, dark, and checkerboard backgrounds. Record missing interior pixels, unwanted holes, magenta fringe, stray components, and clipped silhouettes separately.

Preserve the raw sheet, extraction version and parameters, original mask, corrected mask, and final indexed sprite. Human mask corrections can later support refinement or segmentation training. A mask correction must retain its verified source image and training-permission state.

Do not loosen the calibrated 3x3 gates without a measured comparison. The 2x2 and 3x3 paths need separate validation. Thin limbs and changing silhouettes in animation also need sequence-specific checks before any change to foreground or component rules.

### Treat partial opacity as a new representation

Glass, smoke, and translucent effects require intermediate alpha values. One RGB image composited onto magenta cannot uniquely determine both the original foreground colour and its opacity. Preserve real RGBA source targets rather than expecting the keyer to reconstruct them.

A possible pixel-art prototype uses a small quantized alpha vocabulary, with the number of levels selected through visual evaluation. A future model could predict palette colour and alpha separately, or use a palette of RGBA entries. Compare these representations before committing to one. Keep colour and opacity values intact in storage even if an initial model uses quantized targets.

Native transparent diffusion needs appropriate model and decoder support. [LayerDiffusion](https://arxiv.org/abs/2402.17113) is a research example, not verified compatibility with Krea 2 or the current sd.cpp path. Prompt wording alone does not establish an alpha channel.

Evaluate composites against several backgrounds. Check premultiplied-alpha handling, colour fringes, transparency holes, and consistency between saved PNGs and editor rendering. Preserve pixel-grid sharpness during scaling.

## Animation collection and representation

### Change the meaning of a sheet

Existing sheets contain multiple independent assets. An animation sheet must contain ordered frames of the same asset. Keep a fixed native canvas, anchor, palette identity, facing direction, and frame order. Do not independently crop or recenter frames, since that can create artificial jitter.

Start experiments with constrained motions such as blinking, idle movement, flames, or sparkles. Walk cycles, attacks, and turns need stronger pose and identity control. This order is a proposed scope choice, not a measured difficulty ranking on your LoRA.

Test the existing LoRA on explicitly ordered animation-sheet prompts before changing model architecture. Review identity drift, pose order, repeated frames, alignment, and loop closure. A failed motion sequence may still yield a reviewed static sprite, but must not enter the animation set as a valid sequence.

If the existing LoRA does not produce useful coherent sequences, build a separate curated animation-sheet dataset and consider a new-folder LoRA experiment under separate authorization. Keep the current LoRA and its datasets intact. Synthetic sequences require review and explicit provenance; real or human-corrected sequences provide a separate source of supervision.

### Proposed asset record

The organizing unit is an asset containing one or more ordered frames. Static images are single-frame assets. Keep native dimensions independent of display size.

| Field group | Proposed fields |
| --- | --- |
| Identity | Asset ID, schema version, subject, original prompt, corrected caption when needed. |
| Pixels | Native width and height, palette hash, colour-index arrays, optional alpha arrays, content hashes. |
| Sequence | Sequence ID, action, facing direction, loop flag, ordered frame IDs, per-frame duration in milliseconds. |
| Alignment | Coordinate convention, fixed anchor, optional ground-contact metadata. |
| Provenance | Source sheet ID, sheet geometry, generation settings, model and LoRA hashes, extraction version, source permissions. |
| Lineage | Reference asset ID, parent revision ID, procedural transforms, human edits. |
| Review | Static quality, identity consistency, motion correctness, flicker, loop quality, reviewer judgments. |
| Admission | Training-permission state, holdout-policy hash, split group, validation status. |

Keep related sequences, retries, source sheets, and edit descendants in the same split. Apply frozen Blind-50 and operator-provided Eval-50 exclusions before generation and training admission. The existing review found enforcement gaps, so current exports must not be treated as a proven admission gate.

## Deliver product features before learned motion

Extend the editor with frame duplication, ordering, per-frame duration, onion-skin previews, shared palettes, anchor controls, and loop playback. Preserve verified original frames and revision ancestry when users edit a sequence.

Export a sprite-sheet PNG plus a JSON manifest containing frame rectangles, timing, anchors, loop behavior, and native dimensions. Animated previews are a separate presentation format. Select formats according to transparency and timing requirements; keep the native asset as the training source of truth.

Procedural transformations can provide initial effects without a learned animation model. Record the exact transformations, distinguish them from generated motion, and avoid counting duplicated or translated frames as independent original sprites. A transformation that moves the whole sprite is not evidence of articulated motion.

Collect lightweight sequence feedback such as whether the action is correct, the character stays consistent, and the loop looks usable. Ask before treating an edit as an improvement. Retain negative and uncertain judgments rather than converting every save into a positive training label.

## Model options

| Design | Benefit | Main risk |
| --- | --- | --- |
| Joint sequence generation | All frames can share context. | More image tokens, memory, and sequence-level decoding complexity. |
| Reference-conditioned frame generation | Explicit identity source and incremental generation. | Temporal drift, repeated frames, and loop closure still need modeling. |

Prototype the reference-conditioned option first. Supply the original sprite, action, direction, and frame phase. A prototype may also condition on the preceding frame, but should retain the original reference to reduce dependence on accumulated errors. This is a design preference, not a proven quality result.

For a candidate frame-conditioned autoregressive design, separate reference-image conditioning from text and motion metadata. Frame phase alone does not guarantee coherent movement. Compare independent reference-conditioned decoding with temporal conditioning using complete-sequence human judgments.

[AnimateDiff](https://arxiv.org/abs/2307.04725) illustrates temporal modeling added to an image-generation system. It does not establish a drop-in motion module for Krea 2. Check architecture and implementation compatibility before considering external animation components.

The existing 16x16 P50 checkpoint cannot become a native 32x32 sequence model through an inference flag. A new target size or temporal architecture needs its own data, compatibility checks, and explicit training scope.

## Sequence-level ranking and evaluation

The current image ranker can provide frame-quality signals but cannot establish good animation. Preserve frame-level and sequence-level scores separately, along with scorer versions.

Collect paired sequence judgments with randomized presentation order and identical playback settings. Allow ties and both-bad outcomes. Separate subject/action correctness, identity stability, palette consistency, intentional motion, anchor stability, transparency quality, and loop closure.

Do not minimize inter-frame pixel change as the sole objective. A frozen image would score well. Compare useful action and temporal stability together. Group all derived pairs from one judgment so they do not appear as independent annotations.

Proposed acceptance checks include exact frame dimensions, valid palette and alpha values, legal timings, preserved frame order, reproducible exports, and train/evaluation separation. Quality acceptance requires reviewed held-out sequences. Numeric targets should be set after a baseline study; none are claimed here.

## Hardware and staged rollout

Use independent jobs first. The 3080 Ti can remain assigned to the existing main training recipe, and the 3060 Ti to chat if the complete worker fits. Use the part-time 5070 Ti for independently resumable generation, archive scoring, or bounded experiments. These are recommendations rather than measured device assignments.

| Stage | Deliverable | Evidence needed before proceeding |
| --- | --- | --- |
| 1 | Better binary-alpha review and mask lineage. | Corrected masks survive storage, rendering, and export without changing locked gates. |
| 2 | Sequence asset schema, frame editor, playback, and export. | Round-trip pixels, timing, anchors, and lineage remain correct. |
| 3 | Small curated animation pilot using Krea candidates and human corrections. | Reviewed sequence quality, grouped splits, permissions, and holdout exclusions. |
| 4 | Reference-conditioned animation design and authorized prototype. | Comparison against procedural and existing-LoRA baselines on complete sequences. |
| 5 | Partial-alpha experiment if required by target assets. | Real alpha targets and satisfactory composites across backgrounds. |
| 6 | Broader model training only after separate approval. | Defined image resolution, architecture, dataset manifest, fresh output folder, and measured hardware fit. |

No generation, training, benchmark, or quality evaluation was run for this proposal. The next step is a separately scoped implementation of the asset schema and editor features, with tests for pixel, alpha, timing, and export behavior.
