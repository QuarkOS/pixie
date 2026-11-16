# Data flywheel review

You want to gather as much useful data as possible for future sprite generators, reward models, and refinement models. This review proposes collecting richer raw evidence and deriving separate, versioned training datasets from it.

The findings are based on source inspection at `8a94120c0e29129a627265608f6c3d1770c0de8e`. Collection yield, annotation quality, and downstream model gains were not measured. The [pipeline review](pipeline-hardware-review.md) covers training correctness, extraction, inference, and hardware allocation.

## Preserve the data already being generated

### F1. Archive every candidate independently of display policy

[worker_v3.py](../worker/worker_v3.py#L370) normally generates 64 reward-v7 candidates. Its archive list at line 299 keeps only `rows[:ncand]`, usually 20. [chat.js](../web/api/_lib/chat.js#L258) also caps the received candidate list. The remaining images are lost even though scores and seed labels survive.

Archive every generated image as validated native palette indices. Keep the four public results and the optional twenty-candidate view as separate presentation policies. A uint16 16x16 grid requires 512 bytes, so 64 grids require 32 KiB before metadata and compression. This is a format calculation, not a storage benchmark.

Store a stable candidate ID, pixel hash, palette hash, generator checkpoint hash, prompt, conditioning, sampler version, temperature, top-k, CFG, generation batch, original candidate position, scorer artifact version, and score components. The existing archive already stores much of the model, score, and conditioning metadata at [flywheel.mjs](../web/lib/flywheel.mjs#L276). Temperature and top-k are configured but absent from worker model information at [worker_v3.py](../worker/worker_v3.py#L176).

The current seed labels do not establish reproducibility. Preserve job and batch context until the sampler contract described in the pipeline review is fixed. Design the archive schema to carry native dimensions and palette identity so future resolutions do not depend on today's hard-coded 256-pixel representation.

### F2. Quarantine invalid archives instead of inventing transparent sprites

[flywheel.mjs](../web/lib/flywheel.mjs#L265) replaces failed PNG conversions or missing pixels with all-transparent grids. This turns an archive failure into an apparently valid candidate.

Accept validated worker indices directly, record an explicit invalid state when conversion fails, and exclude invalid candidates from training exports. Track archive completeness and failure counts. Do not label persistence failures as model failures.

### F3. Make event delivery reliable and idempotent

[chat.js](../web/public/chat.js#L59) sends auxiliary events without confirming persistence and ignores failures. None-reason UI at line 483 can acknowledge a reason before it is durably stored. [editor.js](../web/public/editor.js#L2129) allows a download to finish even when edit logging fails.

Keep downloads available, but add client event IDs, acknowledgements, bounded retry storage, and server-side uniqueness. Retries must not multiply votes or edits. Measure attempted, accepted, duplicate, rejected, and abandoned events separately.

## Preserve what each judgment actually means

### F4. Record exposure and separate reward rank from display position

[chat.js](../web/public/chat.js#L432) sorts candidates by rank. [flywheel.mjs](../web/lib/flywheel.mjs#L18) records toggles and actions but has no explicit exposure event containing the shown candidate IDs and their display order.

Record exposure ID, candidate IDs, display permutation, view, visibility, and selection policy. Visibility is not proof that a person compared every candidate. Dedicated comparison screens should record the explicit comparison set.

Randomize left/right positions in comparison tasks and hide reward scores. Test a small exploration allocation that exposes candidates beyond the existing ranker's favorites. Record inclusion probabilities so later analysis can distinguish selection effects from preference. The allocation is an experiment, not an established optimum.

Human-comparison reward learning is supported by [Christiano et al.](https://arxiv.org/abs/1706.03741) and [Stiennon et al.](https://arxiv.org/abs/2009.01325). These papers do not establish the best sampling or weighting policy for Pixie.

### F5. Export valid grid comparisons without fabricating unseen negatives

[flywheel.mjs](../web/lib/flywheel.mjs#L938) rejects only the top four candidates even when a user chooses from the full grid. The none-good branch at line 885 also exports only four negatives.

Use explicit comparison sets to retain valid additional supervision. Do not interpret every unselected archived image as rejected. Keep one choice group as one observation even if it produces several training pairs, and normalize its total contribution. Pair multiplication does not create independent human judgments.

Keep ties, both-bad judgments, and uncertainty as distinct labels in dedicated comparisons. Preserve the existing `chosen_grid: null` representation for none-good examples. A pairwise loss requiring a winner cannot consume these examples unchanged.

### F6. Join downloads and shares to the exact candidate and actor

[flywheel.mjs](../web/lib/flywheel.mjs#L859) collects download and share events into a set keyed only by job ID. At line 920, any such event strengthens votes on that job, regardless of which candidate or actor produced the action.

The action boost also precedes anonymous downweighting at lines 927 through 933. An anonymous vote can receive weight 2 instead of 0.2 when the download branch applies. These are heuristic policy weights, not calibrated confidence.

Join actions using candidate ID, actor, edit revision, and time. Keep trust and action strength as separate factors. Preserve raw evidence so weighting policies can change without recollecting data. Downloads indicate utility but are weaker than an explicit controlled comparison.

### F7. Verify canonical selections and edit parents server-side

The normal vote endpoint already validates ownership and derives canonical selections at [vote.js](../web/api/chat/vote.js#L20). Its downstream hook at [handlers.mjs](../web/lib/handlers.mjs#L251) instead copies candidate and rank values from the request body. Prefer the canonical vote record and reject inconsistent linkage.

The generic event handler at [handlers.mjs](../web/lib/handlers.mjs#L81) also accepts job and candidate metadata that need stronger linkage checks. Edit submission at line 128 accepts the client's original pixels, prompt, and conditioning without resolving them against the archived parent.

Resolve original pixels and conditioning from the archived candidate. Store the edited result with a verified parent revision. Preserve multi-step edit ancestry. Validate event relationships without implying that the existing voting endpoint lacks all ownership checks.

### F8. Separate editing effort from confirmed improvement

[flywheel.mjs](../web/lib/flywheel.mjs#L757) calls small edits gold polish. Export at line 834 derives weights from pixel-change counts and actions. Edits changing 97 through 160 pixels receive no positive weight under the current branches.

A small edit can worsen a sprite. A large redraw can be valuable. Preserve all valid, permitted edits with before and after indices, masks, operations, duration, action, and lineage. Those fields are already largely captured at [flywheel.mjs](../web/lib/flywheel.mjs#L388).

Collect an optional improvement judgment or a blinded before/after comparison. Allow edit-intent tags such as outline repair, colour correction, or changed subject. If the subject changes, collect a corrected caption rather than retaining an incorrect original label. Treat edit count and duration as activity metadata rather than proof of quality.

### F9. Separate relative preference from absolute acceptability

The user's favorite can still be the least-bad candidate. Add an occasional acceptability judgment such as usable as-is, needs editing, or wrong subject. Keep normal generation lightweight and ask one focused question at a useful moment.

For targeted annotation, collect separate judgments of subject correctness, pixel cleanliness, conditioning compliance, and game-asset usability. They can support separate reward outputs or diagnostic classifiers. No architecture or quality improvement is established by this proposal.

## Derive distinct training products

| Product | Admission evidence | Possible use |
| --- | --- | --- |
| Accepted sprite-caption-conditioning set | Explicit acceptability or reviewed final edit, with valid permission and provenance | Generator fine-tuning |
| Preference groups | A recorded preferred candidate and the alternatives actually compared | Reward learning and preference training |
| Rejection and failure labels | Explicit none-good or candidate-specific reason | Absolute quality calibration and failure classifiers |
| Edit pairs and trajectories | Verified parent, edited result, intent, and improvement status | Refinement and localized correction models |
| Unlabeled archive | Valid indexed pixels and generation provenance | Later annotation, deduplication, rescoring, and coverage analysis |

Do not automatically convert every click, download, retry, or generated sample into a positive or negative training example. Preserve training-permission state through every derivative. Keep raw data richer than today's export format.

[flywheel.mjs](../web/lib/flywheel.mjs#L939) exports preference tuples with useful job and trust metadata but omits much of the generation provenance. Publish an immutable dataset manifest containing source hashes, schema version, label policy, split assignments, inclusion criteria, and provenance references.

The `who='trusted'` export actually includes every tier except excluded at [flywheel.mjs](../web/lib/flywheel.mjs#L820). Make that policy explicit. Cap contributions by actor, job, prompt family, and duplicate image group. Calibrate confidence with repeated blinded judgments and disagreement analysis rather than relying only on account status.

## Choose useful generation and annotation work

The existing diagnostics already identify high-reward none-good cases, lower-ranked grid wins, and repeated failing prompts at [flywheel.mjs](../web/lib/flywheel.mjs#L661).

Extend these queues with reward-model disagreement, novel examples, underrepresented subjects and conditioning combinations, and uncertain close comparisons. Retain a random sample to measure ordinary traffic and avoid letting the current reward model select every future training example. Exact-prompt failure thresholds miss rare paraphrases, so group by subject family while retaining original text.

| Observed evidence | Proposed follow-up |
| --- | --- |
| High score followed by none-good | Review a potential reward-model blind spot. |
| Human prefers a lower-ranked image | Compare it directly with the scorer's favored images. |
| Repeated wrong-subject labels | Gather reviewed examples of that subject and nearby confusing subjects. |
| Palette or style failures | Collect controlled conditioning comparisons. |
| Retry chain ending in an accepted image | Preserve the chain, final intent, and successful result. A retry alone is not rejection. |

The part-time 5070 Ti is suited to independent archive rescoring and generation shards once software and memory fit are verified. This should not interrupt the main training run or interactive worker. No additional generation or native 32x32 training is authorized by this review.

## Protect evaluation and measure useful yield

Fix the demonstrated Blind-50 export gap before expanding collection. Require frozen Blind-50 and operator-provided Eval-50 policies at training admission and generation admission. Missing manifests must not silently permit training exports.

Split by prompt family and image/edit lineage rather than individual derived pairs. Keep related candidates, retries, and revisions out of opposite train/evaluation splits. Record holdout hashes in each immutable dataset manifest.

Measure unique accepted sprites, valid comparison groups, confirmed improvements, subject and conditioning coverage, event delivery, archive completeness, duplicate rates, and annotator disagreement. Report useful labels per generation and per reviewer minute. Do not use exported pair count alone as evidence of dataset growth.

Implementation order is complete candidate archives and reliable canonical events, then exposure-aware exports and precise action joins, then validated edit lineage and separate training products, then active annotation and targeted generation experiments.
