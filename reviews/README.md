# Pixie review findings

This documentation-only review records issues and proposals for the Pixie training pipeline, data gathering, inference stack, and data flywheel. It considers a 3060 Ti, 3080 Ti, and occasional 5070 Ti. Findings refer to source revision `8a94120c0e29129a627265608f6c3d1770c0de8e`.

- [Pipeline and hardware findings](pipeline-hardware-review.md) cover correctness, extraction, resume, inference, and proposed performance experiments.
- [Data flywheel findings](data-flywheel-review.md) cover collection reliability, archives, label meaning, edits, reward datasets, and targeted annotation.
- [Decision record](pipeline-review-decisions.tsv) records review choices and evidence limits.

## Prioritized follow-up checklist

The checklist describes proposed follow-up work. No application fixes are implemented in this PR.

- [ ] Enforce frozen holdouts across generation, caption variants, and training exports.
- [ ] Refuse destructive or reused preparation destinations and preserve immutable manifests.
- [ ] Validate image-cache alignment with ordered dataset rows, palette, and split identity.
- [ ] Correct selected-device CUDA operations and scope GPU busy checks to the selected GPU.
- [ ] Validate resume invariants and distinguish resumable time limits from completion.
- [ ] Fix candidate RNG metadata and honor comparison-grid seed and sampling contracts.
- [ ] Fingerprint generator jobs, fix short subject queues, and record extraction versions.
- [ ] Preserve all generated candidates and quarantine invalid archive data.
- [ ] Persist canonical events reliably with verified candidate and edit ancestry.
- [ ] Record comparison exposure and join downloads and shares to exact candidates and actors.
- [ ] Separate accepted sprites, preference groups, rejection labels, edit trajectories, and unlabeled candidates.
- [ ] Version confidence policies, permissions, provenance, grouped splits, and dataset manifests.
- [ ] Add targeted annotation and generation queues while retaining random coverage.
- [ ] Run hardware-specific memory, latency, throughput, and quality experiments before changing the recipe.
- [ ] Repair portable local test setup and retain standalone reproductions for the demonstrated defects.

## Evidence limits

The review distinguishes source findings, isolated runtime reproductions, calculations, and unmeasured proposals. No real dataset contamination, GPU speedup, model-quality gain, or complete worker memory fit was established. No training, deployment, production configuration change, or extraction-gate change was performed.
