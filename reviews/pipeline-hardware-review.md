# Pixie pipeline and hardware review

You want correctness issues and useful optimization ideas across data gathering, training, and inference for a 3060 Ti, 3080 Ti, and occasional 5070 Ti.

Reviewed the code at `8a94120c0e29129a627265608f6c3d1770c0de8e`. This report changes no application code, datasets, extraction gates, or training settings. P1 means fix before relying on the affected workflow. P2 means a conditional correctness or operational problem.

## Findings that should come before optimization

### 1. P1. Training exports can include Blind-50

[init.mjs](../web/lib/init.mjs#L51) constructs the flywheel with `holdoutPrompts: []`. [flywheel.mjs](../web/lib/flywheel.mjs#L41) optionally loads holdout files from the data directory and silently accepts missing or malformed files. It does not automatically load the repository's Blind-50. Training export relies on this set at line 829.

An isolated run of the actual flywheel code with a fresh temporary directory recorded an opted-in owner edit using a phrase from the repository's Blind-50. The export returned one training tuple for that phrase. Observed results were `knownBlind50Recognized=false` and `exportedHoldoutTuples=1`. No existing labels or datasets were accessed.

Load the shared frozen Blind-50 explicitly. Require the operator-provided Eval-50 before admitting training exports. Missing or invalid holdout configuration must block training admission. This proves a guard defect, not contamination of an existing dataset.

### 2. P1. Generation and subject augmentation have holdout gaps

[runner.py](../data-pipeline/krea2-gen/app/runner.py#L133) loads subjects unchanged, then queues them at lines 170 through 181. The Windows installer copy does the same. No holdout guard was found in `data-pipeline/`. Executing the actual loader in isolation preserved a synthetic forbidden phrase. The checked-in default generation subjects had zero exact normalized intersections with Blind-50; arbitrary supplied queues remain unguarded.

[build_data_v3.py](../train/build_data_v3.py#L85) excludes Eval-50 main captions and rechecks synonyms at line 201. Subject-only captions receive only a Blind-50 check at line 210. [train_v3.py](../train/train_v3.py#L418) subsequently samples these subject captions. A subject equal to an Eval-50 caption can pass if its main caption differs. An isolated counterexample demonstrated that predicate gap. The absent Eval-50 and actual data prevent measuring affected rows.

Use one normalization and admission policy for generation, exports, main captions, synonyms, subject captions, and typos. Audit tokenized truncation separately because CLIP sees the trimmed sequence.

### 3. P1. Dataset scripts can delete or overwrite existing artifacts

[sheets.py](../data-pipeline/scripts/sheets.py#L4) unconditionally deletes the dataset directory before rebuilding it. Line 11 overwrites sprite metadata with new splits. [select.py](../data-pipeline/scripts/select.py#L94) writes fixed sprite IDs and metadata. These are direct conflicts with the project's preservation rule.

[reextract.py](../data-pipeline/reextract.py#L144), [extract_v2.py](../data-pipeline/extract_v2.py#L371), and [prep_v3.py](../train/prep_v3.py#L15) accept existing output directories and replace matching artifacts. Re-extraction can also leave old sprites in a previous acceptance bin.

Require fresh destinations and immutable source manifests. Do not run the destructive sheet builder against existing data. The typo and CLIP-cache builders already demonstrate the appropriate new-directory behavior.

### 4. P1. Image and caption alignment is not validated

[prep_v3.py](../train/prep_v3.py#L51) records manifest and output hashes. [train_v3.py](../train/train_v3.py#L290) loads the labels and image cache separately and joins them by position without reading `cache_manifest.json`.

A stale cache with the same row count can silently pair images with incorrect captions and evaluation rows after rebuilding a split. The CLIP-caption hash checks a different relationship and cannot catch this.

Bind caches to ordered train and evaluation row IDs, palette identity, and relevant tokenization metadata. Preserve compatibility with typo-data derivatives whose image mapping is intentionally unchanged. Merely comparing the entire data manifest may reject valid derivatives while leaving the underlying relationship unspecified.

### 5. P1 for multiple GPUs. The trainer mixes selected and default CUDA devices

[train_v3.py](../train/train_v3.py#L274) accepts a device without setting the current CUDA device. Prefetch copies into `self.dev`, but records an event on the current stream of the current device at line 223. RNG save and restore at lines 389 and 355 also omit the device. Synchronization and memory reporting at line 528 use the default device.

When training on a noncurrent `cuda:1`, RNG restoration and memory accounting refer to the wrong GPU. The prefetch completion event is also associated with the wrong device rather than the target transfer. Runtime corruption frequency was not measured.

Until fixed, expose one selected GPU with `CUDA_VISIBLE_DEVICES` and use logical `--device cuda`. Specify the device explicitly for events, RNG, synchronization, and memory statistics in the eventual fix.

### 6. P1/P2. Resume can change the meaning of a run

[train_v3.py](../train/train_v3.py#L349) restores without comparing batch, seed, dataset identity, and other exact-resume arguments. At line 387, image counts are recomputed using the current batch. At step 10,000, changing batch 128 to 64 reinterprets 1,280,000 historical image views as 640,000. That is an arithmetic counterexample, not an executed training run. Epoch plans and schedule interpretation change too.

Keep batch and data identity fixed for an exact resume. An intentional recipe change needs a distinct branch operation and output folder with explicit image accounting.

[train_v3.py](../train/train_v3.py#L567) also writes `train_summary.json` after `--max-hours`, while startup exits if that file exists at line 277. A time-limited, incomplete run cannot resume normally. Distinguish a resumable pause from completion before scheduling training on the part-time GPU.

### 7. P2. Reported candidate seeds do not reproduce the candidates

[fast_sample.py](../worker/fast_sample.py#L333) seeds the default Gumbel sampler only from `seeds[0]`. Subsequent candidate seeds do not affect the draw. [worker_v3.py](../worker/worker_v3.py#L270) nevertheless generates and returns a separate seed for each image.

Changing the first seed or batch shape changes the shared random stream. A displayed candidate seed alone cannot reproduce its image. The worker also ignores the queue's requested job seed. Comparison-grid seed controls therefore do not establish matched random draws.

Either implement per-candidate deterministic random streams or store an honest job seed plus candidate position, batch shape, and sampler version. Verify single-candidate, reordered-batch, and chunked-batch reproduction before relying on seeds for comparisons.

### 8. P2. Comparison-grid sampling budgets are ignored

[grid.mjs](../web/lib/grid.mjs#L307) submits the selected seed and sample count. Under reward-v7, [worker_v3.py](../worker/worker_v3.py#L370) uses its fixed configured candidate count, normally 64, instead of the grid's requested sampling budget. Along with the ignored seed, this makes comparisons uncontrolled and their requested cost inaccurate. Keep best-of-64 as an explicit chat policy while honoring or clearly rejecting the grid contract.

## Operational issues

| Issue | Evidence and consequence | Recommended correction |
| --- | --- | --- |
| All inference workers share one running slot. | [chat.js line 213](../web/api/_lib/chat.js#L213). Two isolated mock worker polls dispatched one of two queued jobs. | Keep one worker for now. Adding concurrent workers requires per-worker leases and capacity, timeout ownership, and idempotent results. |
| Launchers reject work when any GPU is busy. | [full launcher line 14](../train/run_v3_full_linux.sh#L14) and [decay launcher line 14](../train/run_v3_decay18_linux.sh#L14). A worker on the 3060 Ti blocks training on an idle 3080 Ti. | Select one physical GPU and scope the busy check and lock to it. |
| Generator resume uses sheet number rather than content. | [runner.py line 184](../data-pipeline/krea2-gen/app/runner.py#L184). A changed queue can skip jobs because old IDs are marked complete. | Use a run manifest and content fingerprints. Give each shard its own output directory. |
| Small queues can generate no sheets or crash. | [runner.py line 180](../data-pipeline/krea2-gen/app/runner.py#L180). One subject with a 2x2 grid pads to two subjects and schedules zero sheets. [fast_batch.py line 24](../data-pipeline/krea2-win/fast_batch.py#L24) has an out-of-range variant. | Cycle the original nonempty list until the final sheet is full and record repeated cells. |
| Online extraction uses legacy gates. | [runner.py line 12](../data-pipeline/krea2-gen/app/runner.py#L12) imports the old extractor. The same synthetic stats passed v2 and failed legacy gates. | Record extraction version and route 3x3 candidates through the locked v2 process into a fresh folder. Do not globally replace 2x2 gates. |
| VRAM benchmarks can report another GPU's memory. | [bench.py line 42](../data-pipeline/krea2-win/bench.py#L42) queries every GPU and takes the largest value. | Select the device explicitly and report baseline, device peak, and process memory separately. |
| Cached training assumes POSIX file APIs. | [train_v3.py line 196](../train/train_v3.py#L196) and line 208 use `posix_fadvise` and `preadv`. | Use the existing cached path on Linux or WSL2. Native Windows needs a supported fallback and verification. |
| Worker device selection accepts only the literal `cuda`. | [worker_v3.py line 158](../worker/worker_v3.py#L158) selects CPU for a configuration value such as `cuda:1`. The default sampler can then try to prepare CUDA graphs. | Isolate one GPU with CUDA visibility. Fail clearly without CUDA or implement a tested CPU path. |

Further data risks need the actual corpus to quantify. [sheets.py](../data-pipeline/scripts/sheets.py#L7) splits per sprite, while [select.py](../data-pipeline/scripts/select.py#L74) permits related variants, so related examples may cross train and evaluation. [reextract.py](../data-pipeline/reextract.py#L56) chooses the first matching sheet basename, which is ambiguous across combined runs. [caption.py](../data-pipeline/scripts/caption.py#L5) replaces its loaded caption dictionary from another backup and saves only at the end of its batch loop. These warrant provenance, uniqueness checks, and atomic batch saves.

The optional pitch import at [extract_v2.py](../data-pipeline/extract_v2.py#L32) depends on the working directory and does not directly point to the pitch module's `app` directory. Pitch diagnostics can silently be unavailable. This was not shown to change the locked extraction gates.

The semantic SigLIP filter is documented as absent from this sanitized copy. Structural acceptance alone must not be interpreted as semantic acceptance or permission to train.

## Use the GPUs for independent jobs first

This is a recommendation from the code and documented memory budgets, not a benchmark of your cards. NVIDIA lists [8 GB for the 3060 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/), [12 GB for the 3080 Ti](https://www.nvidia.com/en-eu/geforce/graphics-cards/30-series/rtx-3080-3080ti/), and [16 GB for the desktop 5070 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/).

| GPU | Default assignment | Qualification |
| --- | --- | --- |
| 3080 Ti | Main s7 training with cached CLIP, bf16, compile, and CE chunk 1. | The project documents this recipe for a 12 GB consumer GPU. Fix device selection and per-GPU launcher guards first. |
| 3060 Ti | Interactive inference while the 3080 Ti trains. | The documented worker budget is about 4.5 GB, but actual reward models, graph captures, display usage, and candidate count must be fit-tested. Do not co-schedule generation on this card while serving chat. |
| Part-time 5070 Ti | Independent generation shards, semantic scoring, cache building, evaluation, or new-folder experiments. | Work must be independently resumable. Avoid making the main run depend on this GPU remaining available. |

The trainer has neither DDP nor gradient accumulation. Independent jobs avoid adding distributed recovery and synchronization to a heterogeneous, intermittent pool. DDP does not combine the cards' memory into one larger GPU. If later justified by measurement, use DDP rather than DataParallel, following [PyTorch's guidance](https://docs.pytorch.org/docs/main/notes/cuda.html).

Use a Blackwell-capable torch build for the 5070 Ti. [PyTorch 2.7 introduced Blackwell support with CUDA 12.8 wheels](https://pytorch.org/blog/pytorch-2-7/). Verify the actual installed torch, CUDA, and compiler combination in a project venv. No software upgrade was performed here.

## Optimization experiments in priority order

All speed and quality benefits below are unmeasured hypotheses.

1. **Remove repeated inference CLIP work.** [worker_v3.py line 273](../worker/worker_v3.py#L273) encodes the same prompt N times and the same blank N times per request. Encode one prompt row, cache the blank feature, then expand. Compare features and sampled distributions against the existing path.
2. **Reuse extraction masks.** [extract_v2.py line 202](../data-pipeline/extract_v2.py#L202) reconstructs the full-sheet grown mask for each cell's spill calculation. Compute it once per sheet. Require identical pixels, scores, and gate decisions before accepting a faster implementation.
3. **Profile ranking and candidate count.** Measure sampling, ranking, transfer, PNG encoding, and total latency separately. Test candidate counts 16, 32, and 64 with a held-out preference protocol before reducing the current count. Fewer candidates trade selection quality for latency.
4. **Reduce redundant feature-cache reads if storage is limiting.** At text length 18, each stored caption row is 17 x 768 x 2 = 26,112 bytes. Read duplicate caption IDs once per batch, scatter them, or use a bounded hot-caption cache. This format calculation is not a measured I/O bottleneck.
5. **Cache evaluation text features.** Training already caches frozen CLIP features, while periodic evaluation still encodes text online. Bind any new cache to CLIP weights, token table, text length, and precision.
6. **Tune batch only after resume fixes.** Compare batch 64 and 128 first, then 192 or 256 if memory permits. Compare equal image budgets and report convergence as well as images per second. CE chunks 2 or 4 are memory experiments; the project's historical speed result favors chunk 1.
7. **Tune LoRA offload per card.** The existing command swaps ten blocks. Precache frozen text and VAE features, then reduce swapping only when a fit test leaves adequate memory. No fixed swap count or speedup is established for these cards.

Further inference experiments include chunking reward-model scoring before reducing candidate counts, avoiding unnecessary device-to-host round trips, and profiling the slot head, top-k, and Gumbel operations outside the existing trunk CUDA graph. These are proposed profile targets, not established runtime bottlenecks. The source review found matching cached prefix positions, causal masks, patch shifts, slot ordering, and palette/style CFG conditioning between the reference and fast samplers. GPU numerical parity remains untested.

The project's historical notes report roughly 1.2k images per second. At that rate, 36 million views require about 8.3 hours of compute. This is an estimate from supplied notes, excludes preparation, compile, evaluation, and checkpoint overhead, and is not a measured runtime on the 3080 Ti.

For future benchmarks, warm up compilation and graph capture separately. Run at least three steady-state trials per configuration, alternate their order, and report spread, peak allocated and reserved memory, and end-to-end throughput. Keep prompts, conditioning, candidate counts, image views, and scoring comparable. Identify the limiting stage before adopting an optimization.

## Verification and boundaries

- Parsed all 56 repository Python files with the standard-library AST parser. No syntax errors were found. This does not validate torch behavior.
- Training and worker `v3_model.py` files have identical SHA-256 values. This rules out source drift between those copies, not sampler bugs.
- Isolated actual-function checks reproduced the one-subject queue loss and legacy/v2 gate difference.
- Isolated actual web modules reproduced the single inference slot and the Blind-50 export gap.
- The detailed reproduction commands were executed by the component reviewers. Their results are recorded here, but no standalone reproduction script is included. This limits independent rerunning of those checks from the report alone.
- The existing `web/tests/chat_wait_test.mjs` was attempted against its temporary local server. It exited with an unsettled top-level await in cleanup at line 167. The cleanup waits for a child exit after it may already have exited. No end-to-end pass or performance result is claimed.
- `features_test.mjs` requires an absent external PNG, and `ab_serve_test.mjs` assumes the removed real A/B corpus. The staged auth test also assumes external data. These were not treated as portable tests or executed against external services.
- No project venv was available for torch tests. No checkpoint, real dataset, GPU benchmark, or model-quality evaluation was used. Native 32x32 training was not started.

With an operator-provided project venv, the existing CPU model tests can be run from the repository root with `python train/test_v3.py`. The queue test is `cd web` followed by `node tests/chat_wait_test.mjs`, but its startup and cleanup issue must be resolved before it can provide reliable end-to-end evidence. No extraction or training command should run against existing output folders.

Prove It Works shaped the review by requiring actual-module reproductions where practical and marking unexecuted model paths. Explain the Number kept historical recipe numbers and calculated estimates separate from measurements on your hardware.

The next implementation order is holdout admission and output preservation, then cache/resume/device correctness, then seed contracts and GPU scheduling, then the measured optimization experiments.

## Attention

An independent reviewer using the available parent model family spot-checked the primary findings and found no material contradiction. Cross-family review was unavailable. The review corrected the CUDA-event wording to specify the current stream and flagged the missing standalone reproduction scripts. The decision log's initial entries have date resolution rather than exact timestamps, and component agent names are session evidence rather than durable artifacts. None of these checks establish real-dataset contamination, hardware throughput, or complete worker memory fit.
