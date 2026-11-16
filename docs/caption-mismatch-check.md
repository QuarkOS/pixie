# Rank possible caption mismatches with a local model

Use a local SigLIP/SigLIP2 checkpoint to rank suspicious image/caption pairs
before more generation. This is a diagnostic check, not a calibrated junk
filter. A mismatched sprite can still score highly, and a valid ambiguous
caption can lose to another caption. The tool changes no labels, training
weights, keep-lists, extraction gates or active runs.

The optional dependency is separate from the trainer. Install it only into a
project venv if needed; no command downloads model weights:

```bash
"$PY" -m pip install -r train/pixie32/requirements-quality.txt
"$PY" -m train.pixie32.caption_quality \
  --data-root "$DATA_ROOT" --r1 "$DATA_ROOT/r1_list.txt" \
  --blind50 holdout/blind-50.json --eval50 holdout/eval-50.json \
  --model-dir "$SIGLIP_DIR" --device cuda:0 --batch 8 \
  --negatives 3 --review 200 --expect-rows 133688 \
  --out "$RUNS/caption-review-new"
```

`SIGLIP_DIR` must be an explicitly supplied local Hugging Face model directory
containing the intended weights, tokenizer and processor. Both loader calls use
`local_files_only=True` and `trust_remote_code=False`. CPU is also supported via
`--device cpu`. CUDA requires one explicit index and an idle GPU. Do not run it
alongside the training job. No model, checkpoint or path discovery is performed.

For each sprite, the checker composites transparency onto RGB (180,180,180),
then uses the local model's processor and FP32 encoders. It compares the cosine
of the supplied caption to three deterministic captions from distinct other
subjects, using the template `a 32x32 pixel art game icon of {caption}`. The
margin is own-caption cosine minus the best alternative cosine. A negative
margin is flagged for review by default; `--margin` changes only that review
threshold. The alternatives are the next distinct normalized subjects in
sorted order, not an exhaustive or unbiased semantic confusion set.

Read `caption-scores.json` for all ID/image/caption hashes, cosines, margins
and alternative IDs/captions. `caption-summary.json` records the input and
local-model-directory hashes, preprocessing and explicit limitations without
absolute paths. The lowest margins appear first in `caption-review-*.png` and
`review.json`. Holdout matches are excluded before scoring or packet creation.
Choose a fresh output directory; existing outputs are refused.

SigLIP's processor pads to its maximum text length and truncates longer text.
A long caption's omitted attributes cannot be checked by this score. Model
scores and the number of review flags are **not a measured caption-mismatch
rate**. Have a person inspect the ranked examples and record `caption_match`
labels with the existing R1 audit. Compare false positives on a fully labelled
uniform packet before choosing any hard filter or weight. Do not use this
biased ranked review packet to estimate population prevalence.

The official [SigLIP documentation](https://huggingface.co/docs/transformers/model_doc/siglip)
describes the separate image/text embeddings and maximum-length padding. This
checker normalizes those embeddings and reports cosines instead of treating
sigmoid logits as calibrated probabilities of caption correctness.

Measured locally: **four CPU behavior tests passed**. They run the actual
packet code on synthetic data, verify mismatched-caption ranking and hashes,
reject invalid scores, exercise the real local-model adapter with fixture
embeddings, and verify both pretrained calls prohibit downloads and remote
code. No actual SigLIP weights or full R1 set were available; no model quality,
GPU speed or true mismatch-rate measurement is claimed.

```bash
"$PY" -m unittest train.pixie32.test_caption_quality
```
