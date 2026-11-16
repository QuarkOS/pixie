# Pixie

Pixie is a small autoregressive model that paints palette-indexed pixel-art
game sprites from a short text prompt. This repository is the training code,
the sheet-generation and extraction tools, the chat worker, and the web app.
It does not include the training corpus or third-party weights.

**Weights** (both 66M parameters, on Hugging Face):

| Model | Output | Hugging Face |
| --- | --- | --- |
| P32 | native 32×32 sprites | [EmilioSchwaiger/pixie-32](https://huggingface.co/EmilioSchwaiger/pixie-32) |
| v3 / P50 | 16×16 sprites | [EmilioSchwaiger/pixie-16](https://huggingface.co/EmilioSchwaiger/pixie-16) |

See the [model card](MODEL_CARD.md) for training data, evaluation and known limits.

The code and the released weights are under the [Apache License 2.0](LICENSE).
The training images were generated with Krea 2 Turbo; see [NOTICE](NOTICE)
for the Krea attribution and third-party credits.

## Quick start: sample a released model

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch numpy pillow tokenizers safetensors huggingface_hub
hf download EmilioSchwaiger/pixie-32 --local-dir pixie-32   # or pixie-16
hf download openai/clip-vit-large-patch14 model.safetensors tokenizer.json \
  --local-dir clip-vit-large-patch14
python -m train.sample_pixie --model-dir pixie-32 \
  --clip clip-vit-large-patch14/model.safetensors \
  --tokenizer clip-vit-large-patch14/tokenizer.json \
  --prompt "red potion" --prompt "iron helmet" --n 4 --out sprites.png
```

Each row of `sprites.png` is one prompt, upscaled 4× with nearest-neighbour.
The same command samples `pixie-16` (the config picks the architecture). These
are raw samples; the chat app additionally ranks many candidates with a reward
model that is not released. P32 samples can show a few stray magenta pixels
(see the model card). Use a
GPU if you can: on a CPU, 12 sprites took about 6 minutes in one test run.

## Architecture

Two pixel models share one text encoder and one conditioning scheme.

**P32** is the native 32×32 model (`train/pixie32/`):

- width 512, 16 layers, 8 heads
- 256 raster-ordered 2×2 patch tokens (four palette slots each: TL, TR, BL, BR)
- partial axial 2D-RoPE on half of each head; the text-and-condition prefix has zero coordinates
- frozen CLIP ViT-L/14 text features (penultimate layer, 18 positions: BOS + 16 tokens + EOS)
- three palette slots and one style slot (fill, outline, colour-count)
- sequence `[text | cond | patches]`
- **66,424,320 parameters**, the same count as the released checkpoint and
  the synthetic fixture in
  [`train/pixie32/evidence/verification.md`](train/pixie32/evidence/verification.md).

**v3 / P50** is the 16×16 line (`train/v3_model.py`, `worker/v3_model.py`):
the same width, depth, heads, CLIP text encoder, and palette/style slots, with
64 patches instead of 256. The same verification note counts the matching v3
fixture at 66,309,632 parameters. Warm-start into P32 copies shape-compatible
tensors and expands the 8×8 position tables to 16×16.

Images are palette indices. The palette vocab is padded to 4160; indices at
or above 4097 are masked at sample time. Unused palette slots use
`NO_PALETTE` (4097).

## Data pipeline

Sheets are generated with Krea 2 Turbo (the 5-step variant) through
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp): a
quantized Krea 2 diffusion model, a Qwen3-VL-4B text encoder, and a Wan 2.1
VAE, plus a pixel-art LoRA. Prompts ask for a sprite on a magenta background
so the keyer can cut cells. Two layouts are wired up: 2×2 at 512, and 3×3 at
576. The generator lives in `data-pipeline/krea2-gen/` (Linux) and
`data-pipeline/krea2-win/` (Windows). LoRA training configs sit next to them
and expect the public musubi-tuner tree, which is not vendored here.

`data-pipeline/extract_v2.py` turns a 3×3 sheet into 32×32 sprites: seed the
magenta key, grow it by a short flood fill, strip a tight fringe, then apply
the locked gates (foreground fraction, spill, main component, purity).
`train/pixie32/build_data_p32.py` builds the training arrays from reviewed
rows, drops anything that matches the frozen holdouts, and writes digests the
trainer checks before the first step.

The Blind-50 and Eval-50 holdout lists are evaluation-only and are not
published. The tools expect them at `holdout/blind-50.json` and
`holdout/eval-50.json` (or under `PIXIE_HOLDOUT_DIR`); supply your own 50-phrase
lists to run the builder. Tests that need the original files skip without them.
Do not train on holdouts, typo-augment them, or generate them.

Animation tooling under `data-pipeline/animation/` can ingest artist sprite
sheets. Those packs are not in this repository. See
[what is not redistributed](#what-is-not-redistributed).

## Train and sample

Use a project virtual environment. Paths come from the environment, not from
a machine-specific default.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r train/pixie32/requirements.txt   # PyTorch; see that file
```

**v3 / P50.** `train/run_v3_full_linux.sh` is the reference launcher. Set
`PIXIE_ROOT` to a directory that holds the venv, data, CLIP cache, and run
folders. The recorded recipe is 36 million image views, batch 128, AdamW,
weight decay 0.05, dropout 0.1, WSD warmup of 1500 steps then a constant
learning rate of `3e-4`, and permanent checkpoints at 18M, 27M, and 36M
views. A decay branch is a new folder (`train/run_v3_decay18_linux.sh`), not
an overwrite of the parent run.

**P32.** Set `PY`, `DATASET_OUT`, `RUNS`, `V3_CKPT`, `CLIP_DIR`, and
`TOKENIZER`, then:

```bash
export RUN_NAME=p32-smoke-001
bash train/pixie32/run_p32.sh --smoke --smoke-steps 200 --batch 32
```

The smoke uses the full model. A proposed first full recipe (12 million
views, batch 32, peak LR `1e-4`) is written up in `train/pixie32/README.md`.
That recipe is a plan. It is not a measured training run in this tree.

**Sample.** `worker/worker_v3.py` loads a v3 checkpoint, runs CLIP, and
samples with classifier-free guidance (typically 2.0), optional palette and
style conditioning, and a candidate ranker. Point `worker/config_v3.json` at
local files. The vote-fitted fallback ranker (`worker/feat_linear.json`) is not
shipped; without it the fallback score is neutral, or set `PIXEL_FEAT_LINEAR`
to your own fit. For the released checkpoints use `train/sample_pixie.py`
(see the quick start). `worker/fast_sample.py` is the sampling kernel.

**Web.** `cd web && cp .env.example .env`, fill placeholders locally, and do
not commit `.env`.

## Measured results

Numbers below are copied from files in this tree unless marked otherwise.

| Result | Value | Source |
| --- | --- | --- |
| P32 parameter count | 66,424,320 | `train/pixie32/evidence/verification.md` (synthetic fixture, CPU); same count in the released checkpoint |
| Released P32 checkpoint | step 17,746, about 1.25M images, IID NLL 0.5237 | training log of that run (not in this tree); see [MODEL_CARD.md](MODEL_CARD.md) |
| Released v3 (pixie-16) checkpoint | 64M image views (40M base + 24M fine-tune), IID NLL 1.0231 | run summary (not in this tree); see [MODEL_CARD.md](MODEL_CARD.md) |
| v3 fixture parameter count | 66,309,632 | same file |
| P32 unit tests | 34 passed in 12.385 s | same file, on that machine's CPU |
| P32 production throughput and 12 GB fit | unmeasured | same file |
| v3 CLIP-cache speed | about 9.6 step/s vs about 8.9 step/s online; about 1.2k images/s | `AGENTS.md`, recorded for a compiled batch-128 run on a 12 GB consumer GPU |
| v3 reserved memory | about 6–6.5 GB | `AGENTS.md`, same recipe |
| Chat worker memory | about 4.5 GB | `AGENTS.md` |
| P32 profiled update | median 221 ms at batch 80; reserved 11.66 GB | `train/pixie32/compute-benchmark.md` (operator-supplied measurement; phase split is in that file) |

Sheet timings for several GPU classes are tabulated in
`docs/training-stack.md`. Read that file for the setup. Do not treat a number
marked estimate or "not measured" as a new result.

## Hardware

The v3 recipe above was recorded on a single 12 GB consumer GPU. The chat
worker is the lighter job (about 4.5 GB in the notes) and should not share
that card with training. P32's 12 GB fit is unmeasured until you run the
smoke on the card you care about. Generation notes also cover an 8 GB card
and a 16 GB card; those are product-class labels in the docs, not a required
cluster.

## Layout

| Path | What |
| --- | --- |
| `train/` | v3 trainer and the P32 trainer |
| `worker/` | chat worker and samplers |
| `web/` | labelling, chat, A/B, and admin UI |
| `data-pipeline/` | Krea 2 generation, extract_v2, LoRA configs, animation ingest |
| `docs/` | plans and measurement notes |

## What is not redistributed

See the note at the bottom of [CONTRIBUTING.md](CONTRIBUTING.md) and
[NOTICE](NOTICE). In short: Krea, Qwen, Wan, and CLIP weights are downloaded
by you; artist packs (LPC, Kenney, itch.io, OpenGameArt) are not in this tree;
the training sprites, holdout lists, preference votes and reward models are
not released. Nothing here is a license to those upstream weights or packs.

## Contributing

Issues and patches are welcome. Please:

- Keep secrets, checkpoints, datasets, `.env`, and logs out of git.
- Take paths from environment variables.
- Do not train on the Blind-50 or Eval-50 holdouts.
- Put each training run in a new folder.
- Add or update a test when you change behaviour, and report only numbers
  you measured.

Details and the test commands are in [CONTRIBUTING.md](CONTRIBUTING.md).
