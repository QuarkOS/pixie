# Pixie project guide

This repository is a sanitized, code-only copy of the Pixie pipeline. Use it to propose and implement changes to training, data extraction, sampling, the chat worker, and the web app.

Pixie is a small autoregressive model that generates palette-indexed pixel-art game sprites from a text prompt. The live 16×16 line is called v3 (s7 / condboost). A native 32×32 model is planned but must not be trained unless the operator explicitly asks.

## Hard rules

- Do not look for, infer, or record who operates this project, where it is hosted, or which accounts, domains, machines, or file paths it uses. If a task needs a secret, a host, or a path, ask; do not hunt for them.
- No secrets in code, logs, commits, or replies. Read secrets only from environment variables the operator provides. Never print secret values.
- Do not delete, overwrite, or move existing training runs, checkpoints, labels, or evaluation sets. Every new run goes in a new folder.
- Never train on the Blind-50 or Eval-50 holds. They are evaluation-only.
- Do not deploy, SSH, push to a live service, or change production config.
- Do not contact anyone, open accounts, or spend money (including paid cloud GPUs).
- Do not install Python or torch globally. Use a project venv.
- Send back: what you changed, why, commands to run, and measured numbers. Label estimates as estimates. Do not invent metrics.

## Repository layout

- `train/` — v3 model, trainer, data prep, typo/CLIP-cache helpers, eval/compare, launch scripts.
- `worker/` — chat worker and sampling (palette/style cond). `v3_model.py` here is the same s7 model as in `train/`.
- `web/` — web app (labelling, chat, A/B, admin, auth). Generated vote/pool dumps are stubbed empty; this is code, not production data.
- `data-pipeline/` — Krea 2 sheet generation, `extract_v2` / `reextract`, LoRA train configs and helper scripts.
- `holdout/blind-50.json`, `holdout/eval-50.json` — not shipped. Supply your own (or set `PIXIE_HOLDOUT_DIR`). Do not train on them.

Checkpoints, image caches, sheets, sprites, logs, and labels are not in this repo. Point scripts at local data via arguments or env vars (`PIXIE_ROOT`, `DATA_DIR`, and the flags on each script).

## Model (v3 / s7)

Architecture **P50**: width 512, 16 layers, 8 heads, 2×2 patch tokens (64 patches, 4 slots each), slot-aware causal MLP intra-patch head. Autoregressive over the 64 patches.

Text: frozen CLIP-L/14 text transformer, penultimate hidden states (layer 11). With `--text-len 18` the CLIP ids are BOS + 16 tokens + EOS. Causal CLIP features match a longer context for captions that fit in 16 ids.

Conditioning (`--cond`): 3 palette slots (pixel-embedding ids, with a palette-cond type embedding; unused slots are `NO_PALETTE = 4097`) plus one style slot built from three tag embeddings:

- fill: any / small / medium / full
- outline: any / dark / soft
- colour-count: any / low / mid / high

Sequence is `[text | cond | patches]`. Palette vocab is padded to 4160; indices ≥ 4097 are masked to -inf at sample time. Images are 16×16 palette indices. An older pixel-token trunk (`pix`, d256 × L12, 256 raster tokens) and a v2 fine-tune path (`v2ft`) still exist in the trainer for comparisons.

## Training recipe (s7 full run)

Use `train/train_v3.py --arch p50 --cond --text-len 18 --schedule wsd`. The launcher `train/run_v3_full_linux.sh` is the reference command (paths are placeholders such as `$PIXIE_ROOT/...`; retarget them).

Measured recipe:

- 36 million image views, batch 128, AdamW, weight decay 0.05, dropout 0.1
- WSD: 1500-step warmup, then constant LR `3e-4` (`--lr-min 1e-5` is for the decay branch)
- Permanent full checkpoints at 18M / 27M / 36M views (`--save-at-images`); also periodic saves every 1M
- `--compile`, CLIP feature cache (`--clip-cache`, built by `build_clip_cache_s7.py`), `--ce-chunks 1` (patch-order CE). A 4-way CE chunk was slower in the speed test.
- Caption mix (drawn on the GPU at epoch start so resume is exact): 10% blank captions for CFG, 17.5% subject-as-caption, 10% synonym when one exists, typo captions with `--p-typo 0.15` (1–2 small misspellings: dropped, swapped, doubled, or adjacent-key letter). Never typo-augment Blind-50 prompts. Horizontal flip 0.5 except directional captions. Style cond dropped to ANY with probability `1 - p_style` (default `p_style=0.4`).
- Rare-subject sampling: `--balance-alpha 0.35`, `--balance-cap 6`.
- Alpha-channel loss weight 0.25. Eval every 1M images on the v3 probe (100 unseen subjects) and an iid split, including conditioned palette/style stats.

Decay branch (`train/run_v3_decay18_linux.sh`): new folder only, `--schedule decay --images 2e6 --branch-from` a full checkpoint, linear LR `3e-4 → 1e-5`. Do not write into the parent run folder.

Speed notes (order-of-magnitude, 12 GB consumer GPU, compiled, batch 128): about 9.6 step/s with the CLIP cache vs about 8.9 step/s online CLIP; roughly 1.2k images/s. Peak reserved memory for this recipe is about 6–6.5 GB. The chat worker needs about 4.5 GB. Do not run training and the chat worker on the same 12 GB GPU at once.

Every run must write to a new output folder. Resume is supported; do not overwrite someone else's run.

## Pixie-32

There is no native 32×32 training corpus in the usual 16×16 trees. Options, in order: ship nearest-neighbour downloads (already a display/download size), prototype a small 16→32 refiner if 32×32 targets exist, or train a native 32×32 model (2×2 patches with 256 tokens, or 4×4 patches keeping 64 tokens). Do not start a native 32×32 pretrain unless the operator explicitly asks.

## Data pipeline (Krea 2 → 32×32 sprites)

Sheets are generated with **Krea 2 Turbo** (the 5-step Turbo variant) via `sd.cpp`: `krea2_turbo` IQ4_XS weights, Qwen3-VL-4B Q4_K_M text encoder, wan 2.1 VAE. Two layouts:

- 2×2 sheet at 512, LoRA `pix32-final`
- 3×3 sheet at 576, LoRA `p3-1500`

Generation code is under `data-pipeline/krea2-gen/` (Linux runner + dashboard) and `data-pipeline/krea2-win/` (Windows installer/scripts). LoRA training configs and helper scripts are under `data-pipeline/scripts/` and `data-pipeline/configs/`. The third-party musubi-tuner tree is not vendored here; use the public musubi-tuner and these configs.

Prompts should ask for a pixel-art game icon / sprite on a **magenta** (`#FF00FF`) background so the keyer can extract cells. Do not include huge prompt queues in git.

### extract_v2 (3×3 @ 576, 6 screen px per art px)

`data-pipeline/extract_v2.py`, plus `reextract.py` and calib helpers. Locked after a labelled 378-sprite sweep:

1. Seed the key with the v1 rule on the 32×32 (key tolerance 90, plus the original magenta-tint rule).
2. Widen only by flood-fill from already-keyed magenta into nearby near-magenta on the raw 192 cell: `grow_tol=120`, 6 px.
3. Fringe: 2 rounds of outline strip at `tol <= 100` on the 32×32. Do not apply the loose tint rule to interior colours (pink frosting, purple subjects).
4. Thresholds: `fg` 0.04–1.00, `spill` ≤ 0.08, `main_comp` ≥ 0.55, `purity` ≥ 0.35. `edge_touch` is a flag only, not a hard reject.

A semantic filter (SigLIP so400m, drop if cosine to `a 32x32 pixel art game icon of {phrase}` is below 0) is part of the intended pipeline; the filter script is not in this dump. Do not loosen extract gates or the never-train-on-holdout rule without a measured comparison.

## Eval

- Blind-50: the phrases in `holdout/blind-50.json`. Never in training data, typo augmentation, or datagen for those exact subjects.
- Eval-50: the captions in `holdout/eval-50.json` (frozen). A separate exact-caption holdout; also never trained on.
- Probe / iid splits are for train-time eval only. Report numbers from the scripts; do not invent scores.

## Inference and web app

The chat worker (`worker/worker_v3.py`) loads an s7 checkpoint, runs CLIP, samples with CFG (typically 2.0), optional cond (palette + style), candidate ranking (reward-v7 / `fast_v7.py`), and returns PNGs. `fast_sample.py` is the sampling kernel (CUDA graphs on the live path). Config is env + `worker/config_v3.json` (paths in the checked-in file are placeholders).

The web app (`web/`) serves chat, A/B preference collection, a pixel editor, share pages, and an admin/analysis UI. Auth is email magic-link plus optional Google and X OAuth; all client ids and secrets are environment variables (`AUTH_GOOGLE_CLIENT_ID`, `AUTH_X_CLIENT_ID`, `AUTH_SECRET`, and so on). Admin session and ingest tokens also come from env (`ADMIN_PASSWORD`, `TOKEN_SECRET`, `TRAINING_INGEST_TOKEN`, `CHAT_WORKER_KEY`). Do not hard-code values or bake tokens into admin URLs.

Owner-imported calibration labels and live A/B/pool dumps were removed from this copy and replaced with empty stubs (`web/api/_lib/ownerdata.js`, `abdata.js`, `pooldata.js`) so the app still imports. Rebuild those from local data when you need to run the site against real votes.

There is no typo-correction step in chat.

## How to work

1. Read this file and the script headers that apply to your task.
2. Make changes on a branch. Keep diffs reviewable. Do not rewrite unrelated files.
3. For training or extraction experiments, use a new output folder and a short, recorded command line.
4. Prefer tests already in `train/test_v3.py` and `web/tests/`. Add tests when you change behaviour.
5. Return: a summary, the patch or PR, how to run it, and measured results. If you could not run something, say so clearly.

If you need a checkpoint, a data path, a GPU, or a secret, stop and ask. Do not try to discover the live deployment.
