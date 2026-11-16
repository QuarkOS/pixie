# TASK: native 32×32 Pixie (Pixie-32) training code

You implement the code. **We launch the run**; do not start long training
yourself. Develop and test on CPU with **synthetic fixtures** (shapes below).
The real warm-start smoke test (v3 checkpoint, palette, CLIP-L weights,
tokenizer) is run by the operator on the training machine; you will not get
those files or their paths.

Read first: `AGENTS.md`, `docs/pixie32-plan.md`, `docs/dataset-krea2-combined.md`,
`reviews/` on branch `review/pipeline-hardware-20261115` (findings 4 and 6).

## Deliverable

A new folder `train/pixie32/` (do **not** edit the live v3 files in `train/`
or `worker/` in place; import from them instead):

1. `build_data_p32.py`: from `$DATA_ROOT` + the R1 list, write a **new**
   dataset folder `$DATASET_OUT` (refuse if it exists):
   - `ids.txt` (ordered row ids), `tokens.npy` `[N,1024] uint16` palette ids,
     `text.npy` / `subj_text.npy` `[N,18] uint16` CLIP ids (BOS + ≤16 + EOS,
     EOS-padded, long captions cut with EOS in the last slot, same as
     `train/s7_data.py::trim`), `subject_id`, `weight` (balance × edge bucket),
     `noflip`, split arrays (train / iid eval by sheet / unseen-subject probe).
   - `manifest.json` with the sha256 of `ids.txt`, of every array, of the
     palette, the tokenizer file and the R1 list, plus all drop counts.
2. `p32_model.py`: P32 = P50 at 16×16 patches with 2D-RoPE, the same
   `SlotCausalHead`, CLIP-L penultimate text (18 positions), 3 palette + 1 style
   cond slots, and `load_v3_warmstart(path)` that reports copied / remapped /
   new tensors.
3. `train_p32.py`: trainer (see recipe and guards below).
4. `run_p32.sh`: launcher (env-driven paths, refuses busy GPU **for the selected
   device only**, writes `launch.txt`), and `stop_p32.sh`.
5. Unit tests (CPU): patchify round trip at 32×32; causality (output for patch m
   unchanged when patches ≥ m change); RoPE identity on prefix; warm-start load;
   cache alignment check fails on a shuffled id list; resume fingerprint
   mismatch is refused.

## Approved recipe points

- Native 32×32, 2×2 patches → 256 patches, sequence 279.
- Keep the 4097-entry palette (padded vocab 4160, `NO_PALETTE = 4097`).
- Warm-start from the best v3 s7 checkpoint (condboost 64M, `img_064M.pt`,
  model-only; config `p50 d512 L16 h8 text_len 18 n_pal 3 style`).
- Causal intra-patch head, 2D-RoPE.
- Tighter dedup (quantized-token dedup) and holdout (caption **and** subject,
  exact and contains, also on the trimmed caption; eval split by sheet).
- v3 defaults to start from: AdamW, wd 0.05, dropout 0.1, alpha loss weight
  0.25, caption dropout 0.1 (CFG), subject-as-caption ~0.15, h-flip 0.5 except
  directional or text-like captions, cond rates p_pal_top1 0.30 / p_pal_top3
  0.50 / p_style 0.70, balance alpha 0.35 cap 6, WSD schedule (warmup → constant
  → linear decay to 1e-5). v3 used peak 3e-4 from scratch; for the warm start
  propose a value and justify it.
- Style tags at 32×32: rescale the fill thresholds (v3: < 80 / ≤ 180 of 256
  opaque pixels → × 4 of 1024) and recalibrate colour-count thresholds on this
  data (v3: ≤ 12 / ≤ 24 colours); document the numbers.
- Run length: propose one for ~134k sprites with numbers (images/s from your
  smoke test or an estimate clearly labelled as one). Keep `best.pt` by iid NLL.
- Periodic: `last.pt` (full, resumable), model-only `img_XXXX.pt`, eval, and a
  sample grid of everyday prompts at 32×32 (not Blind-50 / Eval-50 phrases)
  every N images.

## Guards from the review (must have)

- **Cache alignment:** the trainer loads `manifest.json`, recomputes the sha256
  of `ids.txt` and every array it reads, and asserts they match and that all
  arrays have the same row count and order. Never join caches by position alone.
- **Resume:** store a fingerprint (batch, seed, dataset sha, schedule, total
  images, model config) in `last.pt`; refuse to resume on mismatch instead of
  reinterpreting step × batch. `--max-hours` must write a *paused* marker, not
  `train_summary.json`; only a finished run writes `train_summary.json`. Epoch
  plans come from a per-epoch seeded generator so resume is exact.
- **Holdouts:** Blind-50 (`holdout/blind-50.json`) and Eval-50
  (`holdout/eval-50.json`, frozen; not in the public tree). Both are JSON arrays of
  50 lowercase phrases. Load both by default (`--blind50`, `--eval50` may
  override the paths); a missing, unparsable or non-50-entry file is a hard
  error. Assert 0 matches in the train split at startup and log it.
- **New output folder:** refuse to start if the output folder has any
  checkpoint and `--resume` is not given; never write outside it; atomic saves
  (`.tmp` + rename).
- **Device:** use one explicit device everywhere (events, RNG, sync, memory
  stats), or document `CUDA_VISIBLE_DEVICES` + `cuda` only.

## Hardware limits

- One **12 GB** consumer GPU (Ampere, bf16, SDPA flash). Peak must stay under
  ~11 GB with headroom; add gradient checkpointing and CE chunking flags. v3 at
  16×16 used ~6.5 GB at batch 128; the 32×32 sequence is ~3.2× longer, so expect
  to need a smaller batch (estimate, not measured).
- Host: 4 CPU cores, 15 GB RAM. Keep the token cache on GPU or in RAM
  (~280 MB for 134k × 1024 uint16).
- The root SSD is nearly full; all outputs go under `$RUNS` / `$DATASET_OUT`.
- No paid GPUs, no global installs (project venv: torch 2.9 + cu128,
  numpy, Pillow, scipy, tokenizers).

## Warm-start inputs (operator-provided; expose as flags / env vars)

| Env var | CLI flag | What |
|---|---|---|
| `$V3_CKPT` | `--v3-ckpt` | v3 s7 condboost checkpoint (model-only, 64M views) |
| `$PALETTE` | `--palette` | 4097-colour palette |
| `$CLIP_DIR` | `--clip-dir` | folder holding the CLIP-L text weights file `clip_l14_text_l11_f32.pt` |
| `$TOKENIZER` | `--tokenizer` | CLIP BPE `tokenizer.json` |

The flag wins over the env var. Fail with a clear message if one is missing.
Build synthetic fixtures with the **same formats and shapes** (random weights
are fine) so the loaders and the warm-start remap are tested on CPU.
The details below were read from the real files.

### `$V3_CKPT`: `torch.save` dict, load with `torch.load(..., map_location="cpu", weights_only=False)`

Top-level keys: `model` (dict str → tensor), `step` (int, 187500), `images`
(int, 24,000,000 = views in this fine-tune), `base_images` (int, 40,000,000 =
views before it; total = 64M), `train_secs`, `next_eval`, `next_save`, `best`
(floats), `arch` (`"p50"`), `config` (dict), `args` (dict of the trainer's
argparse values), `epoch` (int), `cursor` (int), `rng` (dict `torch`, `cuda`).
There is **no** `optimizer` key (model-only checkpoint), so a warm start begins
with a fresh optimizer.

`config` = `{"arch": "p50", "d_model": 512, "n_layer": 16, "n_head": 8,
"ffn_mult": 4, "dropout": 0.1, "head_mult": 2, "text_len": 18, "n_pal": 3,
"style": true}`; relevant `args`: `text_len 18, cond true, batch 128,
alpha_weight 0.25`.

`model`: 224 tensors, all float32, 66,309,632 parameters. Non-block tensors:

| Key | Shape |
|---|---|
| `bos`, `sep`, `pal_type` | `[512]` |
| `clip_ln.weight`, `clip_ln.bias` | `[768]` |
| `text_proj.weight` / `.bias` | `[512,768]` / `[512]` |
| `style_emb.0.weight`, `style_emb.1.weight`, `style_emb.2.weight` | `[4,512]`, `[3,512]`, `[4,512]` |
| `slot_emb.{0..3}.weight` | `[4160,512]` each |
| `pos.weight` | `[87,512]` (23 prefix + 64 patch-output positions) |
| `row_in.weight`, `col_in.weight`, `row_out.weight`, `col_out.weight` | `[8,512]` each |
| `ln_f.weight`, `ln_f.bias` | `[512]` |
| `head.q` | `[4,1024]` |
| `head.W_h.weight` / `.bias` | `[1024,512]` / `[1024]` |
| `head.W_j.{0,1,2}.weight` | `[1024,512]` each (no bias) |
| `head.E_pix.weight` | `[4160,512]` |
| `head.W_2.weight` / `.bias` | `[512,1024]` / `[512]` |
| `head.norm.w` | `[512]` (RMSNorm) |
| `head.W_out.weight` | `[4160,512]` (no bias) |

Per block `blocks.{0..15}.`: `ln1.weight/bias [512]`, `qkv.weight [1536,512]`,
`qkv.bias [1536]`, `out.weight [512,512]`, `out.bias [512]`,
`ln2.weight/bias [512]`, `fc1.weight [2048,512]`, `fc1.bias [2048]`,
`fc2.weight [512,2048]`, `fc2.bias [512]`. This matches `train/v3_model.py`
`P50` (`V3Config` above); `build(V3Config(**config))` + `load_state_dict(model)`
loads it strictly.

### `$PALETTE`: NumPy `.npy`, `np.load(..., allow_pickle=False)`

Shape `[4097, 3]`, dtype `uint8`, RGB. Row 0 is the transparent sentinel
(stored as `[0,0,0]`; never use it as a nearest-colour target). Rows 1..4096
are the opaque colours (4096 distinct rows in total, so row 0 equals one
opaque colour; map by index, not by RGB lookup, for id 0).

### `$CLIP_DIR/clip_l14_text_l11_f32.pt`: `torch.save` of a plain state dict

178 float32 tensors, 115,971,072 parameters: the first **11 of 12** layers of
the OpenAI CLIP ViT-L/14 text model in Hugging Face naming, **without**
`final_layer_norm` (penultimate features).

- `text_model.embeddings.token_embedding.weight [49408,768]`
- `text_model.embeddings.position_embedding.weight [77,768]`
- `text_model.encoder.layers.{0..10}.` + `layer_norm1.{weight,bias} [768]`,
  `layer_norm2.{weight,bias} [768]`, `self_attn.{q,k,v,out}_proj.weight [768,768]`
  and `.bias [768]`, `mlp.fc1.weight [3072,768]`, `mlp.fc1.bias [3072]`,
  `mlp.fc2.weight [768,3072]`, `mlp.fc2.bias [768]`

Load with `train/v3_model.py::ClipTextPenultimate().load_hf(sd)` (strict).
Causal attention, quick-GELU; run frozen in bf16 under `no_grad`.

### `$TOKENIZER`: Hugging Face `tokenizers` JSON (`Tokenizer.from_file`)

BPE, vocab 49,408, 48,894 merges, end-of-word suffix `</w>`, unk
`<|endoftext|>`. Normalizer: NFC → collapse whitespace → lowercase.
Pre-tokenizer: CLIP regex split + ByteLevel. Post-processor adds
BOS `<|startoftext|>` = 49406 and EOS `<|endoftext|>` = 49407. No truncation or
padding is configured. Example: `encode("Red Potion").ids` =
`[49406, 736, 46055, 49407]` (`add_special_tokens=False` gives `[736, 46055]`).
Model input = 18 ids: BOS + up to 16 ids + EOS, padded with EOS (49407); longer
captions are cut and EOS goes in the last slot (`train/s7_data.py::trim`).

## Placeholders

`$V3_CKPT`, `$PALETTE`, `$CLIP_DIR`, `$TOKENIZER` (above), `$RUNS` (parent of
run folders), `$DATA_ROOT` (combined dataset), `$DATASET_OUT` (new p32 dataset
folder). Older scripts in this repo also use `$PIXIE_ROOT`.

## Send back

What you changed, test output, smoke numbers if any (loss, step/s, peak VRAM),
the exact smoke and launch commands you propose (for the operator to run), and open questions. Label estimates.
