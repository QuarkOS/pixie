# Pixie model cards

Two checkpoints are released, both under Apache-2.0 with the Krea attribution
in [NOTICE](NOTICE). Files in each Hugging Face repo: `model.safetensors`,
`config.json`, `palette.npy` (the same 4097×3 palette for both models).

| Model | Hugging Face |
| --- | --- |
| P32, native 32×32 | [EmilioSchwaiger/pixie-32](https://huggingface.co/EmilioSchwaiger/pixie-32) |
| v3 / P50, 16×16 | [EmilioSchwaiger/pixie-16](https://huggingface.co/EmilioSchwaiger/pixie-16) |

## pixie-32 (P32)

| Field | Value |
| --- | --- |
| Name | Pixie P32 (`pixie-32`) |
| Version | Best checkpoint of the first full P32 run: step 17,746, about 1.25M training images seen |
| Architecture | P32: autoregressive transformer over palette indices. Width 512, 16 layers, 8 heads, 2×2 patch tokens (256 patches of four palette slots), partial axial 2D-RoPE, three palette slots and one style slot. See the README. |
| Parameters | 66,424,320 (counted from the released `model.safetensors`; 224 float32 tensors) |
| Checkpoint SHA256 | `1a26888728f3c523075f691e4fddb270889e0a4db814c516da016709a06b9dbc` (`model.safetensors`) |
| Output | One 32×32 RGBA sprite per sample; index 0 is transparent |
| Text encoder | External, not shipped: the first 11 layers of the [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) text model (penultimate features, 18 positions) |
| Intended use | Generate palette-indexed pixel-art game sprites (items, weapons, characters, creatures, props) from a short English prompt. |
| Out of scope | Photoreal images, text rendering, anything larger than 32×32, and subjects far outside game-sprite content. |

### Training data

| Field | Value |
| --- | --- |
| Training set | About 130k native 32×32 sprites generated for this project with Krea 2 Turbo plus a pixel-art LoRA, cut from 3×3 sheets by `data-pipeline/extract_v2.py`. Not released. |
| License of the data | The generated sprites are Krea 2 Outputs, owned by their generator per section 5.3 of the [Krea 2 Community License](https://github.com/krea-ai/krea-2/blob/main/docs/KREA-2-COMMUNITY-LICENSE). See [NOTICE](NOTICE). |
| Holdouts excluded | The Blind-50 and Eval-50 phrase lists were excluded from training by the builder's frozen-holdout check. The lists themselves are not published. |
| Personal data | None in the weights files. Captions are the generation prompts. |

### Weights license

| Field | Value |
| --- | --- |
| License | Apache-2.0, the same as the code. See [LICENSE](LICENSE) and [NOTICE](NOTICE). |
| Inherited restrictions | None claimed. No Krea or CLIP weights are part of the release; CLIP (MIT) is downloaded separately at sampling time. |

### Evaluation

| Field | Value |
| --- | --- |
| IID validation NLL | 0.5237 at the released step (lowest of the run, which is why this checkpoint was kept) |
| Blind-50 / Eval-50 | Not reported as a number for this release. |

### Known limitations

- Stray magenta pixels: the training sheets used a magenta key background, and
  some samples keep a few magenta background-key pixels near the edges.
- The run overfits after about 1.25M images; later checkpoints were worse on
  IID NLL, so this is not the last checkpoint of the run.
- Needs the external CLIP ViT-L/14 text encoder, which is linked, not shipped.
- Quality varies by subject; many samples are rough. Treat it as a small
  research model, not a finished asset generator.

## pixie-16 (v3 / P50)

| Field | Value |
| --- | --- |
| Name | Pixie v3 / P50 (`pixie-16`) |
| Version | Checkpoint served by the Pixie chat app: run `v3-full-s7-condboost40M`, file `full_064M`. 64M image views in total: a 36M-view constant-LR run decayed to 40M, then a 24M-view decay fine-tune (step 187,500 of that fine-tune) with heavier palette and style conditioning. |
| Architecture | v3 / P50: autoregressive transformer over palette indices. Width 512, 16 layers, 8 heads, 2×2 patch tokens (64 patches of four palette slots), slot-aware causal head, three palette slots and one style slot. See the README. |
| Parameters | 66,309,632 (counted from the released `model.safetensors`; 224 float32 tensors) |
| Checkpoint SHA256 | `fd6507350aedf58fa839fd94b95f84f889b72c7fc1e4c6f4bf66e2f6dd6178e8` (`model.safetensors`) |
| Output | One 16×16 RGBA sprite per sample; index 0 is transparent |
| Text encoder | External, not shipped: the first 11 layers of the [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) text model (penultimate features, 18 positions) |
| Intended use | Generate 16×16 palette-indexed pixel-art game icons and sprites from a short English prompt. |
| Out of scope | Photoreal images, text rendering, anything larger than 16×16, and subjects far outside game-sprite content. |

### Training data

| Field | Value |
| --- | --- |
| Training set | About 1.36M training rows built from [Angais/pixel-art-atlas-1.4m](https://huggingface.co/datasets/Angais/pixel-art-atlas-1.4m) (CC0-1.0; synthetic 16×16 sprites generated with Krea 2 Turbo and a pixel-art LoRA), with typo-augmented caption copies. |
| License of the data | CC0-1.0. The atlas images are Krea 2 Outputs; see [NOTICE](NOTICE). |
| Holdouts excluded | The Blind-50 and Eval-50 phrase lists were dropped from training (1,635 Blind-50 phrase/subject rows and 118 Eval-50 exact captions). The lists are not published. |
| Personal data | None in the weights files. Captions come from the dataset. |

### Weights license

| Field | Value |
| --- | --- |
| License | Apache-2.0, the same as the code. See [LICENSE](LICENSE) and [NOTICE](NOTICE). |
| Inherited restrictions | None claimed. No Krea or CLIP weights are part of the release. |

### Evaluation

| Field | Value |
| --- | --- |
| IID validation NLL | 1.0231 (with palette/style conditioning: 0.9698; held-out-subject probe: 1.0896), from the run summary at 64M views. Not comparable with the P32 number: different resolution and data. |
| Blind-50 / Eval-50 | Not reported as a number for this release. |

### Known limitations

- 16×16 only; small details (faces, text, thin tools) are often lost.
- Raw samples vary. The chat app draws 64 candidates per prompt and ranks them
  with a reward model fitted on preference votes; that ranker is not released,
  so expect to pick from several samples yourself.
- Sprites tend to fill the whole 16×16 cell.
- Needs the external CLIP ViT-L/14 text encoder, which is linked, not shipped.

## How to sample

```bash
python -m train.sample_pixie --model-dir pixie-32 \
  --clip clip-vit-large-patch14/model.safetensors \
  --tokenizer clip-vit-large-patch14/tokenizer.json \
  --prompt "red potion" --prompt "iron helmet" --n 4 --out sprites.png
```

Use `--model-dir pixie-16` for the 16×16 model; `config.json` selects the
architecture. `pixie-32` / `pixie-16` are the downloaded Hugging Face folders
and `clip-vit-large-patch14` holds `model.safetensors` and `tokenizer.json`
from [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14).
