#!/bin/bash
set -x
: "${KREA_LORA_ROOT:?set KREA_LORA_ROOT}"
cd "$KREA_LORA_ROOT/musubi-tuner" && . ../.venv/bin/activate
for c in 2x2_512 4x4_1024; do
  python src/musubi_tuner/krea2_cache_text_encoder_outputs.py --dataset_config ../configs/box/dataset_$c.toml --text_encoder ../models/qwen3vl_4b_bf16.safetensors --batch_size 4 --device cpu --skip_existing
done
for c in 2x2_512 4x4_1024; do
  python src/musubi_tuner/krea2_cache_latents.py --dataset_config ../configs/box/dataset_$c.toml --vae ../models/qwen_image_vae.safetensors --device cpu --skip_existing
done
echo CACHE_ALL_DONE
