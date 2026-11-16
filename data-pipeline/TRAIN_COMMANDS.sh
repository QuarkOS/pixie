#!/bin/bash
# Run only after the GPU is free (sd-server / other benchmark stopped) and with >=16 GB free disk. Not run yet.
set -e
: "${KREA_LORA_ROOT:?set KREA_LORA_ROOT}"
cd "$KREA_LORA_ROOT/musubi-tuner"
# 1) project venv. uv reuses its cache when torch is already downloaded.
uv venv -p 3.12 .venv && . .venv/bin/activate
uv pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
# 2) base weights (ungated Comfy-Org mirror). 13.49 GB DiT + 0.25 GB VAE. Text encoder NOT needed: TE outputs are pre-cached.
mkdir -p ../models
curl -fL -o ../models/krea2_raw_int8_convrot.safetensors https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_raw_int8_convrot.safetensors
curl -fL -o ../models/qwen_image_vae.safetensors https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors
# 3) train 2x2@512 (caches in ../cache/2x2_512 already). Saves every 250 steps.
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/krea2_train_network.py \
  --dit ../models/krea2_raw_int8_convrot.safetensors --vae ../models/qwen_image_vae.safetensors \
  --dataset_config ../configs/dataset_2x2_512.toml \
  --convrot_int8 --blocks_to_swap 10 --block_swap_h2d_only --block_swap_ring_size 2 \
  --sdpa --mixed_precision bf16 --gradient_checkpointing \
  --timestep_sampling krea2_shift --weighting_scheme none \
  --optimizer_type adamw8bit --learning_rate 1e-4 --lr_warmup_steps 50 \
  --network_module networks.lora_krea2 --network_dim 32 --network_alpha 32 \
  --max_train_steps 1500 --save_every_n_steps 250 --seed 42 \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --output_dir ../out/2x2_512 --output_name krea2-pixel32-sheet2x2
# If OOM: raise --blocks_to_swap (max 26). If it fits with VRAM to spare: lower it (faster).
# 4x4@1024: copy cache/4x4_1024 from the box, use ../configs/dataset_4x4_1024.toml, start with --blocks_to_swap 16.
# 4) convert for sd.cpp / diffusers-style loaders:
# python src/musubi_tuner/convert_lora.py --input ../out/2x2_512/krea2-pixel32-sheet2x2.safetensors --output ../out/2x2_512/krea2-pixel32-sheet2x2-diffusers.safetensors --target other
