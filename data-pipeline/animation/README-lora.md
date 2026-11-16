# Proposed Krea 2 2x2 animation LoRA

This recipe is gated on at least **400 measured artist-made four-frame training
sequences**. The 400 count is a launch target, not a measurement. Build the
sheet manifest with the procedural cap, then check its train records before
starting a real run. Blind-50 and Eval-50 remain excluded by the shared policy.

The proposed first bounded run uses 512x512 2x2 sheets, 256 screen pixels per
32x32 logical frame, rank 32, alpha 32, learning rate `1e-4`, 50 warmup steps,
1,500 total steps, batch size 1, one repeat, AdamW 8-bit, BF16, SDPA,
gradient checkpointing and 10 swapped blocks. These are proposed settings,
not measured performance or quality results. Start in a new output folder and
keep the frozen pilot baseline separate.

The dataset config is [animation_2x2_512.toml](../configs/animation_2x2_512.toml).
The training command follows the repository's musubi-tuner setup:

```sh
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/krea2_train_network.py \
  --dit "$KREA_DIT" --vae "$KREA_VAE" \
  --dataset_config "$ANIMATION_DATASET_CONFIG" \
  --convrot_int8 --blocks_to_swap 10 --block_swap_h2d_only --block_swap_ring_size 2 \
  --sdpa --mixed_precision bf16 --gradient_checkpointing \
  --timestep_sampling krea2_shift --weighting_scheme none \
  --optimizer_type adamw8bit --learning_rate 1e-4 --lr_warmup_steps 50 \
  --network_module networks.lora_krea2 --network_dim 32 --network_alpha 32 \
  --max_train_steps 1500 --save_every_n_steps 250 --seed 42 \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --output_dir "$RUNS/animation-lora-2x2-01" --output_name animation-2x2
```

Captions come from the sheet builder: stable trigger, subject, action,
direction, frame count, grid, loop intent, fixed camera/scale, row-major order
and magenta background. Do not add per-frame captions or timing claims to the
image caption; durations remain in the manifest. Keep the caption vocabulary
versioned with the sheet manifest.

Evaluate the candidate checkpoint by generating the same frozen audit prompts,
seeds, steps, resolution and LoRA multiplier as the pilot. Score both outputs
with the existing animation scorer, then compare the score exports:

```sh
$PY -m animation_lora_eval \
  --baseline "$RUNS/animation-pilot-score" \
  --candidate "$RUNS/animation-lora-score" \
  --candidate-sheets "$RUNS/animation-sheets-2x2" \
  --output "$RUNS/animation-lora-comparison.json" \
  --blind50 "$DATA_ROOT/holdout/blind-50.json" \
  --eval50 "$DATA_ROOT/holdout/eval-50.json"
```

The evaluator refuses fewer than 400 artist training sheets unless the
explicit fixture-only `--allow-below-target` flag is supplied. It requires the
same subject/action keys in both score exports, checks the candidate sheet
manifest's canonical holdout hashes, and reports structural-valid, moving,
jitter, loop-error and optional embedding summaries plus candidate-minus-
baseline deltas. These metrics remain uncalibrated; human whole-sequence review
is required before promotion. Report denominators and keep Blind-50/Eval-50
out of training and audit prompts.
