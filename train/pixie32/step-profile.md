# Profile a bounded P32 update segment

The existing reporting clocks separate training, evaluation, sampling and
checkpoint work. This opt-in profile divides completed training updates into
data/setup, online CLIP or RAM cache transfer, forward/loss, backward, and
gradient clipping/AdamW. It does not change the recipe or resume fingerprint.
No GPU timings or speed improvements have been measured for this change.

Use the project venv and an idle explicitly selected GPU. Supply the **full**
`last.pt` from the b80/cache branch and keep its original recipe. The script
refuses a busy GPU, a reused output folder, a model-only source, or incompatible
resume settings. Original byte-identical holdout JSON files must match the
dataset manifest. Do not reformat those files to make the example match.

```bash
"$PY" -m train.pixie32.profile_steps \
  --out "$RUNS/p32-step-profile-001" --warmup 10 --steps 50 -- \
  --data "$DATASET_OUT" --resume "$P32_CKPT" \
  --blind50 "$BLIND50" --eval50 "$EVAL50" \
  --clip-dir "$CLIP_DIR" --tokenizer "$TOKENIZER" \
  --clip-cache "$CLIP_CACHE" --clip-cache-max-gb 4 \
  --device cuda:0 --batch 80 --compile --ce-chunks 16 \
  --no-gradient-checkpointing --images 12000000 --warmup 1500 \
  --decay-images 2000000 --lr 1e-4 --lr-min 1e-5 --seed 42 \
  --save-every-images 100000 --eval-every-images 100000 \
  --sample-every-images 100000
```

Unset `CUDA_VISIBLE_DEVICES` for this wrapper so its physical-GPU idle check
matches the explicit index. If parent recipe values differ, use its values.
The first 10 completed segment updates are excluded from the step profile and
the next 50 are recorded. The smoke stops after 60 updates, runs the normal
final evaluation/grid, and retains a full checkpoint in the new folder.
Do not use its reported training throughput as an uninstrumented speed result.

`step-profile.json` contains each completed profiled step, actual image count,
host phase seconds, CUDA stream phase milliseconds, and per-phase medians/ranges.
It omits all input paths, caption text and weights. `paused.json` retains the
existing whole-segment and reporting clocks. Fewer available image views or an
interrupt can produce a partial profile; inspect `completed_profiled_updates`.

| Phase | Work inside the boundary |
|---|---|
| `data_setup` | Caption/image/condition preparation, host-to-device copies, augmentation, LR assignment and gradient zeroing |
| `clip_or_cache` | Online frozen CLIP or host feature lookup/transfer |
| `forward_loss` | Model forward and CE, including the existing finite-loss check |
| `backward` | Loss backward and checkpoint recomputation, if configured |
| `clip_grad_adamw` | Gradient norm clipping and completed optimizer dispatch |

Host intervals include Python dispatch and blocking waits. CUDA event intervals
measure elapsed time on the selected stream, including host dispatch gaps and
stream idle time. They are **not pure kernel execution time**. Host and CUDA
measurements overlap and must not be added. The completed-update wall duration
also includes the existing final device synchronization; that wait is outside
the host phase intervals.

Events are recorded only during requested profile steps. The trainer reuses its
existing end-update synchronization to read them and adds no per-phase device
synchronization. Instrumentation still perturbs execution. Warmup is a fixed
number of updates, not proof that compilation completed or caches stabilized;
check the first rows for outliers and increase warmup up to 100 if needed.
Use a separate unprofiled comparison for any throughput claim.

A large `data_setup` host interval motivates testing RAM gather/pinned transfer.
A large `clip_or_cache` interval bounds the value of further CLIP work. Large
CUDA forward/backward intervals motivate a compiler/kernel trace. When a stream
interval is large but has dispatch gaps, a trace is needed before calling it
GPU compute. These measurements identify the next experiment; they do not
establish its expected speedup.

## CPU verification

```bash
"$PY" -m unittest train.pixie32.test_step_profile -v
```

Fixtures verify bounded completed-update counts, explicit-device CUDA event
wiring without synchronization, and unchanged model/AdamW/RNG/fingerprint after
instrumenting the real trainer loop. They do not establish GPU timing accuracy,
hardware throughput or a new optimal training recipe.
