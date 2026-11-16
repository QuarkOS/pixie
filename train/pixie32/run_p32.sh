#!/usr/bin/env bash
set -euo pipefail
: "${PY:?Set PY to the project venv Python}"
: "${CLIP_DIR:?Set CLIP_DIR to the CLIP text weights folder}"
: "${TOKENIZER:?Set TOKENIZER to tokenizer.json}"
: "${DATASET_OUT:?Set DATASET_OUT to the prepared dataset}"
: "${RUNS:?Set RUNS to the run parent directory}"
: "${RUN_NAME:?Set RUN_NAME to a new run name}"
: "${GPU:?Select exactly one GPU index or UUID}"
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
BLIND50=${BLIND50:-"${PIXIE_HOLDOUT_DIR:-$code_root/holdout}/blind-50.json"}
EVAL50=${EVAL50:-"${PIXIE_HOLDOUT_DIR:-$code_root/holdout}/eval-50.json"}
if [[ "$GPU" == *,* || ! "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Select one GPU and a simple new run name." >&2
  exit 2
fi
if [[ -e "$RUNS/$RUN_NAME" ]]; then
  echo "Output already exists; choose a new RUN_NAME, also for resume." >&2
  exit 2
fi
for arg in "$@"; do
  case "$arg" in
    --data|--data=*|--blind50|--blind50=*|--eval50|--eval50=*|--out|--out=*|--clip|--clip=*|--clip-dir|--clip-dir=*|--tokenizer|--tokenizer=*|--device|--device=*|--warmstart|--warmstart=*|--v3-ckpt|--v3-ckpt=*|--resume|--resume=*|--branch-from|--branch-from=*)
      echo "Set launcher paths and device through environment variables." >&2
      exit 2 ;;
  esac
done
gpu_uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
if [[ $(printf '%s\n' "$gpu_uuid" | wc -l) -ne 1 || "$gpu_uuid" != GPU-* ]]; then
  echo "GPU must resolve to one physical device." >&2
  exit 2
fi
busy=$(nvidia-smi -i "$gpu_uuid" --query-compute-apps=pid --format=csv,noheader)
if [[ -n "${busy//[[:space:]]/}" ]]; then
  echo "Selected GPU is busy; not starting." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
cd -- "$code_root"
args=(--data "$DATASET_OUT" --blind50 "$BLIND50" --eval50 "$EVAL50"
  --out "$RUNS/$RUN_NAME" --clip-dir "$CLIP_DIR"
  --tokenizer "$TOKENIZER"
  --device cuda:0)
if [[ -n "${RESUME:-}" && -n "${P32_CKPT:-}" ]]; then
  echo "Choose RESUME for exact continuation or P32_CKPT for a new batch/cache branch." >&2
  exit 2
elif [[ -n "${RESUME:-}" ]]; then
  args+=(--resume "$RESUME")
elif [[ -n "${P32_CKPT:-}" ]]; then
  args+=(--branch-from "$P32_CKPT")
else
  : "${V3_CKPT:?Set V3_CKPT to a v3 checkpoint}"
  args+=(--v3-ckpt "$V3_CKPT")
fi
exec "$PY" -u -m train.pixie32.train_p32 "${args[@]}" "$@"
