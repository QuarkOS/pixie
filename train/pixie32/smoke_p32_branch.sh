#!/usr/bin/env bash
set -euo pipefail
: "${P32_CKPT:?Set P32_CKPT to the paused full P32 checkpoint}"
: "${CLIP_CACHE:?Set CLIP_CACHE to the verified frozen CLIP cache}"
if [[ -n "${RESUME:-}" ]]; then
  echo "Unset RESUME for a new full-state branch." >&2
  exit 2
fi
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_root/run_p32.sh" \
  --clip-cache "$CLIP_CACHE" --batch 80 --compile --ce-chunks 16 \
  --no-gradient-checkpointing "$@" --smoke --smoke-steps 20
