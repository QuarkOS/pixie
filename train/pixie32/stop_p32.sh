#!/usr/bin/env bash
set -euo pipefail
: "${RUNS:?Set RUNS}"
: "${RUN_NAME:?Set RUN_NAME to the run to pause}"
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Invalid run name." >&2
  exit 2
fi
pid_file="$RUNS/$RUN_NAME/worker.pid"
if [[ ! -f "$pid_file" ]]; then
  echo "No active trainer PID recorded." >&2
  exit 2
fi
read -r trainer_pid < "$pid_file"
if [[ ! "$trainer_pid" =~ ^[0-9]+$ || ! -r "/proc/$trainer_pid/cmdline" ]]; then
  echo "Trainer is not active." >&2
  exit 2
fi
mapfile -d '' -t command < "/proc/$trainer_pid/cmdline"
module_ok=false
output_ok=false
for ((i=0; i<${#command[@]}; i++)); do
  if [[ "${command[$i]}" == -m && "${command[$((i+1))]:-}" == train.pixie32.train_p32 ]]; then
    module_ok=true
  fi
  if [[ "${command[$i]}" == --out && "${command[$((i+1))]:-}" == "$RUNS/$RUN_NAME" ]]; then
    output_ok=true
  fi
done
if [[ "$module_ok" != true || "$output_ok" != true ]]; then
  echo "PID does not match this trainer and output; refusing to signal." >&2
  exit 2
fi
kill -INT "$trainer_pid"
echo "Pause requested. Wait for last.pt and paused.json before resuming into a new folder."
