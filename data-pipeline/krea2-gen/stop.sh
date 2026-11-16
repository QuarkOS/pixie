#!/usr/bin/env bash
# Stop only this generation run. Does not touch Pixie or other services.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PIDS="$ROOT/data/pids.json"

stop_pid() {
  local pid="${1:-}"
  [[ -n "$pid" && "$pid" != 0 ]] || return 0
  kill "$pid" 2>/dev/null || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.3
  done
  kill -9 "$pid" 2>/dev/null || true
}

if [[ -f "$PIDS" ]]; then
  # Parse without requiring python in case the venv is busy
  for key in server dashboard runner; do
    pid=$(sed -n "s/.*\"$key\":\([0-9]*\).*/\1/p" "$PIDS" | head -1)
    stop_pid "${pid:-}"
  done
fi
for f in server dashboard runner; do
  if [[ -f "$ROOT/data/$f.pid" ]]; then
    stop_pid "$(cat "$ROOT/data/$f.pid")"
    rm -f "$ROOT/data/$f.pid"
  fi
done
rm -f "$PIDS"
echo "Stopped krea2-gen. Progress in $ROOT/data is kept; start again to resume."
