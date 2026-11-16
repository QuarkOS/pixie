#!/usr/bin/env bash
# Fast v3 chat worker on the Linux 3080 Ti. Restarts itself if it crashes;
# exits 3 (and stays down) if another instance already holds the lock port.
# Does not start if the live worker's lock (8793) is in use, unless
# PIXEL_FAST_ALLOW_BESIDE=1 is set (only for offline benches, never for polling).
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/worker.log"
PY="${PIXEL_PYTHON:-/path/to/pixie"
cd "$DIR"
while true; do
  "$PY" -u "$DIR/worker_v3.py" >>"$LOG" 2>&1
  st=$?
  if [ "$st" -eq 3 ]; then
    echo "$(date '+%F %T') another worker is running, not restarting" >>"$LOG"
    exit 3
  fi
  echo "$(date '+%F %T') worker exited $st, restarting in 5s" >>"$LOG"
  sleep 5
done
