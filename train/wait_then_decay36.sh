#!/usr/bin/env bash
# Waits for the v3-full-s7 main run to finish cleanly, then starts the 36M decay branch (one GPU job at a time).
# Never touches the main run. Log: $PIXIE_ROOT/logs/wait_decay36.log
set -u
P=$PIXIE_ROOT; MAIN=$RUNS/v3-full-s7; LOG=$P/logs/wait_decay36.log
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "waiter started (pid $$)"
while pgrep -f -- "train_v3.py .*--out $RUNS/v3-full-s7( |$)" >/dev/null; do sleep 60; done
say "main training process gone"
sleep 30
if [ ! -e "$MAIN/train_summary.json" ] || [ ! -e "$MAIN/full_036M.pt" ]; then
  say "NOT launching: train_summary.json or full_036M.pt missing (main run did not finish cleanly)"; ls -la "$MAIN" | tail -5 >> "$LOG"; exit 1
fi
# wait until the GPU has no compute processes for 3 checks in a row (max ~30 min)
ok=0; for i in $(seq 1 180); do
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then ok=0; else ok=$((ok+1)); fi
  [ $ok -ge 3 ] && break; sleep 10; done
if [ $ok -lt 3 ]; then say "NOT launching: GPU still busy after 30 min"; nvidia-smi >> "$LOG"; exit 1; fi
say "launching decay36 from $MAIN/full_036M.pt"
bash $P/v3/run_v3_decay36_linux.sh
say "decay36 script exited rc=$? ; $(tail -1 $RUNS/v3-full-s7-decay36M/launch.txt 2>/dev/null)"
