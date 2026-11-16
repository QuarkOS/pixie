#!/usr/bin/env bash
# Foreground supervisor: sd-server, dashboard, runner. Safe for systemd Type=simple.
# Writes only under this directory (on the HDD). Does not touch Pixie.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SPRITES="${KREA2_SPRITES:-}"
if [[ -z "$SPRITES" ]]; then echo "set KREA2_SPRITES to the sd.cpp install (bin/, models/, pyenv/)" >&2; exit 1; fi
PY="${KREA2_PYTHON:-$SPRITES/pyenv/bin/python}"
SD="${KREA2_SD_SERVER:-$SPRITES/bin/sd-server}"
MODELS="$SPRITES/models"
cd "$ROOT"
mkdir -p "$ROOT/data" "$ROOT/logs" "$ROOT/loras"

if [[ ! -x "$SD" ]]; then echo "missing sd-server: $SD" >&2; exit 1; fi
if [[ ! -x "$PY" ]]; then echo "missing python: $PY" >&2; exit 1; fi
if [[ ! -f "$ROOT/loras/pix32-final.safetensors" ]]; then
  echo "missing LoRA $ROOT/loras/pix32-final.safetensors" >&2; exit 1
fi

export LD_LIBRARY_PATH="$SPRITES/bin/lib:${LD_LIBRARY_PATH:-}"
export CUDA_MODULE_LOADING=LAZY
export CUDA_DEVICE_ORDER=PCI_BUS_ID

PORT=17890
DPORT=8765
if [[ -f "$ROOT/config.json" ]]; then
  PORT="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("port",17890))' "$ROOT/config.json")"
  DPORT="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("dashboard_port",8765))' "$ROOT/config.json")"
fi

alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

stop_pid() {
  local pid="${1:-}"
  if alive "$pid"; then
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      alive "$pid" || return 0
      sleep 0.3
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

write_pids() {
  printf '{"server":%s,"dashboard":%s,"runner":%s}\n' \
    "${SRV_PID:-0}" "${DASH_PID:-0}" "${RUN_PID:-0}" > "$ROOT/data/pids.json"
}

start_server() {
  local stamp; stamp="$(date +%Y%m%d-%H%M%S)"
  # Bind loopback only. Logs stay on the HDD.
  nohup "$SD" \
    --listen-ip 127.0.0.1 --listen-port "$PORT" \
    --diffusion-model "$MODELS/krea2_turbo-iq4_xs.gguf" \
    --llm "$MODELS/Qwen3-VL-4B-Instruct-Q4_K_M.gguf" \
    --vae "$MODELS/wan_2.1_vae.safetensors" \
    --lora-model-dir "$ROOT/loras" --lora-apply-mode at_runtime \
    --disable-image-metadata --conditioning-cache-size 64 \
    > "$ROOT/logs/sd-server-$stamp.log" 2>&1 &
  SRV_PID=$!
  echo "$SRV_PID" > "$ROOT/data/server.pid"
}

start_dashboard() {
  nohup "$PY" "$ROOT/app/dashboard.py" --port "$DPORT" --data "$ROOT/data" \
    > "$ROOT/logs/dashboard.log" 2>&1 &
  DASH_PID=$!
  echo "$DASH_PID" > "$ROOT/data/dashboard.pid"
}

start_runner() {
  nohup "$PY" "$ROOT/app/runner.py" --config "$ROOT/config.json" --data "$ROOT/data" \
    > "$ROOT/logs/runner.log" 2>&1 &
  RUN_PID=$!
  echo "$RUN_PID" > "$ROOT/data/runner.pid"
}

cleanup() {
  stop_pid "${RUN_PID:-}"
  stop_pid "${SRV_PID:-}"
  stop_pid "${DASH_PID:-}"
  rm -f "$ROOT/data/server.pid" "$ROOT/data/dashboard.pid" "$ROOT/data/runner.pid" "$ROOT/data/pids.json"
}
trap cleanup EXIT INT TERM

# If a previous run left our pids, stop only those (never pkill, never touch Pixie).
if [[ -f "$ROOT/data/pids.json" ]]; then
  "$PY" - "$ROOT/data/pids.json" <<'PY'
import json, os, signal, sys, time
p=json.load(open(sys.argv[1]))
for k,pid in p.items():
    pid=int(pid or 0)
    if pid<=0: continue
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(1)
for k,pid in p.items():
    pid=int(pid or 0)
    if pid<=0: continue
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
PY
fi

start_server
start_dashboard
start_runner
write_pids
echo "started server=$SRV_PID dashboard=$DASH_PID runner=$RUN_PID"
echo "dashboard http://127.0.0.1:$DPORT"
echo "data $ROOT/data"

# Watchdog: restart sd-server if it dies while the runner is still going.
# After the queue finishes, free the GPU and keep the dashboard until we are stopped.
while alive "$RUN_PID"; do
  if ! alive "$SRV_PID"; then
    echo "$(date --iso-8601=seconds) sd-server exited; restarting" | tee -a "$ROOT/logs/watchdog.log"
    start_server
    write_pids
  fi
  sleep 5
done
echo "$(date --iso-8601=seconds) runner finished; stopping sd-server (GPU free), dashboard stays up" | tee -a "$ROOT/logs/watchdog.log"
stop_pid "$SRV_PID"; SRV_PID=0
write_pids
# Wait until systemd/stop.sh signals us.
while alive "$DASH_PID"; do sleep 30; done
