#!/bin/bash
: "${KREA_LORA_ROOT:?set KREA_LORA_ROOT}"
mkdir -p "$KREA_LORA_ROOT/logs"
setsid nohup "$KREA_LORA_ROOT/bin/eval.sh" > "$KREA_LORA_ROOT/logs/eval.log" 2>&1 < /dev/null &
echo started
