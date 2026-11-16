#!/bin/bash
# usage: eval.sh   (final + step1000)
# Set KREA_LORA_ROOT, KREA_SPRITES, and KREA_EVAL_ROOT.
set -euo pipefail
: "${KREA_LORA_ROOT:?set KREA_LORA_ROOT}"
: "${KREA_SPRITES:?set KREA_SPRITES}"
: "${KREA_EVAL_ROOT:?set KREA_EVAL_ROOT}"
L=$KREA_LORA_ROOT; S=$KREA_SPRITES; T=$S/tools/fast; PY=$S/pyenv/bin/python
E=$KREA_EVAL_ROOT; LD=$E/sdloras
export LD_LIBRARY_PATH="$S/bin/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p $LD
cd $L/musubi-tuner
.venv/bin/python src/musubi_tuner/convert_lora.py --input $L/out/2x2_512/krea2-pixel32-sheet2x2.safetensors --output $LD/pix32-final.safetensors --target other
.venv/bin/python src/musubi_tuner/convert_lora.py --input $L/out/2x2_512/krea2-pixel32-sheet2x2-step00001000.safetensors --output $LD/pix32-s1000.safetensors --target other
cp $S/loras/checkpoint-1000.safetensors $LD/checkpoint-1000.safetensors
PORT=17891; BASE=http://127.0.0.1:$PORT
nohup $S/bin/sd-server --listen-ip 127.0.0.1 --listen-port $PORT \
  --diffusion-model $S/models/krea2_turbo-iq4_xs.gguf --llm $S/models/Qwen3-VL-4B-Instruct-Q4_K_M.gguf \
  --vae $S/models/wan_2.1_vae.safetensors --lora-model-dir $LD --lora-apply-mode at_runtime -v > $E/sd-server.log 2>&1 &
SP=$!; trap "kill $SP 2>/dev/null || true" EXIT
python3 $S/tools/sd_client.py --base $BASE --wait-ready
cd $T
run() { # name grid size px lora_file mult
  local n=$1 g=$2 sz=$3 px=$4 lf=$5 m=$6 d=$E/$1
  mkdir -p $d
  $PY make_jobs.py --grid $g --size $sz --steps 5 --lora $m --px $px --sheets 18 --warmup 1 --seed 1000 --outdir $d -o $d/jobs.json
  $PY - $d/jobs.json "$lf" <<'P'
import json,sys
j=json.load(open(sys.argv[1]))
for x in j:
    if sys.argv[2]=="none": x["no_lora"]=True
    else: x["lora"]=sys.argv[2]
json.dump(j,open(sys.argv[1],"w"),indent=1)
P
  $PY bench.py $d/jobs.json --base $BASE --depth 2 --warmup 1 --result $d/timing.json
  $PY extract.py $d/jobs.json $d/sprites --px $px | tee $d/extract.json
  echo "== $n done"
}
run final_m1.0 2 512 32 pix32-final.safetensors 1.0
run final_m0.8 2 512 32 pix32-final.safetensors 0.8
run s1000_m1.0 2 512 32 pix32-s1000.safetensors 1.0
run angel16 2 512 16 checkpoint-1000.safetensors 1.0
run base1024 2 1024 32 none 0
echo EVAL_DONE
