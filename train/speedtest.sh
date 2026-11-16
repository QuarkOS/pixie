#!/usr/bin/env bash
# Pixie v3 Linux speed test: pilot B config, scratch outputs only. One GPU job at a time (sequential).
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
P=$PIXIE_ROOT; V3=$P/v3; OUT=$V3/speedtest-linux; PY=$P/.venv/bin/python
mkdir -p $OUT
COMMON="-u $V3/code/train_v3.py --arch p50 --clip $V3/clip/clip_l14_text_l11_f32.pt --data $V3/data --cache $V3/cache \
 --palette $P/data/subset150k/palette.npy --images 8e6 --lr-min 1.5e-5 --warmup 1000 --wd 0.05 --layers 16 --width 512 \
 --heads 8 --dropout 0.1 --caption-dropout 0.1 --p-subj 0.175 --p-syn 0.1 --flip-p 0.5 --alpha-weight 0.25 \
 --eval-every-images 1e9 --save-every-images 1e9 --log-every 100 --max-hours 3.5 --sample-per 8 --no-samples --seed 0 --eval-limit 500"
run() { name=$1; shift; d=$OUT/$name
  if [ -e "$d" ]; then echo "skip $name (exists)"; return; fi
  echo "=== $name $(date)"; $PY $COMMON --out $d "$@" > $OUT/$name.log 2>&1; echo "exit $? $(date)"; }
run A_eager_b128      --batch 128 --lr 3e-4 --max-steps 3000
run B_compile_b128    --batch 128 --lr 3e-4 --max-steps 3000 --compile --nan-check-every 100
run E_compile_cudagraphs_b128 --batch 128 --lr 3e-4 --max-steps 2000 --compile --compile-mode reduce-overhead --nan-check-every 100
echo ALL_DONE
