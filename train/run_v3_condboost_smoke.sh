#!/usr/bin/env bash
# Short GPU smoke for the condboost recipe. Writes ONLY to v3/smoke-s7/condboost-300.
# Does not create or touch runs/v3-full-s7-condboost40M. Does not restart chat (caller owns that).
set -u
P=$PIXIE_ROOT; V3=$P/v3; PY=$P/.venv/bin/python
SRC=${SRC:-$RUNS/v3-full-s7-decay36M/last.pt}
OUT=${OUT:-$V3/smoke-s7/condboost-300}
if [ ! -e "$SRC" ]; then echo "missing $SRC"; exit 1; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then echo "GPU busy, not starting"; exit 1; fi
# Fresh smoke dir only; never resume a leftover
if [ -e "$OUT" ]; then
  echo "refusing to overwrite existing $OUT"; exit 1
fi
mkdir -p "$OUT"
echo "started $(date '+%a %m/%d/%Y %H:%M:%S').00 condboost smoke 300 steps from $SRC" >> "$OUT/launch.txt"
TEXT_ARGS="--clip-cache $V3/clipcache_s7_t18 --ce-chunks 1"
$PY -u $V3/code/train_v3.py --arch p50 --clip $V3/clip/clip_l14_text_l11_f32.pt --data $V3/data_typo --cache $V3/cache \
  --palette $P/data/subset150k/palette.npy --out "$OUT" --branch-from "$SRC" --schedule decay --images 24e6 \
  --batch 128 --lr 3e-5 --lr-min 1e-5 --wd 0.05 --layers 16 --width 512 --heads 8 --dropout 0.1 \
  --caption-dropout 0.1 --p-subj 0.175 --p-syn 0.1 --p-typo 0.15 --flip-p 0.5 --alpha-weight 0.25 \
  --text-len 18 --cond --p-pal-top1 0.30 --p-pal-top3 0.50 --p-style 0.70 \
  --balance-alpha 0.35 --balance-cap 6 $TEXT_ARGS \
  --eval-every-images 1e6 --save-every-images 1e6 --save-at-images 48e6,56e6,64e6 \
  --log-every 50 --sample-per 8 --seed 2 --compile --nan-check-every 1 \
  --max-steps 300 --eval-limit 64 --no-samples \
  >> "$OUT/stdout.log" 2>> "$OUT/stderr.log"
rc=$?
echo "exit $rc $(date '+%a %m/%d/%Y %H:%M:%S').00" >> "$OUT/launch.txt"
exit $rc
