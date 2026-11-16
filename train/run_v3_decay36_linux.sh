#!/usr/bin/env bash
# Pixie v3 s7 DECAY BRANCH from the FINAL 36M checkpoint (operator go 2026-11-11 ~23:00): 4M views (~11% of 36M,
# same ratio as the 18M branch's 2M/18M), linear LR 3e-4 -> 1e-5, from runs/v3-full-s7/full_036M.pt (model + optimizer).
# Copy of run_v3_decay18_linux.sh with only SRC / OUT / --images changed. Writes ONLY to runs/v3-full-s7-decay36M.
# Needs the 3080 Ti free. Started automatically by v3/wait_then_decay36.sh (tmux v3decay) after the main run finishes.
# 4M / ~1,230 img/s = ~55 min incl. 8 evals (every 0.5M) + final samples.
set -u
P=$PIXIE_ROOT; V3=$P/v3; PY=$P/.venv/bin/python
SRC=${SRC:-$RUNS/v3-full-s7/full_036M.pt}
OUT=${OUT:-$RUNS/v3-full-s7-decay36M}
if [ ! -e "$SRC" ]; then echo "missing $SRC (main run has not reached it yet)"; exit 1; fi
if [ -e "$OUT/train_summary.json" ]; then echo "already finished: $OUT"; exit 0; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then echo "GPU busy, not starting"; exit 1; fi
mkdir -p "$OUT"
echo "started $(date '+%a %m/%d/%Y %H:%M:%S').00 branch from $SRC" >> "$OUT/launch.txt"
TEXT_ARGS="--clip-cache $V3/clipcache_s7_t18 --ce-chunks 1"   # cached CLIP-L features (9.60 step/s vs online 8.90); patch-order CE (4-way chunked was slower)
$PY -u $V3/code/train_v3.py --arch p50 --clip $V3/clip/clip_l14_text_l11_f32.pt --data $V3/data_typo --cache $V3/cache \
  --palette $P/data/subset150k/palette.npy --out "$OUT" --branch-from "$SRC" --schedule decay --images 4e6 \
  --batch 128 --lr 3e-4 --lr-min 1e-5 --wd 0.05 --layers 16 --width 512 --heads 8 --dropout 0.1 \
  --caption-dropout 0.1 --p-subj 0.175 --p-syn 0.1 --p-typo 0.15 --flip-p 0.5 --alpha-weight 0.25 \
  --text-len 18 --cond --balance-alpha 0.35 --balance-cap 6 $TEXT_ARGS \
  --eval-every-images 5e5 --save-every-images 1e6 --log-every 100 --sample-per 8 --seed 1 --compile --nan-check-every 100 ${EXTRA:-} \
  >> "$OUT/stdout.log" 2>> "$OUT/stderr.log"
rc=$?
echo "exit $rc $(date '+%a %m/%d/%Y %H:%M:%S').00" >> "$OUT/launch.txt"
