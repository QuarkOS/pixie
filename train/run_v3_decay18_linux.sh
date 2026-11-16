#!/usr/bin/env bash
# Pixie v3 s7 DECAY BRANCH (Opus s7 #1): 2M views, linear LR 3e-4 -> 1e-5, starting from the main run's
# permanent 18M checkpoint (model + optimizer). Writes ONLY to its own new folder; the main run is untouched.
# NOT started automatically. Needs the 3080 Ti free (one GPU job at a time: run it after the main run has
# finished or is stopped; not on the 3060 Ti, which serves chat). ~2M / ~1,150 img/s = ~30 min + evals.
#   PIXIE_ROOT=/path/to/pixie tmux new -d -s v3decay "bash $PIXIE_ROOT/v3/run_v3_decay18_linux.sh"
# Other branch points: SRC=.../full_027M.pt OUT=.../v3-full-s7-decay27M bash run_v3_decay18_linux.sh
set -u
: "${PIXIE_ROOT:?set PIXIE_ROOT}"
P=$PIXIE_ROOT; V3=$P/v3; PY=$P/.venv/bin/python
SRC=${SRC:-$P/runs/v3-full-s7/full_018M.pt}
OUT=${OUT:-$P/runs/v3-full-s7-decay18M}
if [ ! -e "$SRC" ]; then echo "missing $SRC (main run has not reached it yet)"; exit 1; fi
if [ -e "$OUT/train_summary.json" ]; then echo "already finished: $OUT"; exit 0; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then echo "GPU busy, not starting"; exit 1; fi
mkdir -p "$OUT"
echo "started $(date '+%a %m/%d/%Y %H:%M:%S').00 branch from $SRC" >> "$OUT/launch.txt"
TEXT_ARGS="--clip-cache $V3/clipcache_s7_t18 --ce-chunks 1"   # cached CLIP-L features (9.60 step/s vs online 8.90); patch-order CE (4-way chunked was slower)
$PY -u $V3/code/train_v3.py --arch p50 --clip $V3/clip/clip_l14_text_l11_f32.pt --data $V3/data_typo --cache $V3/cache \
  --palette $P/data/subset150k/palette.npy --out "$OUT" --branch-from "$SRC" --schedule decay --images 2e6 \
  --batch 128 --lr 3e-4 --lr-min 1e-5 --wd 0.05 --layers 16 --width 512 --heads 8 --dropout 0.1 \
  --caption-dropout 0.1 --p-subj 0.175 --p-syn 0.1 --p-typo 0.15 --flip-p 0.5 --alpha-weight 0.25 \
  --text-len 18 --cond --balance-alpha 0.35 --balance-cap 6 $TEXT_ARGS \
  --eval-every-images 5e5 --save-every-images 1e6 --log-every 100 --sample-per 8 --seed 1 --compile --nan-check-every 100 ${EXTRA:-} \
  >> "$OUT/stdout.log" 2>> "$OUT/stderr.log"
rc=$?
echo "exit $rc $(date '+%a %m/%d/%Y %H:%M:%S').00" >> "$OUT/launch.txt"
