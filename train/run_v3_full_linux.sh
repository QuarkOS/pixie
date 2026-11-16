#!/usr/bin/env bash
# Pixie v3 FULL run on Linux, s7 recipe (Opus roadmap s7, the owner go 2026-11-11):
#   P50 d512xL16, 2x2 patches, CLIP-L penultimate, 16-id text (TL 18), 3 palette + 1 style cond slots,
#   rare-subject balanced sampling (n_s^-0.35, cap 6x), typo captions (v3/data_typo, --p-typo 0.15),
#   WSD: 1500-step warmup -> constant 3e-4, permanent full checkpoints at 18M / 27M / 36M views,
#   compile, TEXT_ARGS below (CLIP feature cache or online CLIP + CE mode, whichever benchmarked faster).
# Decay branch (separate, NOT automatic): v3/run_v3_decay18_linux.sh
# Previous (pre-s7) launcher: run_v3_full_linux.pre-s7.sh
#   PIXIE_ROOT=/path/to/pixie tmux new -d -s v3full "bash $PIXIE_ROOT/v3/run_v3_full_linux.sh"
set -u
: "${PIXIE_ROOT:?set PIXIE_ROOT}"
P=$PIXIE_ROOT; V3=$P/v3; OUT=$P/runs/v3-full-s7; PY=$P/.venv/bin/python
TEXT_ARGS="--clip-cache $V3/clipcache_s7_t18 --ce-chunks 1"   # cached CLIP-L features (9.60 step/s vs online 8.90); patch-order CE (4-way chunked was slower)
if [ -e "$OUT/train_summary.json" ]; then echo "already finished: $OUT"; exit 0; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then echo "GPU busy, not starting"; exit 1; fi
mkdir -p "$OUT"
echo "started $(date '+%a %m/%d/%Y %H:%M:%S').00 s7 full run" >> "$OUT/launch.txt"
$PY -u $V3/code/train_v3.py --arch p50 --clip $V3/clip/clip_l14_text_l11_f32.pt --data $V3/data_typo --cache $V3/cache \
  --palette $P/data/subset150k/palette.npy --out "$OUT" --images 36e6 --batch 128 --lr 3e-4 --lr-min 1e-5 \
  --schedule wsd --warmup 1500 --save-at-images 18e6,27e6,36e6 --wd 0.05 --layers 16 --width 512 --heads 8 --dropout 0.1 \
  --caption-dropout 0.1 --p-subj 0.175 --p-syn 0.1 --p-typo 0.15 --flip-p 0.5 --alpha-weight 0.25 \
  --text-len 18 --cond --balance-alpha 0.35 --balance-cap 6 $TEXT_ARGS \
  --eval-every-images 1e6 --save-every-images 1e6 --log-every 100 --sample-per 8 --seed 0 --compile --nan-check-every 100 \
  >> "$OUT/stdout.log" 2>> "$OUT/stderr.log"
rc=$?
echo "exit $rc $(date '+%a %m/%d/%Y %H:%M:%S').00" >> "$OUT/launch.txt"
