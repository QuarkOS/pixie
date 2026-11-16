#!/usr/bin/env bash
# Pixie v3 s7 COND BOOST fine-tune (operator go 2026-11-12 22:55 local):
#   Branch from the live 40M decayed model into a NEW dir. Does not touch any existing run.
#   last.pt model weights were checked equal to img_040M.pt (sha256 95a0fcde370de86b...).
#   last.pt is used because --branch-from needs model+optimizer (img_040M.pt is model-only).
#
#   Cond keep rates (mutually exclusive palette buckets, must sum to <= 1):
#     previous defaults:  p_pal_top1=0.25  p_pal_top3=0.25  (50% none)  p_style=0.40
#     this run:           p_pal_top1=0.30  p_pal_top3=0.50  (20% none)  p_style=0.70
#   Not 0.6/0.6/0.7: those palette probs sum to 1.2 and would break the none/top1/top3 split
#   in train_v3.py (k = 0 if r < 1-p1-p3, else 1 if r < 1-p3, else 3).
#
#   LR: short linear decay 3e-5 -> 1e-5 over 24M views (fine-tune; previous decay ended at 1e-5).
#   Same typo train set as s7 (Blind-50 and eval-50 already dropped in data_typo manifest).
#   24M / ~1,230 img/s ≈ 5.4 h plus evals. Chat is paused via DISABLE + systemctl --user stop;
#   this script always restarts chat on exit (success, failure, or signal).
#
#   tmux new -d -s condboost "bash $PIXIE_ROOT/v3/run_v3_condboost40_linux.sh"
set -u
P=$PIXIE_ROOT; V3=$P/v3; PY=$P/.venv/bin/python
SRC=${SRC:-$RUNS/v3-full-s7-decay36M/last.pt}
OUT=${OUT:-$RUNS/v3-full-s7-condboost40M}
DISABLE=$P/chat-worker-v3fast/DISABLE

if [ ! -e "$SRC" ]; then echo "missing $SRC"; exit 1; fi
if [ -e "$OUT/train_summary.json" ]; then echo "already finished: $OUT"; exit 0; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then echo "GPU busy, not starting"; exit 1; fi
mkdir -p "$OUT"
echo "started $(date '+%a %m/%d/%Y %H:%M:%S').00 condboost branch from $SRC  p_pal_top1=0.30 p_pal_top3=0.50 p_style=0.70  lr 3e-5->1e-5 decay 24e6" >> "$OUT/launch.txt"

restart_chat() {
  rm -f "$DISABLE"
  systemctl --user start pixie-chat
  echo "chat restart attempted $(date '+%a %m/%d/%Y %H:%M:%S').00" >> "$OUT/launch.txt"
}
trap restart_chat EXIT

TEXT_ARGS="--clip-cache $V3/clipcache_s7_t18 --ce-chunks 1"
$PY -u $V3/code/train_v3.py --arch p50 --clip $V3/clip/clip_l14_text_l11_f32.pt --data $V3/data_typo --cache $V3/cache \
  --palette $P/data/subset150k/palette.npy --out "$OUT" --branch-from "$SRC" --schedule decay --images 24e6 \
  --batch 128 --lr 3e-5 --lr-min 1e-5 --wd 0.05 --layers 16 --width 512 --heads 8 --dropout 0.1 \
  --caption-dropout 0.1 --p-subj 0.175 --p-syn 0.1 --p-typo 0.15 --flip-p 0.5 --alpha-weight 0.25 \
  --text-len 18 --cond --p-pal-top1 0.30 --p-pal-top3 0.50 --p-style 0.70 \
  --balance-alpha 0.35 --balance-cap 6 $TEXT_ARGS \
  --eval-every-images 1e6 --save-every-images 1e6 --save-at-images 48e6,56e6,64e6 \
  --log-every 100 --sample-per 8 --seed 2 --compile --nan-check-every 100 ${EXTRA:-} \
  >> "$OUT/stdout.log" 2>> "$OUT/stderr.log"
rc=$?
echo "exit $rc $(date '+%a %m/%d/%Y %H:%M:%S').00" >> "$OUT/launch.txt"
exit $rc
