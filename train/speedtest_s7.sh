#!/usr/bin/env bash
# s7 speed + loss-parity tests, sequential (one GPU job at a time). Scratch outputs: v3/speedtest-s7/<name>
set -u
: "${PIXIE_ROOT:?set PIXIE_ROOT}"
P=$PIXIE_ROOT; V3=$P/v3; PY=$P/.venv/bin/python; ROOT=$V3/speedtest-s7; mkdir -p $ROOT
COMMON="--arch p50 --clip $V3/clip/clip_l14_text_l11_f32.pt --data $V3/data_typo --cache $V3/cache --palette $P/data/subset150k/palette.npy
 --images 36e6 --batch 128 --lr 3e-4 --lr-min 1e-5 --wd 0.05 --layers 16 --width 512 --heads 8 --dropout 0.1 --caption-dropout 0.1
 --p-subj 0.175 --p-syn 0.1 --p-typo 0.15 --flip-p 0.5 --alpha-weight 0.25 --log-every 100 --seed 0 --compile --nan-check-every 100
 --no-samples --eval-limit 256 --eval-every-images 1e12 --save-every-images 1e12"
S7="--text-len 18 --cond --balance-alpha 0.35 --balance-cap 6"
CACHE="--clip-cache $V3/clipcache_s7_t18"
run() { n=$1; shift; [ -e $ROOT/$n/summary_max_steps.json ] && { echo "skip $n"; return; }
  echo "== $n $(date +%T)"; $PY -u $V3/code/train_v3.py $COMMON --out $ROOT/$n "$@" > $ROOT/$n.log 2>&1; echo "rc $? $(date +%T)"
  grep -E "step (500|1000)/|clip cache" $ROOT/$n.log | cut -c1-200; }
for t in ${TESTS:-r0 s1 s2 s3 s4 s5}; do case $t in
  r0) run r0-old-online --max-steps 1000 --warmup 1000 ;;                                   # old model def on new loop (parity vs 2.4556)
  s1) run s1-s7-online-ce1 --max-steps 1000 --warmup 1000 $S7 --ce-chunks 1 ;;
  s2) run s2-s7-online-ce4 --max-steps 1000 --warmup 1000 $S7 --ce-chunks 4 ;;
  s3) run s3-s7-cached-ce1 --max-steps 1000 --warmup 1000 $S7 --ce-chunks 1 $CACHE ;;
  s4) run s4-s7-cached-ce4 --max-steps 1000 --warmup 1000 $S7 --ce-chunks 4 $CACHE ;;
  s5) run s5-s7-online-ce1-static --max-steps 600 --warmup 1000 $S7 --ce-chunks 1 --bench-static ;;
  s6) run s6-s7-cached-ce1-static --max-steps 600 --warmup 1000 $S7 --ce-chunks 1 $CACHE --bench-static ;;
  r1) run r1-old-online-static --max-steps 600 --warmup 1000 --bench-static ;;
esac; done
