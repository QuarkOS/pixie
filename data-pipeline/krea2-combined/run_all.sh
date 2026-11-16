#!/usr/bin/env bash
# Resumable driver: extract (resume) each group, then finalize. Safe to re-run.
set -euo pipefail
R=$DATA_ROOT
PY=$PY
cd $R/code
for grp in "3080ti" "3060ti 5070ti"; do
  tag=${grp// /-}
  # wait if a previous extractor for this group is still running
  while pgrep -f "build_combined.py --only $grp " >/dev/null; do sleep 30; done
  if ! grep -q EXTRACT_DONE $R/logs/extract-$tag.log 2>/dev/null; then
    $PY build_combined.py --only $grp --resume --out-rows $R/work/cells-$tag.jsonl >> $R/logs/extract-$tag.log 2>&1
  fi
done
if [ ! -f $R/summary.json ]; then
  cat $R/work/cells-3080ti.jsonl $R/work/cells-3060ti-5070ti.jsonl > $R/work/cells.jsonl
  rm -rf "$R/final"  # only our own partial output from an interrupted finalize (summary.json missing)
  $PY finalize.py > $R/logs/finalize.log 2>&1
fi
echo ALL_DONE
