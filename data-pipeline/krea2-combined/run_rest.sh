#!/usr/bin/env bash
set -e
R=$DATA_ROOT
until grep -q EXTRACT_DONE $R/logs/extract-3080ti.log; do sleep 30; done
cd $R/code
$PY build_combined.py --only 3060ti 5070ti --out-rows $R/work/cells-3060ti-5070ti.jsonl > $R/logs/extract-3060ti-5070ti.log 2>&1
