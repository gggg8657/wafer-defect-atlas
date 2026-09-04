#!/usr/bin/env bash
# Class imbalance is the whole problem here (9,680 Edge-Ring vs 149 Near-full),
# so it gets a controlled sweep rather than a guess: same architecture, same
# 40-epoch budget, same lot-level split, one knob changed at a time.
set -e
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/pdeno/bin/python
E=${1:-40}
run () { CUDA_VISIBLE_DEVICES=$1 $PY scripts/train_cls.py --epochs $E --out runs/$2 ${@:3} \
         > runs/$2.log 2>&1 & }
mkdir -p runs
run 1 cls_bce      --loss bce
run 2 cls_posw     --loss bce   --posweight-power 0.5
run 3 cls_bal      --loss bce   --sample-power 0.5
wait
run 1 cls_focal    --loss focal --posweight-power 0.5
run 2 cls_ce       --loss ce    --sample-power 0.5
run 3 cls_bce_noaug --loss bce  --aug 0
wait
echo "sweep done"
