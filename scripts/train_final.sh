#!/usr/bin/env bash
# The val-selected imbalance strategy, trained on a longer schedule at a wider
# encoder, three seeds -- one per H100.  Three seeds because a 0.01 difference
# in macro-F1 between two strategies means nothing until you know what a rerun
# of the *same* strategy costs.
set -e
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/pdeno/bin/python
mkdir -p runs/final
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((s+1)) $PY scripts/train_cls.py --loss bce --epochs 150 \
    --width 64 --seed $s --out runs/final/seed$s > runs/final/seed$s.log 2>&1 &
done
wait
echo "final done"
