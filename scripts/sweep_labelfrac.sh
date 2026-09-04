#!/usr/bin/env bash
# Does SimCLR on 447k unlabeled maps buy labels?
# Every cell gets the SAME optimizer budget (100 steps x 60 epochs = 6,000
# steps) regardless of how many labels it sees, so the comparison is about the
# initialization and not about how long each cell trained.
set -e
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/pdeno/bin/python
BASE="${BASE:---loss bce}"
SSL=runs/ssl/ssl.pt
mkdir -p runs/labelfrac
run () { CUDA_VISIBLE_DEVICES=$1 $PY scripts/train_cls.py --epochs 60 \
         --steps-per-epoch 100 --width 64 $BASE --label-frac $2 --out runs/labelfrac/$3 ${@:4} \
         > runs/labelfrac/$3.log 2>&1 & }
for f in 0.01 0.05 0.25 1.0; do
  t=$(echo $f | tr -d '.')
  run 1 $f scratch_$t
  run 2 $f ssl_$t     --init $SSL
  run 3 $f probe_$t   --init $SSL --freeze 1
  wait
done
echo "labelfrac done"
