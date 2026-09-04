#!/usr/bin/env bash
# The whole real-data pipeline, in order, on GPUs 1-3.  ~2 h wall clock.
set -e
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/pdeno/bin/python

CUDA_VISIBLE_DEVICES=1 $PY scripts/prepare_data.py        # LSWMD.pkl -> data/proc
bash scripts/sweep_imbalance.sh 60                        # 6 strategies, 2 waves
bash scripts/train_final.sh                               # 3 seeds, 150 epochs
$PY scripts/select_best.py                                # val-only -> runs/cls_best

CUDA_VISIBLE_DEVICES=1 $PY scripts/pretrain_simclr.py --out runs/ssl --epochs 25 --width 64
bash scripts/sweep_labelfrac.sh                           # SSL vs scratch vs probe
$PY scripts/collect_labelfrac.py

CUDA_VISIBLE_DEVICES=1 $PY scripts/eval_gradcam.py --run runs/cls_best --per-class 400
CUDA_VISIBLE_DEVICES=1 $PY scripts/cluster_atlas.py --cls-run runs/cls_best --ssl runs/ssl/ssl.pt

# why the label-free encoder loses to raw pixels: ablate the view-invariances
for a in "d4:d4" "geom:d4,affine" "noise:d4,noise"; do
  CUDA_VISIBLE_DEVICES=1 $PY scripts/pretrain_simclr.py --out runs/ssl_${a%%:*} \
    --epochs 25 --width 64 --aug ${a#*:}
done
CUDA_VISIBLE_DEVICES=1 $PY scripts/cluster_atlas.py --cls-run /none --ssl /none --no-pixel \
  --ks 32 128 --out runs/cluster_aug.json \
  --extra d4=runs/ssl_d4/ssl.pt geom=runs/ssl_geom/ssl.pt noise=runs/ssl_noise/ssl.pt

CUDA_VISIBLE_DEVICES=1 $PY scripts/make_figures.py
$PY scripts/report.py                                     # RESULTS.md + README.md
