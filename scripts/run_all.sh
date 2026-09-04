#!/usr/bin/env bash
# Everything, in order, on GPUs 1-3.  ~2 h wall clock.
set -e
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/pdeno/bin/python

CUDA_VISIBLE_DEVICES=1 $PY scripts/prepare_data.py
bash scripts/sweep_imbalance.sh            # 6 configs over 3 GPUs -> runs/cls_*
CUDA_VISIBLE_DEVICES=1 $PY scripts/pretrain_simclr.py --out runs/ssl
bash scripts/sweep_labelfrac.sh            # SSL vs scratch at 1/5/25/100% labels
CUDA_VISIBLE_DEVICES=1 $PY scripts/eval_gradcam.py --run runs/cls_best
CUDA_VISIBLE_DEVICES=1 $PY scripts/cluster_atlas.py --cls-run runs/cls_best
CUDA_VISIBLE_DEVICES=1 $PY scripts/make_figures.py
$PY scripts/report.py
