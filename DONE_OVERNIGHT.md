# Overnight run — 2026-09-04

`wafer-defect-atlas` went from a synthetic demo to a result on the real
WM-811K corpus. Every number in `README.md` and `RESULTS.md` is substituted by
`scripts/report.py` from a JSON one of the runs below wrote; both files are
regenerated, not edited, and `report.py` fails loudly if a `{{hole}}` in
`README.tmpl.md` has no JSON behind it.

## What I ran

| stage | script | cost |
|---|---|---|
| preprocess 811,457 maps + lot-level split | `scripts/prepare_data.py` | 575 s, 1 GPU |
| class-imbalance sweep, 6 strategies | `scripts/sweep_imbalance.sh` | 2 waves x 7 min, 3 GPUs |
| final model, 3 seeds x 150 epochs | `scripts/train_final.sh` | 18 min, 3 GPUs |
| SimCLR on 447,573 unlabeled maps | `scripts/pretrain_simclr.py` | 21 min, 1 GPU |
| label-fraction grid, 12 cells | `scripts/sweep_labelfrac.sh` | 20 min, 2 GPUs |
| Grad-CAM validation, 2,054 test wafers | `scripts/eval_gradcam.py` | 2 min |
| clustering KPI, 3 encoders x 5 k | `scripts/cluster_atlas.py` | 12 min |
| SimCLR augmentation ablation, 3 variants | `scripts/pretrain_simclr.py --aug` | 24 min, 3 GPUs |

Total ~2 h wall clock on GPUs 1–3. GPU 0 was never touched.

## What it showed

**Classification — the KPI is met on one reading and not on the other.**
Multi-label macro-F1 over the eight failure signatures is **0.897** on 25,897
wafers from held-out lots; 9-class weighted-F1 is **0.980** and accuracy
**0.9807**. Weighted-F1 clears the ≥0.95 target and macro-F1 does not. The
shortfall is three classes — Loc 0.794, Scratch 0.795, Edge-Loc 0.877 — and the
confusion matrix says they leak into `none` (15% / 18% / 12%), not into each
other: weak, low-contrast instances of a pattern the label asserts and the map
barely shows.

**The split is by lot.** 46,293 lots partitioned 70/15/15, labeled and unlabeled
wafers together (6,077 lots contain both). This is the single decision that
makes the numbers comparable to anything; a row-level split on this dataset puts
near-duplicate wafers from one lot on both sides.

**Class imbalance is a null result, and the seeds prove it.** Pos-weighting,
focal loss, balanced sampling and a 9-way softmax span 0.016 in test macro-F1,
while three seeds of a single configuration span 0.012. The one knob outside the
noise is the d4 augmentation: removing it costs 0.042.

**Grad-CAM localizes real defects, measured causally.** Deleting the failed dies
under the top-10% CAM area drops true-class probability 0.389 against 0.083 for
a random 10% (4.7x). Fail density inside the top-10% CAM area is 1.41x the
wafer's own; the CAM peak is on a failed die 46% of the time against 30% chance.
One measure of the four — CAM centroid vs failed-die centroid — is *worse* than
the trivial null, because background failures drag the target centroid to the
wafer centre. It is reported.

**The clustering KPI splits in two.** k-means on all 447,573 unlabeled train-lot
maps, centroids frozen, purity scored on 25,897 labeled wafers from held-out
lots: **0.948** defect-only purity with the supervised encoder, against 0.416
chance. The protocol works. But that encoder was trained on these eight classes
and cannot surface a ninth. The label-free encoder that could reaches **0.623**,
*below* the 0.777 raw-pixel PCA floor.

**SimCLR does not work here, and the ablation says why it is not the view set.**
Pretraining helps at 1% labels (0.699 vs 0.677 from scratch), does nothing at
100% (0.882 vs 0.889, inside seed noise), and its frozen representation tops out
at 0.612 under a linear probe against 0.889 finetuned. Ablating the
augmentations: the affine translate/scale is the costly invariance (dropping it
takes k=128 purity 0.623 → 0.686) and the Bernoulli die resampling is the
helpful one — but all four variants lose to raw-pixel PCA, so the deficit is the
objective, not the augmentation set.

## What is left

- **Closing the macro-F1 gap.** The three ambiguous classes are the whole
  shortfall. Candidates, in the order I would try them: higher input resolution
  (a scratch is a few dies wide and 64×64 blurs it), a deeper encoder, and
  MixedWM38 to give the compound cases a label of their own. None of these is
  tuning against the test split, which is why none of them was done here.
- **A working label-free encoder.** Contrastive learning with hand-picked
  invariances is the wrong tool for a signal that *is* position and morphology.
  Masked reconstruction of the fail field, or contrasting wafers *within* a lot
  rather than augmented copies of one wafer, are the two I would try next.
- **Genuine unknown-pattern discovery.** Everything measured here scores
  clusters against the eight known classes. The clusters that hold real
  unlabeled mass with no labeled defect in them are recorded per-cluster in
  `runs/cluster.json` (`n_unlabeled`, `n_clusters_none_dominated`) but nobody has
  looked at what is inside them. That is a human-in-the-loop task.
- **The unlabeled val/test pools** (94,704 / 96,230 maps) are prepared and
  unused. They exist for a semi-supervised run — pseudo-labeling or FixMatch —
  which is the approach most likely to move Loc/Scratch without new labels.

## What needs the owner's decision

1. **Which F1 the KPI means.** The catalog says "다중라벨 F1 ≥ 0.95". Weighted-F1
   is 0.980 and macro-F1 is 0.897; both are defensible readings of "the F1" and
   they disagree about whether the target is met. The README states both and
   marks macro-F1 as the honest one, but if the KPI was written meaning the
   weighted number, the goal is already met and the remaining work is the
   clustering half.
2. **Whether to pull in MixedWM38.** It is the natural fix for the compound
   Loc/Edge-Loc/Scratch cases and it makes the multi-label framing real rather
   than nominal, but it is a second dataset and a second download.
3. **Whether the supervised 0.948 counts as the "미지패턴 군집 순도" KPI.** It is a
   real measurement under a clean protocol, but it demonstrably cannot discover
   an unknown pattern. I have reported it as the protocol working and the
   unknown-pattern half failing; if the KPI is meant strictly, that half is not
   met and needs the semi-supervised work above.

## Notes for whoever runs this next

- `data/` is gitignored: 1.6 GB raw plus 6.6 GB of preprocessed uint8 tensors.
  The host had 83 GB free at the start.
- `MiniBatchKMeans` dumps core on this 192-core box unless the BLAS pool is
  capped before numpy loads; `scripts/cluster_atlas.py` sets it at import.
- Installed into the `pdeno` env: pandas, scikit-learn, scipy. torch 2.8.0+cu128
  and numpy 2.4.6 were pinned with a constraints file and are unchanged.
- CI is green: `tests/test_smoke.py` (synthetic CPU demo) and
  `tests/test_wm811k.py` (real-data path, CPU, no dataset needed) both pass.
