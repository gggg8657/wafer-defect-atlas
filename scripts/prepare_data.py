"""WM-811K -> normalized tensors + a lot-level split, written once to data/proc/.

    CUDA_VISIBLE_DEVICES=1 python scripts/prepare_data.py

Writes
  data/proc/labeled.npy      (172950, 2, 64, 64) uint8   quantized [0,1] density
  data/proc/unlabeled.npy    (638507, 2, 64, 64) uint8
  data/proc/meta_labeled.npz  y, lot, split, dieSize, H, W
  data/proc/meta_unlabeled.npz
  data/proc/prepare.json      every count/check this script measured
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wafermap.wm811k import CLASSES_9, IMG, load_raw, normalize_maps, split_lots


def main():
    import torch

    out = Path("data/proc")
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    df = load_raw()
    print(f"raw: {df.shape} in {time.time()-t0:.1f}s", flush=True)
    lab = np.array([_unwrap(x) for x in df["failureType"].to_numpy()])
    lot = df["lotName"].astype(str).to_numpy().astype(str)
    die = df["dieSize"].to_numpy().astype(np.int32)
    maps = df["waferMap"].to_numpy()
    H = np.array([m.shape[0] for m in maps], dtype=np.int16)
    W = np.array([m.shape[1] for m in maps], dtype=np.int16)

    is_lab = lab != ""
    cls_idx = {c: i for i, c in enumerate(CLASSES_9)}
    y = np.array([cls_idx[c] if c else -1 for c in lab], dtype=np.int8)

    # ---- lot-level split, decided on ALL wafers at once -------------------
    # 6,077 lots hold both labeled and unlabeled wafers, so the split has to be
    # a single partition of lots; otherwise a lot's unlabeled wafers could sit
    # in the clustering pool while its labeled siblings are the purity probe.
    lots_by_class = {c: set(lot[y == i]) for i, c in enumerate(CLASSES_9)}
    assign = split_lots(lots_by_class, set(lot.tolist()), seed=0)
    split = np.array([assign[l] for l in lot])

    rep = {"raw_rows": int(df.shape[0]), "n_lots": int(len(set(lot.tolist()))),
           "n_labeled": int(is_lab.sum()), "n_unlabeled": int((~is_lab).sum()),
           "img": IMG, "device": dev,
           "distinct_source_shapes": int(len({(int(a), int(b)) for a, b in zip(H, W)})),
           "source_H": [int(H.min()), int(np.median(H)), int(H.max())],
           "source_W": [int(W.min()), int(np.median(W)), int(W.max())],
           "class_counts": {c: int((y == i).sum()) for i, c in enumerate(CLASSES_9)}}

    # ---- normalize + store ------------------------------------------------
    for name, sel in [("labeled", is_lab), ("unlabeled", ~is_lab)]:
        idx = np.flatnonzero(sel)
        t = time.time()
        X = normalize_maps(list(maps[idx]), size=IMG, device=dev)
        cov = float(X[:, 0].mean())
        np.save(out / f"{name}.npy", np.round(X * 255).astype(np.uint8))
        np.savez(out / f"meta_{name}.npz", y=y[idx], lot=lot[idx], split=split[idx],
                 dieSize=die[idx], H=H[idx], W=W[idx], row=idx.astype(np.int64))
        rep[name] = {
            "n": int(len(idx)), "resize_s": round(time.time() - t, 1),
            "mean_coverage": round(cov, 4),
            "mean_coverage_over_pi_4": round(cov / (np.pi / 4), 4),
            "mean_fail_rate_on_wafer": round(float(X[:, 1].sum() / X[:, 0].sum()), 5),
            "split_counts": {s: int((split[idx] == s).sum())
                             for s in ("train", "val", "test")},
        }
        print(name, rep[name], flush=True)

    li = np.flatnonzero(is_lab)
    rep["labeled_split_by_class"] = {
        c: {s: int(((y[li] == i) & (split[li] == s)).sum())
            for s in ("train", "val", "test")} for i, c in enumerate(CLASSES_9)}
    rep["lot_split_counts"] = {s: int(sum(v == s for v in assign.values()))
                               for s in ("train", "val", "test")}
    rep["wall_s"] = round(time.time() - t0, 1)
    (out / "prepare.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["labeled_split_by_class"], indent=2))
    print("wrote data/proc/prepare.json in", rep["wall_s"], "s")


def _unwrap(x):
    while isinstance(x, (list, np.ndarray)):
        if len(x) == 0:
            return ""
        x = x[0]
    return str(x)


if __name__ == "__main__":
    main()
