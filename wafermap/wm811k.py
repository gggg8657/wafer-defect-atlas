"""WM-811K: loading the Python-2-era pickle, normalizing ragged maps, lot-level splits.

The raw file `data/LSWMD.pkl` holds 811,457 wafer maps as a pandas DataFrame
pickled under Python 2, so it needs a module shim to unpickle at all.  Each
`waferMap` is a 2-D uint8 array over the *die grid* of that product:
0 = outside the wafer, 1 = die passed, 2 = die failed.  Grids are ragged --
632 distinct shapes from 6x3 to 300x205 -- which is the first thing a model
needs solved.

Normalization (the decision this file exists to make explicit):

  every map is area-resized to a fixed 64x64 with two channels,
      ch0 = coverage  = fraction of the source cell that is on-wafer (map > 0)
      ch1 = fail rate = fraction of the source cell that failed  (map == 2)

Area resizing, not nearest/bilinear, because a wafer map is a *density*: the
mean of ch1 over the wafer is the wafer's fail rate at any output resolution,
so downsampling a 300x205 map and upsampling a 25x27 one both stay physically
meaningful.  The resize is anisotropic (H and W are scaled independently) on
purpose: the die grid spans the wafer's bounding box, so stretching it to a
square maps the physical wafer disc onto the *same* circle in every sample --
a scale/aspect normalization for free.  Verified in `prepare_data.py`: mean
coverage over the labeled set lands at pi/4, the area of the unit disc in its
bounding square.
"""
from __future__ import annotations

import pickle
import sys
import types
from pathlib import Path

import numpy as np

# 9 classes: the 8 WM-811K failure signatures plus the defect-free majority.
CLASSES_9 = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
             "Loc", "Near-full", "Random", "Scratch"]
DEFECT_CLASSES = CLASSES_9[1:]          # the 8 the multi-label head predicts
IMG = 64                                 # normalized map size


def load_raw(path="data/LSWMD.pkl"):
    """Unpickle the Python-2 pandas DataFrame (needs the pandas.indexes shim)."""
    import pandas.core.indexes as pci

    shim = types.ModuleType("pandas.indexes")
    shim.base = pci.base
    sys.modules["pandas.indexes"] = shim
    sys.modules["pandas.indexes.base"] = pci.base
    sys.modules["pandas.indexes.range"] = pci.range
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def unwrap(x):
    """Labels ship as [[ 'Center' ]]; unlabeled rows as []. -> str, '' if absent."""
    while isinstance(x, (list, np.ndarray)):
        if len(x) == 0:
            return ""
        x = x[0]
    return str(x)


def normalize_maps(maps, size=IMG, device="cpu", batch=4096):
    """Ragged list of uint8 die grids -> float32 (n, 2, size, size) in [0, 1].

    Maps are grouped by source shape so each distinct grid is resized as one
    batched `area` interpolation instead of 811k one-off kernel launches.
    """
    import torch
    import torch.nn.functional as F

    n = len(maps)
    out = np.empty((n, 2, size, size), dtype=np.float32)
    by_shape: dict[tuple, list[int]] = {}
    for i, m in enumerate(maps):
        by_shape.setdefault(m.shape, []).append(i)
    for shape, idx in by_shape.items():
        for s in range(0, len(idx), batch):
            chunk = idx[s:s + batch]
            arr = np.stack([maps[i] for i in chunk])          # (b, H, W) uint8
            t = torch.from_numpy(arr).to(device)
            x = torch.stack([(t > 0).float(), (t == 2).float()], 1)  # (b, 2, H, W)
            if shape == (size, size):
                r = x
            else:
                r = F.interpolate(x, size=(size, size), mode="area")
            out[chunk] = r.clamp_(0, 1).cpu().numpy()
    return out


def split_lots(lots_by_class, all_lots, seed=0, frac=(0.70, 0.15, 0.15)):
    """Assign whole lots to train/val/test.

    Wafers from one lot share process history, so a row-level split leaks:
    near-duplicate maps land on both sides and the score is inflated.  Lots are
    assigned rarest-class-first so that Near-full (149 wafers over 137 lots)
    still reaches every split; whatever is left (mostly unlabeled-only lots)
    is split at random.
    """
    rng = np.random.default_rng(seed)
    assign: dict[str, str] = {}
    names = ["train", "val", "test"]
    for cls in sorted(lots_by_class, key=lambda c: len(lots_by_class[c])):
        pool = np.array(sorted(l for l in lots_by_class[cls] if l not in assign))
        if len(pool) == 0:
            continue
        rng.shuffle(pool)
        cuts = np.cumsum([int(round(f * len(pool))) for f in frac[:2]])
        for name, part in zip(names, np.split(pool, cuts)):
            for l in part:
                assign[l] = name
    rest = np.array(sorted(set(all_lots) - set(assign)))
    rng.shuffle(rest)
    cuts = np.cumsum([int(round(f * len(rest))) for f in frac[:2]])
    for name, part in zip(names, np.split(rest, cuts)):
        for l in part:
            assign[l] = name
    return assign
