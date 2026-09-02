"""Synthetic wafer defect maps.

Draws the canonical WM-811K-style spatial failure patterns (none / center /
edge-ring / scratch) on a circular wafer, as fail-probability maps with noise.
The real project loads WM-811K (811k maps) + MixedWM38; the synthetic set has
the same spatial signature so the model + Grad-CAM pipeline runs with no download.
"""
from __future__ import annotations
import numpy as np

CLASSES = ["none", "center", "edge_ring", "scratch"]


def _wafer_mask(N):
    y, x = np.mgrid[-1:1:N * 1j, -1:1:N * 1j]
    return (x ** 2 + y ** 2) <= 1.0, x, y


def make_map(cls, N=32, rng=None):
    rng = rng or np.random.default_rng()
    mask, x, y = _wafer_mask(N)
    r = np.sqrt(x ** 2 + y ** 2)
    base = 0.03 * np.ones((N, N))  # background fail rate
    if cls == "center":
        base += 0.8 * np.exp(-(r ** 2) / 0.06)
    elif cls == "edge_ring":
        base += 0.8 * np.exp(-((r - 0.85) ** 2) / 0.01)
    elif cls == "scratch":
        ang = rng.uniform(0, np.pi)
        d = np.abs(np.cos(ang) * x + np.sin(ang) * y)
        base += 0.85 * np.exp(-(d ** 2) / 0.002)
    p = np.clip(base, 0, 1) * mask
    fail = (rng.uniform(size=(N, N)) < p) & mask
    img = fail.astype(np.float32)
    img[~mask] = 0.0
    return img


def make_dataset(n_per_class=250, N=32, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        for _ in range(n_per_class):
            X.append(make_map(c, N, rng))
            y.append(ci)
    X = np.stack(X)[:, None, :, :]  # (n, 1, N, N)
    return X.astype(np.float32), np.array(y)
