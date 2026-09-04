"""F1 bookkeeping for a 9-class problem where one class holds 85% of the mass.

Accuracy is close to useless on WM-811K -- predicting `none` for everything
scores 0.852 -- so everything here is per-class F1 first and aggregates second.
"""
from __future__ import annotations

import numpy as np


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(tp + fn, 1e-9)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def per_class_f1(y_true, y_pred, n_class):
    """Returns list of dicts, one per class index."""
    out = []
    for c in range(n_class):
        t, p_ = y_true == c, y_pred == c
        tp, fp, fn = int((t & p_).sum()), int((~t & p_).sum()), int((t & ~p_).sum())
        p, r, f = prf(tp, fp, fn)
        out.append({"support": int(t.sum()), "precision": p, "recall": r, "f1": f,
                    "tp": tp, "fp": fp, "fn": fn})
    return out


def aggregate(rows, subset=None):
    """macro / weighted / micro F1 over `rows` (optionally a class subset)."""
    sel = rows if subset is None else [rows[i] for i in subset]
    sup = np.array([r["support"] for r in sel], dtype=float)
    f1 = np.array([r["f1"] for r in sel])
    tp = sum(r["tp"] for r in sel)
    fp = sum(r["fp"] for r in sel)
    fn = sum(r["fn"] for r in sel)
    return {"macro_f1": float(f1.mean()),
            "weighted_f1": float((f1 * sup).sum() / max(sup.sum(), 1e-9)),
            "micro_f1": prf(tp, fp, fn)[2]}


def tune_thresholds(probs, Y, grid=None):
    """Per-class decision threshold maximizing that class's F1 on the VAL split.

    One threshold per class, not a shared 0.5: the eight defect signatures span
    9,680 to 149 examples, and a single cut sacrifices the rare classes to keep
    the common ones' precision.  Fitted on val, then frozen and applied to test.
    """
    grid = np.linspace(0.02, 0.95, 94) if grid is None else grid
    ths = []
    for c in range(probs.shape[1]):
        best, bt = -1.0, 0.5
        t_c = Y[:, c].astype(bool)
        for t in grid:
            p_ = probs[:, c] >= t
            f = prf(int((t_c & p_).sum()), int((~t_c & p_).sum()), int((t_c & ~p_).sum()))[2]
            if f > best:
                best, bt = f, float(t)
        ths.append(bt)
    return np.array(ths)


def decode_9(probs, ths):
    """Multi-label sigmoid scores -> one of 9 classes (0 = none).

    A wafer is `none` when no defect head clears its own threshold; otherwise
    the winner is the head that clears its threshold by the widest margin, so
    heads with different operating points stay comparable.
    """
    margin = probs - ths[None, :]
    fired = margin.max(1) >= 0
    return np.where(fired, margin.argmax(1) + 1, 0)


def multilabel_f1(probs, Y, ths):
    """Per-defect-class F1 in the one-vs-rest (multi-label) sense."""
    rows = []
    for c in range(probs.shape[1]):
        t, p_ = Y[:, c].astype(bool), probs[:, c] >= ths[c]
        tp, fp, fn = int((t & p_).sum()), int((~t & p_).sum()), int((t & ~p_).sum())
        p, r, f = prf(tp, fp, fn)
        rows.append({"support": int(t.sum()), "precision": p, "recall": r, "f1": f,
                     "tp": tp, "fp": fp, "fn": fn, "threshold": float(ths[c])})
    return rows
