"""Every figure in the README, from the JSON and checkpoints the runs wrote.

    CUDA_VISIBLE_DEVICES=1 python scripts/make_figures.py

Nothing here recomputes a metric: figures read the same JSON that report.py
turns into RESULTS.md, so a figure and the table beside it cannot disagree.
The one exception is the Grad-CAM panel, which needs the model to draw heat
maps -- the numbers under it still come from runs/gradcam.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from wafermap.viz import GRID, INK, INK2, INK3, SEQ, SEQ_ORANGE, SERIES, clean, save
from wafermap.wm811k import CLASSES_9, DEFECT_CLASSES

A = Path("assets")
R = Path("runs")


def jload(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def wafer_rgb(x, cam=None):
    """mask/fail channels -> RGB. Structure in ink, Grad-CAM heat in one hue."""
    mask, fail = x[0], x[1]
    img = np.ones((*mask.shape, 3))
    img[:] = np.array([0.988, 0.988, 0.984])
    on = mask > 0.05
    img[on] = np.array([0.87, 0.87, 0.855])
    f = np.clip(fail, 0, 1)[..., None] * on[..., None]
    img = img * (1 - f) + np.array([0.05, 0.05, 0.05]) * f
    if cam is not None:
        heat = SEQ_ORANGE(np.clip(cam, 0, 1))[..., :3]
        a = (np.clip(cam, 0, 1) ** 1.5 * 0.78)[..., None] * on[..., None]
        img = img * (1 - a) + heat * a
    return np.clip(img, 0, 1)


# ------------------------------------------------------------- dataset ----
def fig_dataset():
    prep = jload("data/proc/prepare.json")
    X = np.load("data/proc/labeled.npy", mmap_mode="r")
    m = np.load("data/proc/meta_labeled.npz", allow_pickle=False)
    te = np.flatnonzero(m["split"] == "test")
    y = m["y"][te]
    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 0.25, 2.2, 0.05],
                          hspace=0.28, wspace=0.12)
    for i, c in enumerate(CLASSES_9):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        sel = te[y == i]
        big = sel[np.argsort(-m["dieSize"][te][y == i])][:200]
        pick = big[len(big) // 2]
        ax.imshow(wafer_rgb(np.asarray(X[pick], dtype=np.float32) / 255))
        ax.set_title(c, fontsize=8.5, color=INK, pad=2)
        ax.axis("off")
    ax = fig.add_subplot(gs[:, 4])
    cnt = [prep["class_counts"][c] for c in CLASSES_9]
    order = np.argsort(cnt)
    # a dot plot, not bars: on a log axis a bar's length is not its value
    ax.hlines(np.arange(9), 80, [cnt[i] for i in order], color=GRID, lw=2)
    ax.plot([cnt[i] for i in order], np.arange(9), "o", color=SERIES[0], ms=8,
            markeredgecolor="none")
    ax.set_yticks(np.arange(9), [CLASSES_9[i] for i in order], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(80, 4e5)
    for j, i in enumerate(order):
        ax.text(cnt[i] * 1.25, j, f"{cnt[i]:,}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("labeled wafers (log)")
    ax.set_title(f"{prep['n_labeled']:,} labeled  ·  {prep['n_unlabeled']:,} unlabeled",
                 fontsize=9, color=INK, loc="left")
    clean(ax)
    fig.suptitle("WM-811K: one real test-split wafer per class, and the class imbalance",
                 fontsize=10.5, x=0.02, ha="left", color=INK)
    save(fig, A / "fig_dataset.png")


# ------------------------------------------------------------------ F1 ----
def fig_f1(runs):
    best = jload(R / "cls_best/eval.json")
    if best is None:
        return
    pc = best["test"]["multilabel_8"]["per_class"]
    order = sorted(DEFECT_CLASSES, key=lambda c: -pc[c]["support"])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    vals = [pc[c]["f1"] for c in order]
    ax.barh(np.arange(8), vals, height=0.62, color=SERIES[0], linewidth=0)
    ax.axvline(0.95, color=INK3, lw=1.2, ls=(0, (4, 3)))
    ax.set_ylim(8.0, -1.1)
    ax.text(0.95, -1.05, " KPI 0.95", fontsize=8, color=INK3, va="top")
    ax.set_yticks(np.arange(8), [f"{c}  ({pc[c]['support']:,})" for c in order],
                  fontsize=8)
    for i, v in enumerate(vals):
        ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("per-class F1 on held-out lots  (class, test support)")
    ax.set_title("Per-class F1 tracks ambiguity, not rarity", fontsize=9.5,
                 loc="left", color=INK)
    clean(ax)

    ax = axes[1]
    names, macro, weighted = [], [], []
    for tag, path in runs:
        e = jload(R / path / "eval.json")
        if e is None:
            continue
        names.append(tag)
        macro.append(e["test"]["multilabel_8"]["macro_f1"])
        weighted.append(e["test"]["class9"]["weighted_f1"])
    o = np.argsort(macro)
    yy = np.arange(len(o))
    ax.barh(yy + 0.17, [macro[i] for i in o], height=0.32, color=SERIES[0],
            linewidth=0, label="multi-label macro-F1 (8 defect classes)")
    ax.barh(yy - 0.17, [weighted[i] for i in o], height=0.32, color=SERIES[1],
            linewidth=0, label="9-class weighted-F1")
    for j, i in enumerate(o):
        ax.text(macro[i] + 0.008, j + 0.17, f"{macro[i]:.3f}", va="center",
                fontsize=7.5, color=INK2)
        ax.text(weighted[i] + 0.008, j - 0.17, f"{weighted[i]:.3f}", va="center",
                fontsize=7.5, color=INK2)
    ax.set_yticks(yy, [names[i] for i in o], fontsize=8)
    ax.set_xlim(0, 1.16)
    ax.set_xlabel("F1 on held-out lots")
    ax.set_title("Imbalance strategy, same budget and split", fontsize=9.5,
                 loc="left", color=INK)
    clean(ax)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="upper center", bbox_to_anchor=(0.76, 0.02),
               ncol=2, labelcolor=INK2, columnspacing=1.4)
    fig.subplots_adjust(top=0.90, bottom=0.22, wspace=0.55)
    save(fig, A / "fig_f1.png")


def fig_confusion():
    best = jload(R / "cls_best/eval.json")
    if best is None:
        return
    C = np.array(best["test"]["confusion"], dtype=float)
    Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.6, 4.9))
    im = ax.imshow(Cn, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(9), CLASSES_9, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(9), [f"{c}  ({int(C[i].sum()):,})"
                             for i, c in enumerate(CLASSES_9)], fontsize=8)
    for i in range(9):
        for j in range(9):
            if Cn[i, j] >= 0.005:
                ax.text(j, i, f"{Cn[i,j]*100:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if Cn[i, j] > 0.55 else INK2)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true  (test support)")
    ax.set_title("Row-normalized confusion, % — held-out lots", fontsize=9.5,
                 loc="left", color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03).outline.set_visible(False)
    save(fig, A / "fig_confusion.png")


# ------------------------------------------------------------ Grad-CAM ----
def fig_gradcam(run="runs/cls_best"):
    gc = jload(R / "gradcam.json")
    if gc is None:
        return
    import torch
    from wafermap.model import WaferResNet, grad_cam
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(Path(run) / "best.pt", map_location=dev)
    n_out = 9 if ck["args"]["loss"] == "ce" else 8
    model = WaferResNet(n_out=n_out, in_ch=2, width=ck["args"]["width"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    X = np.load("data/proc/labeled.npy", mmap_mode="r")
    m = np.load("data/proc/meta_labeled.npz", allow_pickle=False)
    te = np.flatnonzero(m["split"] == "test")
    y = m["y"][te]

    fig = plt.figure(figsize=(12.4, 5.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.28], hspace=0.30,
                             left=0.06, right=0.99, top=0.90, bottom=0.14)
    top = outer[0].subgridspec(1, 8, wspace=0.08)
    bot = outer[1].subgridspec(1, 2, wspace=0.26)
    for k, c in enumerate(DEFECT_CLASSES):
        sel = te[y == k + 1]
        xb = torch.from_numpy(np.asarray(X[sel], dtype=np.float32) / 255).to(dev)
        with torch.no_grad():
            o = model(xb).float()
            p = (o.softmax(1)[:, k + 1] if n_out == 9 else o.sigmoid()[:, k]).cpu().numpy()
        # the median-confidence *correctly* classified wafer -- a typical
        # success, not the best one in the split
        good = np.flatnonzero(p > 0.5)
        good = good if len(good) else np.argsort(-p)[:1]
        pick = good[np.argsort(p[good])[len(good) // 2]]
        x1 = xb[pick:pick + 1]
        cam = grad_cam(model, x1, cls=torch.tensor([k if n_out == 8 else k + 1],
                                                   device=dev))[0].cpu().numpy()
        ax = fig.add_subplot(top[0, k])
        ax.imshow(wafer_rgb(x1[0].cpu().numpy(), cam))
        ax.set_title(c, fontsize=8.5, color=INK, pad=3)
        ax.axis("off")
    fig.text(0.06, 0.955, "Grad-CAM on real WM-811K test wafers "
             "(median-confidence correct example per class)",
             fontsize=10.5, color=INK, ha="left")

    lab = list(gc["per_class"])
    xx = np.arange(len(lab))
    ax = fig.add_subplot(bot[0, 0])
    v = [gc["per_class"][c]["top10_cam_fail_lift"] for c in lab]
    ax.axhline(1.0, color=INK3, lw=1.2, ls=(0, (4, 3)))
    ax.vlines(xx, 1.0, v, color=GRID, lw=2)
    ax.plot(xx, v, "o", color=SERIES[0], ms=8, markeredgecolor="none")
    for i, s_ in enumerate(v):
        ax.text(i, s_ + 0.03, f"{s_:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.text(-0.5, 1.005, "chance = 1.0", fontsize=8, color=INK3, ha="left",
            va="bottom")
    ax.set_xticks(xx, lab, rotation=30, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(lab) - 0.4)
    ax.set_ylabel("fail density inside the top-10%\nCAM area / on the wafer")
    ax.set_title("Where the heat is", fontsize=9.5, loc="left", color=INK)
    clean(ax, grid="y")

    ax = fig.add_subplot(bot[0, 1])
    dc = [gc["per_class"][c]["deletion_drop_cam"] for c in lab]
    dr = [gc["per_class"][c]["deletion_drop_random"] for c in lab]
    ax.bar(xx - 0.17, dc, width=0.32, color=SERIES[0], linewidth=0,
           label="delete the top-10% CAM area")
    ax.bar(xx + 0.17, dr, width=0.32, color=SERIES[1], linewidth=0,
           label="delete a random 10%")
    ax.set_xticks(xx, lab, rotation=30, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(lab) - 0.4)
    ax.set_ylabel("drop in true-class probability")
    ax.set_title("Whether it is the evidence the model used", fontsize=9.5,
                 loc="left", color=INK)
    ax.legend(fontsize=8, labelcolor=INK2, loc="upper center")
    clean(ax, grid="y")
    save(fig, A / "fig_gradcam.png")


# ----------------------------------------------------------- clustering ----
def fig_cluster():
    cl = jload(R / "cluster.json")
    if cl is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9),
                             gridspec_kw={"width_ratios": [1, 1, 1.35]})
    names = [n for n in ("supervised", "simclr", "pixel_pca") if n in cl["encoders"]]
    nice = {"supervised": "supervised encoder", "simclr": "SimCLR (no labels)",
            "pixel_pca": "raw-pixel PCA"}
    for pi, (key, title) in enumerate([
            ("purity", "Cluster purity, all 9 classes"),
            ("purity_defect_only", "Purity on the 8 defect classes only")]):
        ax = axes[pi]
        for i, n in enumerate(names):
            ks = sorted(int(k) for k in cl["encoders"][n]["k"])
            v = [cl["encoders"][n]["k"][str(k)][key] for k in ks]
            ax.plot(ks, v, "-o", color=SERIES[i], label=nice[n])
            ax.text(ks[-1] * 1.06, v[-1], nice[n], fontsize=7.5, color=SERIES[i],
                    va="center")
        base = (cl["majority_baseline"] if key == "purity" else
                max(cl["eval_class_counts"][c] for c in list(cl["eval_class_counts"])[1:])
                / max(sum(list(cl["eval_class_counts"].values())[1:]), 1))
        ax.axhline(base, color=INK3, lw=1.2, ls=(0, (4, 3)))
        ax.text(ks[0], base + 0.012, "one-cluster baseline", fontsize=7.5, color=INK3)
        ax.set_xscale("log")
        ax.set_xticks(ks, [str(k) for k in ks], fontsize=8)
        ax.set_xlim(ks[0] * 0.85, ks[-1] * 2.4)
        ax.set_xlabel("k-means clusters, fitted on unlabeled maps")
        ax.set_ylabel("purity on held-out labeled lots")
        ax.set_title(title, fontsize=9.5, loc="left", color=INK)
        clean(ax, grid="y")

    ax = axes[2]
    enc = names[0]   # the encoder that works; the curves already show which
    k = max(int(x) for x in cl["encoders"][enc]["k"] if int(x) <= 32)
    rows = cl["encoders"][enc]["k"][str(k)]["clusters_defect_only"]
    M = np.array([r["counts"][1:] for r in rows], dtype=float)
    keep = M.sum(1) > 0
    M = M[keep] / np.maximum(M[keep].sum(1, keepdims=True), 1)
    o = np.argsort(-M.argmax(1) - M.max(1) * 0.001)
    im = ax.imshow(M[o], cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(8), DEFECT_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_ylabel(f"{int(keep.sum())} clusters (k={k}) holding a defect wafer")
    ax.set_title(f"{nice[enc]}: class mix inside each cluster", fontsize=9.5,
                 loc="left", color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03).outline.set_visible(False)
    save(fig, A / "fig_cluster.png")


# ------------------------------------------------------------------ SSL ----
def fig_ssl():
    lf = jload(R / "labelfrac.json")
    if lf is None:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    for i, (mode, nice) in enumerate([("scratch", "from scratch"),
                                      ("ssl", "SimCLR init, finetuned"),
                                      ("probe", "SimCLR, linear probe")]):
        if mode not in lf:
            continue
        fr = sorted(float(f) for f in lf[mode])
        v = [lf[mode][f"{f:g}"]["test_macro_f1"] for f in fr]
        ax.plot([f * 100 for f in fr], v, "-o", color=SERIES[i], label=nice)
        ax.text(fr[-1] * 100 * 1.05, v[-1], nice, fontsize=8, color=SERIES[i],
                va="center")
    ax.set_xscale("log")
    ticks = sorted({float(f) * 100 for m in lf for f in lf[m]})
    ax.set_xticks(ticks, [f"{t:g}%" for t in ticks], fontsize=8)
    ax.set_xlim(min(ticks) * 0.8, max(ticks) * 3.2)
    ax.set_xlabel("fraction of the 120,593 training labels used")
    ax.set_ylabel("test multi-label macro-F1")
    ax.set_title("Does pretraining on 447k unlabeled maps buy labels?",
                 fontsize=9.5, loc="left", color=INK)
    clean(ax, grid="y")
    save(fig, A / "fig_ssl.png")


if __name__ == "__main__":
    A.mkdir(exist_ok=True)
    runs = [("plain BCE", "cls_bce"), ("pos-weight 0.5", "cls_posw"),
            ("balanced sampling", "cls_bal"), ("focal + pos-weight", "cls_focal"),
            ("9-way softmax CE", "cls_ce"), ("BCE, no augmentation", "cls_bce_noaug")]
    which = sys.argv[1:] or ["dataset", "f1", "confusion", "gradcam", "cluster", "ssl"]
    if "dataset" in which:
        fig_dataset()
    if "f1" in which:
        fig_f1(runs)
    if "confusion" in which:
        fig_confusion()
    if "gradcam" in which:
        fig_gradcam()
    if "cluster" in which:
        fig_cluster()
    if "ssl" in which:
        fig_ssl()
