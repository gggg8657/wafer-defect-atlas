"""Shared chart style: one palette, one set of mark specs, used by every figure.

Categorical hues are assigned in fixed slot order and never cycled; only the
first three slots are used, which is the number that validates across all pairs
for both normal and CVD vision.  Magnitude (confusion matrix, Grad-CAM heat) is
a single-hue light-to-dark ramp, never a rainbow -- a jet colormap makes a
Grad-CAM look more decisive than it is, because its hue jumps do not track the
underlying scalar.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # blue, orange, aqua
GRID = "#e6e5e1"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
SEQ_ORANGE = LinearSegmentedColormap.from_list(
    "seq_orange", ["#fcfcfb", "#fbd9c8", "#f5a97f", "#eb6834", "#a83f18", "#6b2708"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5,
})


def clean(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, alpha=0.9)
        ax.grid(axis="y" if grid == "x" else "x", visible=False)
    ax.tick_params(length=0)


def hbars(ax, ys, vals, color, height=0.62, label=None):
    """Horizontal bars, thin, with a surface gap between neighbours."""
    return ax.barh(ys, vals, height=height, color=color, linewidth=0, label=label)


def save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
