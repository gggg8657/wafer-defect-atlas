"""CPU tests for the real-data path -- no WM-811K download required.

These cover the three places the pipeline could silently produce a wrong number:
the ragged-map normalization (must preserve fail *density*), the lot-level split
(must not put one lot on both sides), and the threshold/decode logic that turns
eight sigmoid scores into one of nine classes.
"""
import numpy as np
import torch

from wafermap.metrics import aggregate, decode_9, per_class_f1, tune_thresholds
from wafermap.model import WaferResNet, grad_cam
from wafermap.wm811k import CLASSES_9, normalize_maps, split_lots, unwrap


def test_unwrap_matches_dataset_encoding():
    assert unwrap(np.array([["Center"]], dtype=object)) == "Center"
    assert unwrap([]) == ""
    assert unwrap([[]]) == ""


def test_area_resize_preserves_fail_density():
    """A wafer map is a density, so the mean must survive the resize."""
    rng = np.random.default_rng(0)
    maps = []
    for h, w in [(25, 27), (49, 39), (128, 96), (16, 16)]:
        m = np.zeros((h, w), dtype=np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        disc = ((yy / (h - 1) - .5) ** 2 + (xx / (w - 1) - .5) ** 2) <= .25
        m[disc] = 1
        m[disc & (rng.random((h, w)) < 0.3)] = 2
        maps.append(m)
    X = normalize_maps(maps, size=64)
    assert X.shape == (4, 2, 64, 64)
    for m, x in zip(maps, X):
        src = (m == 2).mean() / max((m > 0).mean(), 1e-9)
        dst = x[1].sum() / max(x[0].sum(), 1e-9)
        assert abs(src - dst) < 0.06, f"fail density moved: {src:.3f} -> {dst:.3f}"


def test_lot_split_is_a_partition_and_covers_rare_classes():
    lots = {f"lot{i}" for i in range(300)}
    by_class = {c: {f"lot{i}" for i in range(k, 300, 17)}
                for k, c in enumerate(CLASSES_9)}
    a = split_lots(by_class, lots, seed=0)
    assert set(a) == lots                       # every lot placed exactly once
    parts = {s: {l for l, v in a.items() if v == s} for s in ("train", "val", "test")}
    assert parts["train"] & parts["val"] == set()
    assert parts["train"] & parts["test"] == set()
    assert parts["val"] & parts["test"] == set()
    for c, ls in by_class.items():              # rare classes reach every split
        assert all(ls & parts[s] for s in parts), c


def test_thresholds_and_decode_round_trip():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 9, 4000)
    Y = np.zeros((len(y), 8), dtype=np.int8)
    Y[np.flatnonzero(y > 0), y[y > 0] - 1] = 1
    probs = np.clip(Y * 0.75 + rng.normal(0, 0.12, Y.shape) + 0.1, 0, 1)
    ths = tune_thresholds(probs, Y)
    pred = decode_9(probs, ths)
    rows = per_class_f1(y, pred, 9)
    assert aggregate(rows)["macro_f1"] > 0.8
    # all-zero scores must decode to `none`, never to a defect
    assert (decode_9(np.zeros((5, 8)), np.full(8, 0.5)) == 0).all()


def test_waferresnet_grad_cam_shapes():
    m = WaferResNet(n_out=8, in_ch=2, width=8)
    x = torch.rand(3, 2, 64, 64)
    assert m(x).shape == (3, 8)
    assert m.embed(x).shape == (3, m.n_feat)
    cam = grad_cam(m, x, cls=torch.tensor([0, 1, 2]))
    assert cam.shape == (3, 64, 64)
    assert float(cam.min()) >= 0.0 and abs(float(cam.amax()) - 1.0) < 1e-5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
