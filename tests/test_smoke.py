"""Smoke test: classifier learns 4 patterns, Grad-CAM localizes a center defect."""
import numpy as np
import torch
from wafermap import train, grad_cam, make_map, CLASSES


def test_classifier_learns():
    model, acc, _ = train(epochs=15, verbose=False)
    assert acc > 0.85, f"accuracy too low: {acc}"


def test_gradcam_localizes_center():
    model, _, _ = train(epochs=15, verbose=False)
    N = 32
    img = make_map("center", N, np.random.default_rng(3))
    x = torch.tensor(img)[None, None]
    cam = grad_cam(model, x, cls=torch.tensor([CLASSES.index("center")]))[0].numpy()
    ys, xs = np.mgrid[0:N, 0:N]
    w = cam / (cam.sum() + 1e-8)
    cy, cx = (w * ys).sum(), (w * xs).sum()
    dist = np.hypot(cy - N / 2, cx - N / 2)
    assert dist < N * 0.25, f"Grad-CAM did not localize center: {dist:.1f}px"


if __name__ == "__main__":
    test_classifier_learns()
    test_gradcam_localizes_center()
    print("ok")
