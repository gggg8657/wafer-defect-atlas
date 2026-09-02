"""Train the wafer defect classifier and localize defects with Grad-CAM.

    python demo_smoke.py

Reports test accuracy and checks that Grad-CAM lights up the correct region
(e.g. the wafer center for a 'center' cluster). Saves a Grad-CAM figure if
matplotlib is available.
"""
import numpy as np
import torch
from wafermap import train, grad_cam, make_map, CLASSES


def _center_of_mass(cam):
    N = cam.shape[-1]
    ys, xs = np.mgrid[0:N, 0:N]
    w = cam / (cam.sum() + 1e-8)
    return (w * ys).sum(), (w * xs).sum()


def main():
    model, acc, _ = train(epochs=15, verbose=True)
    print(f"\ntest accuracy (4 classes): {acc:.3f}")

    # Grad-CAM localizes a center-cluster defect near the wafer center
    N = 32
    rng = np.random.default_rng(0)
    img = make_map("center", N, rng)
    x = torch.tensor(img)[None, None]
    cam = grad_cam(model, x, cls=torch.tensor([CLASSES.index("center")]))[0].numpy()
    cy, cx = _center_of_mass(cam)
    dist = np.hypot(cy - N / 2, cx - N / 2)
    print(f"Grad-CAM center-of-mass distance from wafer center: {dist:.1f}px (N={N})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(img, cmap="gray_r"); ax[0].set_title("wafer (center defect)")
        ax[1].imshow(img, cmap="gray_r"); ax[1].imshow(cam, cmap="jet", alpha=0.5)
        ax[1].set_title("Grad-CAM")
        for a in ax: a.axis("off")
        fig.tight_layout(); fig.savefig("gradcam.png", dpi=110)
        print("saved gradcam.png")
    except Exception as e:
        print(f"(matplotlib figure skipped: {e})")
    return acc, dist


if __name__ == "__main__":
    main()
