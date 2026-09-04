"""Does Grad-CAM actually localize *real* WM-811K defects?

    CUDA_VISIBLE_DEVICES=1 python scripts/eval_gradcam.py --run runs/cls_best

The synthetic demo could check the CAM against a defect region it drew itself.
Real wafer maps come with no localization ground truth, so four measurements
stand in, each with an explicit chance level:

  1. fail-die lift  -- CAM mass on failed dies / failed fraction of the wafer.
     Chance = 1.0.  Says the heat sits on failures rather than on the wafer.
  2. radial agreement -- CAM-weighted mean radius vs fail-die-weighted mean
     radius, per class and correlated over wafers.  Separates "on failures"
     from "on the *right* failures": Center and Edge-Ring differ only in radius.
  3. centroid error -- |CAM centroid - fail centroid| in wafer radii, against
     the null of always answering the wafer centre.
  4. deletion test -- zero the fail dies under the top-10% CAM area and watch
     the true-class probability fall, versus deleting a random 10% of the
     wafer.  This one is causal: it asks whether the highlighted dies are the
     evidence the model used, not merely where it looked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wafermap.model import WaferResNet, grad_cam
from wafermap.wm811k import CLASSES_9, DEFECT_CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/cls_best")
    p.add_argument("--out", default="runs/gradcam.json")
    p.add_argument("--per-class", type=int, default=400)
    p.add_argument("--topk", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    dev = "cuda"
    rng = np.random.default_rng(a.seed)

    ck = torch.load(Path(a.run) / "best.pt", map_location=dev)
    ca = ck["args"]
    n_out = 9 if ca["loss"] == "ce" else 8
    model = WaferResNet(n_out=n_out, in_ch=2, width=ca["width"]).to(dev).eval()
    model.load_state_dict(ck["model"])

    X = np.load("data/proc/labeled.npy", mmap_mode="r")
    m = np.load("data/proc/meta_labeled.npz", allow_pickle=False)
    te = np.flatnonzero(m["split"] == "test")
    y = m["y"][te]

    yy, xx = np.mgrid[0:64, 0:64]
    cy = cx = 31.5
    r_grid = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / 32.0

    res, curves = {}, {}
    for ci, cls in enumerate(DEFECT_CLASSES, start=1):
        sel = te[y == ci]
        if len(sel) > a.per_class:
            sel = rng.choice(sel, a.per_class, replace=False)
        xb = torch.from_numpy(np.ascontiguousarray(X[np.sort(sel)])).float().to(dev) / 255
        tgt = torch.full((len(xb),), ci - 1 if n_out == 8 else ci, device=dev)
        cams = []
        for i in range(0, len(xb), 128):
            cams.append(grad_cam(model, xb[i:i + 128], cls=tgt[i:i + 128]).cpu())
        cam = torch.cat(cams).numpy().astype(np.float64)

        mask = xb[:, 0].cpu().numpy().astype(np.float64)
        fail = xb[:, 1].cpu().numpy().astype(np.float64)
        camm = cam * (mask > 0.05)                    # only score on-wafer heat
        w = camm.sum((1, 2)) + 1e-9
        on_fail = (camm * fail).sum((1, 2)) / w
        chance = (fail * (mask > 0.05)).sum((1, 2)) / ((mask > 0.05).sum((1, 2)) + 1e-9)
        lift = on_fail / np.maximum(chance, 1e-6)

        r_cam = (camm * r_grid).sum((1, 2)) / w
        fw = (fail * (mask > 0.05)).sum((1, 2)) + 1e-9
        r_fail = (fail * (mask > 0.05) * r_grid).sum((1, 2)) / fw
        ok = fw > 1e-6
        corr = float(np.corrcoef(r_cam[ok], r_fail[ok])[0, 1]) if ok.sum() > 3 else float("nan")

        cy_cam = (camm * yy).sum((1, 2)) / w
        cx_cam = (camm * xx).sum((1, 2)) / w
        cy_f = (fail * yy).sum((1, 2)) / fw
        cx_f = (fail * xx).sum((1, 2)) / fw
        d_cam = np.hypot(cy_cam - cy_f, cx_cam - cx_f) / 32.0
        d_null = np.hypot(cy - cy_f, cx - cx_f) / 32.0

        # ---- deletion test -------------------------------------------------
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            base = model(xb).float()
            base = (base.softmax(1)[:, ci] if n_out == 9 else base.sigmoid()[:, ci - 1])
        camt = torch.from_numpy(camm).to(dev)
        flat = camt.flatten(1)
        k = max(1, int(a.topk * 64 * 64))
        thr = flat.topk(k, dim=1).values[:, -1:, None]
        del_mask = (camt >= thr).float()
        rnd = torch.zeros_like(del_mask).flatten(1)
        ridx = torch.rand(rnd.shape, device=dev).argsort(1)[:, :k]
        rnd.scatter_(1, ridx, 1.0)
        rnd = rnd.view_as(del_mask)
        outs = {}
        for nm, msk in (("cam", del_mask), ("random", rnd)):
            xd = xb.clone()
            xd[:, 1] = xd[:, 1] * (1 - msk)
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                o = model(xd).float()
                o = (o.softmax(1)[:, ci] if n_out == 9 else o.sigmoid()[:, ci - 1])
            outs[nm] = o
        drop_cam = float((base - outs["cam"]).mean())
        drop_rnd = float((base - outs["random"]).mean())

        res[cls] = {
            "n": int(len(xb)),
            "fail_die_lift": float(lift.mean()),
            "cam_mass_on_fail": float(on_fail.mean()),
            "fail_area_fraction": float(chance.mean()),
            "r_cam": float(r_cam.mean()), "r_fail": float(r_fail.mean()),
            "r_corr": corr,
            "centroid_err_radii": float(d_cam.mean()),
            "centroid_err_null_radii": float(d_null.mean()),
            "p_true_base": float(base.mean()),
            "p_true_after_cam_deletion": float(outs["cam"].mean()),
            "p_true_after_random_deletion": float(outs["random"].mean()),
            "deletion_drop_cam": drop_cam,
            "deletion_drop_random": drop_rnd,
            "deletion_ratio": drop_cam / max(drop_rnd, 1e-6),
        }
        curves[cls] = {"r_cam": r_cam.tolist()[:400], "r_fail": r_fail.tolist()[:400]}
        print(cls, json.dumps(res[cls]), flush=True)

    agg = {k: float(np.mean([res[c][k] for c in res]))
           for k in ("fail_die_lift", "centroid_err_radii", "centroid_err_null_radii",
                     "deletion_drop_cam", "deletion_drop_random")}
    agg["r_corr_over_classes"] = float(np.corrcoef(
        [res[c]["r_cam"] for c in res], [res[c]["r_fail"] for c in res])[0, 1])
    out = {"run": a.run, "topk": a.topk, "per_class": res, "mean": agg}
    Path(a.out).write_text(json.dumps(out, indent=2))
    Path(a.out).with_suffix(".curves.json").write_text(json.dumps(curves))
    print("mean", json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
