"""Train the WM-811K defect classifier on the lot-level split.

    CUDA_VISIBLE_DEVICES=1 python scripts/train_cls.py --out runs/cls_bce

The task is posed as **multi-label**: eight independent sigmoid heads, one per
failure signature, and a defect-free wafer is the all-zero target.  WM-811K
happens to carry at most one label per wafer, but the sigmoid formulation is
what generalizes to mixed-type wafers (MixedWM38) and it is what makes
"per-class F1" mean the same thing for a class with 9,680 examples and one with
149.  `--loss ce` swaps in a 9-way softmax as the control.

Everything measured lands in <out>/eval.json; nothing is printed to be copied.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wafermap.metrics import (aggregate, decode_9, multilabel_f1, per_class_f1,
                              tune_thresholds)
from wafermap.model import WaferResNet
from wafermap.wm811k import CLASSES_9, DEFECT_CLASSES


# ---------------------------------------------------------------- data ----
def load_split(root="data/proc", splits=("train", "val", "test")):
    X = np.load(Path(root) / "labeled.npy", mmap_mode="r")
    m = np.load(Path(root) / "meta_labeled.npz", allow_pickle=False)
    out = {}
    for s in splits:
        idx = np.flatnonzero(m["split"] == s)
        out[s] = (np.ascontiguousarray(X[idx]), m["y"][idx].astype(np.int64),
                  m["lot"][idx])
    return out


def d4(x, g):
    """One element of the dihedral group, applied to a whole batch.

    All eight WM-811K signatures are defined by shape, not orientation -- a
    scratch is a scratch at any angle -- so the d4 group is label-preserving
    and needs no interpolation, unlike a free rotation.
    """
    if g & 4:
        x = torch.flip(x, [-1])
    k = g & 3
    return torch.rot90(x, k, (-2, -1)) if k else x


# ------------------------------------------------------------- training ----
def evaluate(model, Xg, y, bs=1024, n_out=8, loss="bce"):
    model.eval()
    ps = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for i in range(0, len(Xg), bs):
            logit = model(Xg[i:i + bs].float() / 255.0).float()
            ps.append((logit.softmax(1)[:, 1:] if loss == "ce"
                       else logit.sigmoid()).cpu())
    probs = torch.cat(ps).numpy().astype(np.float64)
    Y = np.zeros((len(y), 8), dtype=np.int8)
    d = y > 0
    Y[np.flatnonzero(d), y[d] - 1] = 1
    return probs, Y


def report(probs, y, Y, ths):
    """Every number the README can quote, from one set of predictions."""
    ml = multilabel_f1(probs, Y, ths)
    pred9 = decode_9(probs, ths)
    rows9 = per_class_f1(y, pred9, 9)
    defect = y > 0
    conf = np.zeros((9, 9), dtype=int)
    for t, p in zip(y, pred9):
        conf[t, p] += 1
    return {
        "multilabel_8": {"per_class": {c: r for c, r in zip(DEFECT_CLASSES, ml)},
                         **aggregate(ml)},
        "class9": {"per_class": {c: r for c, r in zip(CLASSES_9, rows9)},
                   **aggregate(rows9),
                   "accuracy": float((pred9 == y).mean())},
        "defect_only_8": aggregate(rows9, subset=list(range(1, 9))),
        "any_defect": {k: v for k, v in zip(
            ("precision", "recall", "f1"),
            __import__("wafermap.metrics", fromlist=["prf"]).prf(
                int((defect & (pred9 > 0)).sum()), int((~defect & (pred9 > 0)).sum()),
                int((defect & (pred9 == 0)).sum())))},
        "confusion": conf.tolist(),
        "thresholds": {c: float(t) for c, t in zip(DEFECT_CLASSES, ths)},
        "n": int(len(y)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/cls")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loss", default="bce", choices=["bce", "ce", "focal"])
    p.add_argument("--posweight-power", type=float, default=0.0,
                   help="pos_weight_c = (n_neg/n_pos)^power, 0 = off")
    p.add_argument("--sample-power", type=float, default=0.0,
                   help="resample p ~ count^-power, 0 = natural, 1 = balanced")
    p.add_argument("--aug", type=int, default=1)
    p.add_argument("--label-frac", type=float, default=1.0)
    p.add_argument("--init", default="", help="SSL checkpoint to start from")
    p.add_argument("--freeze", type=int, default=0, help="linear probe only")
    p.add_argument("--steps-per-epoch", type=int, default=0,
                   help="fix the optimizer budget instead of letting it scale "
                        "with the label count -- required for label-fraction "
                        "comparisons, where otherwise 1%% of labels also means "
                        "1%% of the gradient steps")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    t0 = time.time()

    data = load_split()
    Xtr, ytr, _ = data["train"]
    if args.label_frac < 1.0:
        rng = np.random.default_rng(args.seed)
        keep = []
        for c in range(9):                     # stratified subsample of labels
            ci = np.flatnonzero(ytr == c)
            keep.append(rng.choice(ci, max(1, int(round(len(ci) * args.label_frac))),
                                   replace=False))
        keep = np.sort(np.concatenate(keep))
        Xtr, ytr = Xtr[keep], ytr[keep]
    Xg = torch.from_numpy(Xtr).to(dev)
    yg = torch.from_numpy(ytr).to(dev)
    Yg = torch.zeros(len(ytr), 8, device=dev)
    Yg[torch.arange(len(ytr), device=dev)[yg > 0], yg[yg > 0] - 1] = 1.0
    Xva = torch.from_numpy(data["val"][0]).to(dev)
    Xte = torch.from_numpy(data["test"][0]).to(dev)

    counts = np.array([(ytr == c).sum() for c in range(9)], dtype=float)
    n_out = 9 if args.loss == "ce" else 8
    model = WaferResNet(n_out=n_out, in_ch=2, width=args.width).to(dev)
    if args.init:
        sd = torch.load(args.init, map_location=dev)["model"]
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
        missing = model.load_state_dict(sd, strict=False)
        print("init from", args.init, missing.unexpected_keys, flush=True)
    if args.freeze:
        for n_, prm in model.named_parameters():
            prm.requires_grad = n_.startswith("head.")

    pos_w = None
    if args.posweight_power > 0:
        npos = counts[1:]
        pos_w = torch.tensor(((counts.sum() - npos) / npos) ** args.posweight_power,
                             device=dev, dtype=torch.float32)
    samp_w = None
    if args.sample_power > 0:
        w = (1.0 / np.maximum(counts, 1)) ** args.sample_power
        samp_w = torch.tensor(w[ytr], device=dev, dtype=torch.float64)

    params = [prm for prm in model.parameters() if prm.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps = args.steps_per_epoch or max(1, len(Xg) // args.batch)
    total = steps * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total,
                                                pct_start=0.1)
    log = (out / "log.jsonl").open("w")
    best = {"val_macro_f1": -1.0}

    for ep in range(args.epochs):
        model.train()
        need = steps * args.batch
        if samp_w is not None:
            perm = torch.multinomial(samp_w, need, replacement=True)
        else:                       # repeat the shuffle if one pass is too short
            reps = [torch.randperm(len(Xg), device=dev)
                    for _ in range(max(1, -(-need // len(Xg))))]
            perm = torch.cat(reps)[:need]
        tot = 0.0
        for i in range(steps):
            b = perm[i * args.batch:(i + 1) * args.batch]
            xb = Xg[b].float() / 255.0
            if args.aug:
                xb = d4(xb, int(torch.randint(8, (1,)).item()))
            with torch.autocast("cuda", torch.bfloat16):
                logit = model(xb).float()
                if args.loss == "ce":
                    loss = F.cross_entropy(logit, yg[b])
                elif args.loss == "focal":
                    bce = F.binary_cross_entropy_with_logits(
                        logit, Yg[b], reduction="none", pos_weight=pos_w)
                    pt = torch.exp(-bce.clamp(max=20))
                    loss = ((1 - pt) ** 2 * bce).mean()
                else:
                    loss = F.binary_cross_entropy_with_logits(logit, Yg[b],
                                                              pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        probs, Y = evaluate(model, Xva, data["val"][1], n_out=n_out, loss=args.loss)
        ths = tune_thresholds(probs, Y)
        ml = multilabel_f1(probs, Y, ths)
        agg = aggregate(ml)
        rec = {"epoch": ep + 1, "loss": tot / steps, "lr": sched.get_last_lr()[0],
               "val_macro_f1": agg["macro_f1"], "val_micro_f1": agg["micro_f1"],
               "elapsed_min": (time.time() - t0) / 60}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if agg["macro_f1"] > best["val_macro_f1"]:
            best = {"val_macro_f1": agg["macro_f1"], "epoch": ep + 1,
                    "ths": ths.tolist()}
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": ep + 1}, out / "best.pt")
    log.close()

    model.load_state_dict(torch.load(out / "best.pt", map_location=dev)["model"])
    ths = np.array(best["ths"])                 # frozen from val, never refit
    res = {"args": vars(args), "best_epoch": best["epoch"],
           "wall_min": (time.time() - t0) / 60,
           "n_params": sum(p_.numel() for p_ in model.parameters()),
           "train_class_counts": {c: int(counts[i]) for i, c in enumerate(CLASSES_9)}}
    for s, Xs in (("val", Xva), ("test", Xte)):
        probs, Y = evaluate(model, Xs, data[s][1], n_out=n_out, loss=args.loss)
        res[s] = report(probs, data[s][1], Y, ths)
        np.save(out / f"probs_{s}.npy", probs.astype(np.float32))
    (out / "eval.json").write_text(json.dumps(res, indent=2))
    print("TEST multilabel macro-F1", res["test"]["multilabel_8"]["macro_f1"],
          "9-class weighted-F1", res["test"]["class9"]["weighted_f1"], flush=True)


if __name__ == "__main__":
    main()
