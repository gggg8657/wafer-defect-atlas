"""SimCLR pretraining on the 638,507 UNLABELED WM-811K wafer maps.

    CUDA_VISIBLE_DEVICES=1 python scripts/pretrain_simclr.py --out runs/ssl

79% of WM-811K carries no failure label at all.  That unlabeled majority is the
only part of the dataset big enough to learn a representation from without
labels, and it is also the pool the clustering KPI runs on -- so the encoder
that clusters it should be one that never saw a label.  Only lots assigned to
the *train* split are used, so the labeled val/test wafers stay untouched.

Augmentations are chosen for what a wafer map actually is:
  * d4 -- a signature is defined by shape, not orientation
  * small translate/scale -- the wafer is centred by construction, so this is
    kept mild; cropping away the edge would destroy the edge-ring signature,
    which is the opposite of what an ImageNet recipe wants
  * Bernoulli die dropout + salt noise -- the physical nuisance: a map is a
    Bernoulli sample of an underlying fail-probability field, so two views of
    the same wafer are two draws from it
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wafermap.model import WaferResNet


def augment(x, gen, kinds=("d4", "affine", "noise")):
    b = x.shape[0]
    if "d4" in kinds:
        g = torch.randint(8, (1,), generator=gen, device=x.device).item()
        if g & 4:
            x = torch.flip(x, [-1])
        if g & 3:
            x = torch.rot90(x, g & 3, (-2, -1))
    if "affine" not in kinds and "noise" not in kinds:
        return x
    # mild translate/scale via an affine grid (keeps the wafer disc in frame)
    s = 1.0 + 0.15 * (torch.rand(b, 1, 1, device=x.device, generator=gen) - 0.5)
    t = 0.10 * (torch.rand(b, 2, 1, device=x.device, generator=gen) - 0.5)
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = s[:, 0, 0]
    theta[:, 1, 1] = s[:, 0, 0]
    theta[:, :, 2] = t[:, :, 0]
    if "affine" in kinds:
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    if "noise" not in kinds:
        return x
    # die-level Bernoulli resampling of the fail channel
    keep = (torch.rand(x.shape, device=x.device, generator=gen) > 0.15).float()
    salt = (torch.rand(x.shape, device=x.device, generator=gen) < 0.01).float()
    f = (x[:, 1:2] * keep[:, 1:2] + salt[:, 1:2] * x[:, 0:1]).clamp(0, 1)
    return torch.cat([x[:, 0:1], f], 1)


class Projector(nn.Module):
    def __init__(self, enc, dim=128, out=64):
        super().__init__()
        self.enc = enc
        self.mlp = nn.Sequential(nn.Linear(enc.n_feat, dim), nn.ReLU(),
                                 nn.Linear(dim, out))

    def forward(self, x):
        return F.normalize(self.mlp(self.enc.embed(x)), dim=1)


def nt_xent(z1, z2, tau=0.2):
    z = torch.cat([z1, z2])
    n = z1.shape[0]
    sim = (z @ z.T) / tau
    sim.fill_diagonal_(-1e4)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/ssl")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aug", default="d4,affine,noise",
                   help="comma list from d4,affine,noise -- the ablation that "
                        "asks which view-invariance destroys the defect signal")
    args = p.parse_args()
    kinds = tuple(args.aug.split(","))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    t0 = time.time()

    X = np.load("data/proc/unlabeled.npy", mmap_mode="r")
    m = np.load("data/proc/meta_unlabeled.npz", allow_pickle=False)
    idx = np.flatnonzero(m["split"] == "train")
    Xg = torch.from_numpy(np.ascontiguousarray(X[idx])).to(dev)
    print(f"unlabeled train maps: {len(idx):,}  ({Xg.numel()/1e9:.1f} GB on GPU)",
          flush=True)

    enc = WaferResNet(n_out=8, in_ch=2, width=args.width).to(dev)
    model = Projector(enc).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = len(Xg) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr,
                                                total_steps=steps * args.epochs,
                                                pct_start=0.05)
    log = (out / "log.jsonl").open("w")
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xg), device=dev, generator=gen)
        tot = 0.0
        for i in range(steps):
            xb = Xg[perm[i * args.batch:(i + 1) * args.batch]].float() / 255.0
            with torch.autocast("cuda", torch.bfloat16):
                loss = nt_xent(model(augment(xb, gen, kinds)).float(),
                               model(augment(xb, gen, kinds)).float(), args.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        rec = {"epoch": ep + 1, "nt_xent": tot / steps,
               "elapsed_min": (time.time() - t0) / 60}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        torch.save({"model": enc.state_dict(), "args": vars(args), "epoch": ep + 1},
                   out / "ssl.pt")
    log.close()
    (out / "ssl.json").write_text(json.dumps(
        {"args": vars(args), "n_unlabeled_train": int(len(idx)),
         "final_nt_xent": rec["nt_xent"], "wall_min": (time.time() - t0) / 60},
        indent=2))


if __name__ == "__main__":
    main()
