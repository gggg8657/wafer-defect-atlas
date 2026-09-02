"""Train WaferCNN on synthetic wafer maps (CPU-friendly)."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .data import make_dataset, CLASSES
from .model import WaferCNN


def train(epochs=15, N=32, seed=0, verbose=True):
    torch.manual_seed(seed)
    X, y = make_dataset(n_per_class=250, N=N, seed=seed)
    Xt, yt = make_dataset(n_per_class=80, N=N, seed=seed + 1)
    X, y = torch.tensor(X), torch.tensor(y)
    Xt, yt = torch.tensor(Xt), torch.tensor(yt)

    model = WaferCNN(n_classes=len(CLASSES))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            b = perm[i : i + 64]
            opt.zero_grad()
            loss = F.cross_entropy(model(X[b]), y[b])
            loss.backward()
            opt.step()
        if verbose and (ep + 1) % 5 == 0:
            acc = evaluate(model, Xt, yt)
            print(f"epoch {ep+1:3d}  test acc {acc:.3f}")
    return model, evaluate(model, Xt, yt), (Xt, yt)


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(1)
    return (pred == y).float().mean().item()
