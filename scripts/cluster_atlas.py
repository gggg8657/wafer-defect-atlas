"""The unknown-pattern KPI: cluster the unlabeled wafers, score purity on labels
those clusters never saw.

    CUDA_VISIBLE_DEVICES=1 python scripts/cluster_atlas.py --out runs/cluster.json

Protocol, in the order it has to happen to mean anything:
  1. embed every unlabeled wafer map from *train* lots (no labels exist for
     these at all) and fit k-means on those embeddings only;
  2. freeze the centroids, embed the labeled wafers from *test* lots -- a
     disjoint set of lots, held out of both the classifier and the clustering --
     and assign each to its nearest centroid;
  3. purity = (1/N) sum_k max_c n_kc over that held-out labeled set.

Three encoders are scored under the identical protocol, because purity without
a baseline is unreadable: the supervised classifier's penultimate features, the
SimCLR encoder that never saw a label, and PCA on raw pixels as the floor.
`none` is 85% of the labeled set, so defect-only purity (the same measure
restricted to the 8 failure signatures) is reported alongside.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wafermap.model import WaferResNet
from wafermap.wm811k import CLASSES_9


def embed_all(model, X, idx, dev, bs=4096):
    out = []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for i in range(0, len(idx), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[idx[i:i + bs]])).to(dev)
            out.append(model.embed(xb.float() / 255.0).float().cpu())
    z = torch.cat(out).numpy()
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)


def purity(labels, clusters, k, mask=None):
    lab, cl = (labels, clusters) if mask is None else (labels[mask], clusters[mask])
    if len(lab) == 0:
        return float("nan"), []
    tot, rows = 0, []
    for c in range(k):
        s = cl == c
        if not s.any():
            continue
        cnt = np.bincount(lab[s], minlength=9)
        tot += cnt.max()
        rows.append({"cluster": int(c), "n": int(s.sum()),
                     "dominant": CLASSES_9[int(cnt.argmax())],
                     "dominant_frac": float(cnt.max() / s.sum()),
                     "counts": cnt.tolist()})
    return float(tot / len(lab)), rows


def load_encoder(kind, path, dev):
    if kind == "supervised":
        ck = torch.load(Path(path) / "best.pt", map_location=dev)
        w, n_out = ck["args"]["width"], (9 if ck["args"]["loss"] == "ce" else 8)
        m = WaferResNet(n_out=n_out, in_ch=2, width=w).to(dev).eval()
        m.load_state_dict(ck["model"])
        return m
    ck = torch.load(path, map_location=dev)
    m = WaferResNet(n_out=8, in_ch=2, width=ck["args"]["width"]).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cls-run", default="runs/cls_best")
    p.add_argument("--ssl", default="runs/ssl/ssl.pt")
    p.add_argument("--ks", type=int, nargs="*", default=[9, 16, 32, 64, 128])
    p.add_argument("--fit-n", type=int, default=500000,
                   help="unlabeled maps used to fit k-means")
    p.add_argument("--out", default="runs/cluster.json")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    dev = "cuda"
    rng = np.random.default_rng(a.seed)
    Xu = np.load("data/proc/unlabeled.npy", mmap_mode="r")
    mu = np.load("data/proc/meta_unlabeled.npz", allow_pickle=False)
    fit_idx = np.flatnonzero(mu["split"] == "train")
    n_pool = len(fit_idx)
    if len(fit_idx) > a.fit_n:
        fit_idx = np.sort(rng.choice(fit_idx, a.fit_n, replace=False))

    Xl = np.load("data/proc/labeled.npy", mmap_mode="r")
    ml = np.load("data/proc/meta_labeled.npz", allow_pickle=False)
    te = np.flatnonzero(ml["split"] == "test")
    y = ml["y"][te].astype(int)

    encoders = {"supervised": ("supervised", a.cls_run)}
    if Path(a.ssl).exists():
        encoders["simclr"] = ("ssl", a.ssl)

    res = {"n_unlabeled_train_pool": int(n_pool), "n_fit": int(len(fit_idx)),
           "n_eval_labeled_test": int(len(te)),
           "eval_class_counts": {c: int((y == i).sum()) for i, c in enumerate(CLASSES_9)},
           "encoders": {}}

    feats = {}
    for name, (kind, path) in encoders.items():
        t = time.time()
        enc = load_encoder(kind, path, dev)
        feats[name] = (embed_all(enc, Xu, fit_idx, dev), embed_all(enc, Xl, te, dev))
        del enc
        torch.cuda.empty_cache()
        print(f"embedded {name} in {time.time()-t:.0f}s", flush=True)

    # raw-pixel floor: PCA on the fail channel, same downstream pipeline
    t = time.time()

    def raw_chunks(X, idx, bs=20000):
        for i in range(0, len(idx), bs):
            b = np.ascontiguousarray(X[idx[i:i + bs]])[:, 1]      # fail channel
            yield b.reshape(len(b), -1).astype(np.float32) / 255.0

    sub = np.sort(rng.choice(fit_idx, min(40000, len(fit_idx)), replace=False))
    pca = PCA(n_components=64, random_state=a.seed).fit(
        np.concatenate(list(raw_chunks(Xu, sub))))
    zf = np.concatenate([pca.transform(c) for c in raw_chunks(Xu, fit_idx)])
    zt = np.concatenate([pca.transform(c) for c in raw_chunks(Xl, te)])
    print(f"pixel PCA in {time.time()-t:.0f}s", flush=True)

    defect = y > 0
    for name, (Zfit, Zte) in feats.items():
        res["encoders"][name] = {"dim": int(Zfit.shape[1]), "k": {}}
        for k in a.ks:
            km = MiniBatchKMeans(n_clusters=k, random_state=a.seed, n_init=5,
                                 batch_size=4096, max_iter=300).fit(Zfit)
            cl = km.predict(Zte)
            pur, rows = purity(y, cl, k)
            pur_d, rows_d = purity(y, cl, k, mask=defect)
            # chance: same cluster-size distribution, labels shuffled
            perm = rng.permutation(len(y))
            pur_rand, _ = purity(y[perm], cl, k)
            pur_rand_d, _ = purity(y[perm], cl, k, mask=defect)
            res["encoders"][name]["k"][str(k)] = {
                "purity": pur, "purity_defect_only": pur_d,
                "purity_chance": pur_rand, "purity_defect_only_chance": pur_rand_d,
                "nmi": float(normalized_mutual_info_score(y, cl)),
                "ari": float(adjusted_rand_score(y, cl)),
                "nmi_defect_only": float(normalized_mutual_info_score(y[defect], cl[defect])),
                "n_clusters_used": int(len(np.unique(cl))),
                "clusters": rows if k <= 32 else rows[:32],
                "clusters_defect_only": rows_d if k <= 32 else rows_d[:32],
            }
            print(name, k, "purity", round(pur, 4), "defect-only",
                  round(pur_d, 4), "nmi", round(res["encoders"][name]["k"][str(k)]["nmi"], 4),
                  flush=True)
        np.save(Path(a.out).with_suffix(f".{name}.test_emb.npy"), Zte.astype(np.float32))
    res["majority_baseline"] = float((y == 0).mean())
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
