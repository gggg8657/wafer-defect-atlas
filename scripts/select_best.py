"""Pick the model the README reports, on VAL, and mirror it to runs/cls_best.

    python scripts/select_best.py

Selection reads `val.multilabel_8.macro_f1` only.  Test metrics are already in
each eval.json, but they are never consulted here -- that is the whole point of
having a separate val split of lots.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

R = Path("runs")


def main():
    cands = sorted(list(R.glob("cls_*/eval.json")) + list(R.glob("final/seed*/eval.json")))
    rows = []
    for p in cands:
        e = json.loads(p.read_text())
        rows.append((e["val"]["multilabel_8"]["macro_f1"], str(p.parent), e))
    rows.sort(reverse=True)
    for v, name, e in rows:
        print(f"{name:<24} val {v:.4f}   test {e['test']['multilabel_8']['macro_f1']:.4f}")
    best = rows[0]
    dst = R / "cls_best"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(best[1], dst, ignore=shutil.ignore_patterns("probs_*.npy"))
    seeds = [(v, n, e) for v, n, e in rows if n.startswith("runs/final/seed")]
    sel = {"selected": best[1], "selected_val_macro_f1": best[0],
           "candidates": {n: {"val_macro_f1": v,
                              "test_macro_f1": e["test"]["multilabel_8"]["macro_f1"]}
                          for v, n, e in rows}}
    if len(seeds) > 1:
        t = [e["test"]["multilabel_8"]["macro_f1"] for _, _, e in seeds]
        w = [e["test"]["class9"]["weighted_f1"] for _, _, e in seeds]
        sel["seed_spread"] = {
            "n_seeds": len(seeds), "test_macro_f1": t,
            "test_macro_f1_mean": sum(t) / len(t),
            "test_macro_f1_range": max(t) - min(t),
            "test_weighted_f1_9_mean": sum(w) / len(w),
            "test_weighted_f1_9_range": max(w) - min(w)}
    (R / "selection.json").write_text(json.dumps(sel, indent=2))
    print("-> runs/cls_best =", best[1])


if __name__ == "__main__":
    main()
