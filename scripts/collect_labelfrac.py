"""runs/labelfrac/*/eval.json -> runs/labelfrac.json (what fig_ssl reads)."""
import json, sys
from pathlib import Path

out = {}
for d in sorted(Path("runs/labelfrac").glob("*/eval.json")):
    e = json.loads(d.read_text())
    mode, tag = d.parent.name.split("_", 1)
    frac = e["args"]["label_frac"]
    out.setdefault(mode, {})[f"{frac:g}"] = {
        "test_macro_f1": e["test"]["multilabel_8"]["macro_f1"],
        "test_micro_f1": e["test"]["multilabel_8"]["micro_f1"],
        "test_weighted_f1_9": e["test"]["class9"]["weighted_f1"],
        "val_macro_f1": e["val"]["multilabel_8"]["macro_f1"],
        "n_train_labels": int(sum(e["train_class_counts"].values())),
        "best_epoch": e["best_epoch"],
    }
Path("runs/labelfrac.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
