"""Train everything from the CLI and dump metrics to outputs/metrics.json.

    python scripts/train.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data, models
from src.config import ROOT

df, n_dup = data.load_clean()
X, y, X_tr, X_te, y_tr, y_te, feats = data.split(df)
ms = models.train_all(X_tr, y_tr)
comp = models.compare(ms, X_tr, y_tr, X_te, y_te).round(4)
cv_df, _ = models.repeated_cv({k: v for k, v in ms.items()}, X, y)
out = {
    "samples": len(df), "duplicates_removed": n_dup,
    "test_comparison": comp.reset_index().to_dict("records"),
    "cross_validation": cv_df.round(4).to_dict("records"),
}
(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "outputs" / "metrics.json").write_text(json.dumps(out, indent=2))
print(comp.to_string())
print(cv_df.round(4).to_string())
