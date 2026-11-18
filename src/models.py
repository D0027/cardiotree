"""Model training and evaluation helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from .config import RS, DT_TUNED, RF_TUNED


def make_tree(**kw) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=RS, **kw)


def make_forest(**kw) -> RandomForestClassifier:
    params = {"oob_score": True, **RF_TUNED, **kw}
    return RandomForestClassifier(random_state=RS, n_jobs=-1, **params)


def train_all(X_tr, y_tr) -> dict:
    return {
        "Decision Tree (unrestricted)": make_tree().fit(X_tr, y_tr),
        "Decision Tree (tuned)": make_tree(**DT_TUNED).fit(X_tr, y_tr),
        "Random Forest (default)": make_forest(max_depth=None).fit(X_tr, y_tr),
        "Random Forest (tuned)": make_forest().fit(X_tr, y_tr),
    }


def compare(models: dict, X_tr, y_tr, X_te, y_te) -> pd.DataFrame:
    rows = []
    for name, m in models.items():
        pred, proba = m.predict(X_te), m.predict_proba(X_te)[:, 1]
        rows.append({
            "model": name,
            "train_acc": m.score(X_tr, y_tr),
            "test_acc": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, proba),
        })
    return pd.DataFrame(rows).set_index("model")


def depth_sweep(X_tr, y_tr, X_te, y_te, max_d=15) -> pd.DataFrame:
    skf = StratifiedKFold(5, shuffle=True, random_state=RS)
    rows = []
    for d in range(1, max_d + 1):
        t = make_tree(max_depth=d)
        cv = cross_val_score(t, X_tr, y_tr, cv=skf).mean()
        t.fit(X_tr, y_tr)
        rows.append({"max_depth": d, "train": t.score(X_tr, y_tr), "cv": cv, "test": t.score(X_te, y_te)})
    return pd.DataFrame(rows)


def ccp_sweep(X_tr, y_tr, X_te, y_te) -> pd.DataFrame:
    skf = StratifiedKFold(5, shuffle=True, random_state=RS)
    path = make_tree().cost_complexity_pruning_path(X_tr, y_tr)
    alphas = np.unique(np.round(path.ccp_alphas, 5))[:-1]
    rows = []
    for a in alphas:
        t = make_tree(ccp_alpha=a)
        cv = cross_val_score(t, X_tr, y_tr, cv=skf).mean()
        t.fit(X_tr, y_tr)
        rows.append({"ccp_alpha": a, "train": t.score(X_tr, y_tr), "cv": cv,
                     "test": t.score(X_te, y_te), "leaves": t.get_n_leaves()})
    return pd.DataFrame(rows)


def trees_sweep(X_tr, y_tr, X_te, y_te, grid=(1, 5, 10, 25, 50, 100, 200, 300, 500)) -> pd.DataFrame:
    rows = []
    for n in grid:
        m = make_forest(n_estimators=n, oob_score=n >= 25).fit(X_tr, y_tr)
        rows.append({"n_estimators": n, "test": m.score(X_te, y_te),
                     "oob": m.oob_score_ if n >= 25 else np.nan})
    return pd.DataFrame(rows)


def repeated_cv(models: dict, X, y, n_repeats=3) -> tuple[pd.DataFrame, dict]:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RS)
    rows, scores = [], {}
    for name, m in models.items():
        from sklearn.base import clone
        res = cross_validate(clone(m), X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
        scores[name] = res["test_accuracy"]
        rows.append({"model": name,
                     "accuracy": res["test_accuracy"].mean(), "acc_std": res["test_accuracy"].std(),
                     "f1": res["test_f1"].mean(), "roc_auc": res["test_roc_auc"].mean()})
    return pd.DataFrame(rows), scores


def permutation_imp(model, X_te, y_te, features, n_repeats=30) -> pd.DataFrame:
    p = permutation_importance(model, X_te, y_te, n_repeats=n_repeats, random_state=RS,
                               n_jobs=-1, scoring="accuracy")
    return pd.DataFrame({"mean": p.importances_mean, "std": p.importances_std}, index=features)


# ---- information theory helpers (worked example from the notebook) ----
def entropy(labels) -> float:
    p = np.bincount(labels, minlength=2) / len(labels)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def gini(labels) -> float:
    p = np.bincount(labels, minlength=2) / len(labels)
    return float(1 - (p ** 2).sum())


def info_gain(labels, left_mask) -> float:
    n = len(labels)
    left, right = labels[left_mask], labels[~left_mask]
    return entropy(labels) - len(left) / n * entropy(left) - len(right) / n * entropy(right)


def best_gain_per_feature(X, y_arr) -> pd.Series:
    out = {}
    for col in X.columns:
        vals = np.sort(X[col].unique())
        th = (vals[:-1] + vals[1:]) / 2
        out[col] = max((info_gain(y_arr, (X[col] <= t).values) for t in th), default=0.0)
    return pd.Series(out).sort_values()


def decision_path_text(tree, x_row: pd.DataFrame, features) -> list[str]:
    """Human-readable path a single sample takes through a fitted tree."""
    t = tree.tree_
    node_ids = tree.decision_path(x_row).indices
    steps = []
    for nid in node_ids:
        if t.children_left[nid] == t.children_right[nid]:
            cls = int(np.argmax(t.value[nid]))
            steps.append(f"Leaf → predicts **target {cls}** ({int(t.n_node_samples[nid])} training samples)")
        else:
            f, thr = features[t.feature[nid]], t.threshold[nid]
            v = float(x_row.iloc[0][f])
            steps.append(f"`{f}` = {v:g}  {'≤' if v <= thr else '>'}  {thr:.2f}")
    return steps
