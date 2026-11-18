import numpy as np
from src import data, models


def _setup():
    df, n_dup = data.load_clean()
    return df, n_dup, data.split(df)


def test_cleaning_matches_notebook():
    df, n_dup, _ = _setup()
    assert len(df) == 302 and n_dup == 1 and df.isnull().sum().sum() == 0


def test_split_is_stratified():
    _, _, (X, y, X_tr, X_te, y_tr, y_te, f) = _setup()
    assert len(X_te) == 61 and abs(y_tr.mean() - y_te.mean()) < 0.02 and len(f) == 13


def test_unrestricted_tree_overfits_and_forest_beats_it():
    _, _, (X, y, X_tr, X_te, y_tr, y_te, f) = _setup()
    ms = models.train_all(X_tr, y_tr)
    assert ms["Decision Tree (unrestricted)"].score(X_tr, y_tr) == 1.0
    assert ms["Random Forest (tuned)"].score(X_te, y_te) > ms["Decision Tree (unrestricted)"].score(X_te, y_te)


def test_information_gain_matches_sklearn():
    _, _, (X, y, X_tr, X_te, y_tr, y_te, f) = _setup()
    s = models.make_tree(criterion="entropy", max_depth=1).fit(X_tr, y_tr)
    t = s.tree_
    mask = (X_tr.iloc[:, t.feature[0]] <= t.threshold[0]).values
    n = t.n_node_samples
    sk = t.impurity[0] - n[1] / n[0] * t.impurity[1] - n[2] / n[0] * t.impurity[2]
    assert np.isclose(models.info_gain(y_tr.values, mask), sk)


def test_decision_path_ends_in_leaf():
    _, _, (X, y, X_tr, X_te, y_tr, y_te, f) = _setup()
    t = models.make_tree(max_depth=3).fit(X_tr, y_tr)
    steps = models.decision_path_text(t, X_te.iloc[[0]], f)
    assert steps[-1].startswith("Leaf")
