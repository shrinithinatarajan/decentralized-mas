import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.baselines import RandomForestBaseline


# ---- fit / predict ----

def test_rf_baseline_predict_returns_valid_labels():
    clf = RandomForestBaseline()
    X = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y = np.array([1, 0, 1, 0])
    clf.fit(X, y)
    preds = clf.predict(X)
    assert all(p in ("SENSITIVE", "RESISTANT") for p in preds)


def test_rf_baseline_predict_length_matches_input():
    clf = RandomForestBaseline()
    X = np.array([[1, 0], [0, 1], [1, 1]])
    y = np.array([1, 0, 1])
    clf.fit(X, y)
    assert len(clf.predict(X)) == 3


def test_rf_baseline_predict_proba_in_unit_interval():
    clf = RandomForestBaseline()
    X = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y = np.array([1, 0, 1, 0])
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (4,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_rf_baseline_perfect_separation_yields_auroc_1():
    # Two clearly separable clusters
    X_train = np.array([[10, 0]] * 10 + [[0, 10]] * 10)
    y_train = np.array([1] * 10 + [0] * 10)
    clf = RandomForestBaseline(random_state=42)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_train)
    assert roc_auc_score(y_train, probs) == pytest.approx(1.0)


def test_rf_baseline_raises_if_predict_before_fit():
    clf = RandomForestBaseline()
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(np.array([[1, 0]]))


def test_rf_baseline_raises_if_predict_proba_before_fit():
    clf = RandomForestBaseline()
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_proba(np.array([[1, 0]]))


def test_rf_baseline_sensitive_label_maps_to_class_1():
    # SENSITIVE should correspond to the positive class (1)
    clf = RandomForestBaseline(random_state=0)
    X = np.array([[5.0]] * 6 + [[-5.0]] * 6)
    y = np.array([1] * 6 + [0] * 6)
    clf.fit(X, y)
    preds = clf.predict(X)
    # first half should be SENSITIVE, second RESISTANT
    assert preds[:6] == ["SENSITIVE"] * 6
    assert preds[6:] == ["RESISTANT"] * 6
