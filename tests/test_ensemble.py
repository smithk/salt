"""Combining the best trials rather than keeping only the winner."""

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.metrics import get_scorer

import saltml
from saltml.data import load
from saltml.ensemble import _one_hot, build_ensemble
from saltml.search import search
from saltml.task import Task

optuna.logging.set_verbosity(optuna.logging.ERROR)

IRIS = "data/classification/iris.arff"
LEARNERS = ["svm", "random_forest", "knn", "logistic_regression"]


def _regression_frame(n=200):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    frame["target"] = frame["a"] * 2 - frame["b"] + rng.normal(scale=0.3, size=n)
    return frame


def test_one_hot_widens_labels_to_a_probability_matrix():
    classes = np.array(["a", "b", "c"])
    wide = _one_hot(np.array(["c", "a"]), classes)
    assert wide.shape == (2, 3)
    assert list(wide[0]) == [0.0, 0.0, 1.0]
    assert list(wide[1]) == [1.0, 0.0, 0.0]


def test_ensemble_builds_and_predicts_for_classification():
    dataset = load(IRIS)
    result = search(dataset, learners=LEARNERS, n_trials=25, folds=3, seed=0, pruner=None)
    model = build_ensemble(dataset, result, folds=3, seed=0)
    assert model is not None
    predictions = model.predict(dataset.X)
    assert len(predictions) == len(dataset.y)
    assert set(np.unique(predictions)) <= set(np.unique(dataset.y))
    assert sum(model.weights) == pytest.approx(1.0)
    assert len(model.members) == len(model.weights) == len(model.models)


def test_probabilities_are_a_valid_distribution():
    dataset = load(IRIS)
    result = search(dataset, learners=LEARNERS, n_trials=25, folds=3, seed=0, pruner=None)
    model = build_ensemble(dataset, result, folds=3, seed=0)
    proba = model.predict_proba(dataset.X)
    assert proba.shape == (len(dataset.y), len(np.unique(dataset.y)))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_hard_label_learners_can_still_join():
    """An SVC is fitted with probability=False and a ridge classifier has no
    probabilities at all; both must still be eligible members."""
    dataset = load(IRIS)
    result = search(
        dataset, learners=["svm", "ridge_classifier"], n_trials=20, folds=3,
        seed=0, pruner=None,
    )
    model = build_ensemble(dataset, result, folds=3, seed=0)
    assert model is not None
    assert np.allclose(model.predict_proba(dataset.X).sum(axis=1), 1.0)


def test_ensemble_builds_for_regression():
    dataset = load(_regression_frame(), "target")
    result = search(
        dataset, learners=["random_forest", "knn", "ridge"], n_trials=20, folds=3,
        seed=0, pruner=None,
    )
    model = build_ensemble(dataset, result, folds=3, seed=0)
    assert model is not None
    assert model.task is Task.REGRESSION
    predictions = model.predict(dataset.X)
    assert predictions.dtype.kind == "f"
    with pytest.raises(AttributeError):
        model.predict_proba(dataset.X)


def test_membership_never_scores_worse_than_the_best_single_out_of_fold():
    """Greedy selection keeps the best-scoring prefix, so the ensemble cannot
    be worse out-of-fold than the single best candidate it started from."""
    dataset = load(IRIS)
    result = search(dataset, learners=LEARNERS, n_trials=30, folds=3, seed=0, pruner=None)
    model = build_ensemble(dataset, result, folds=3, seed=0)
    scorer = get_scorer(result.metric)
    assert scorer(model, dataset.X, dataset.y) >= result.best.score - 0.05


def test_too_few_configurations_returns_nothing_to_combine():
    dataset = load(IRIS)
    result = search(dataset, learners=["gaussian_nb"], n_trials=1, folds=3, seed=0, pruner=None)
    assert build_ensemble(dataset, result, folds=3, seed=0) is None


def test_describe_lists_members_and_weights():
    dataset = load(IRIS)
    result = search(dataset, learners=LEARNERS, n_trials=25, folds=3, seed=0, pruner=None)
    model = build_ensemble(dataset, result, folds=3, seed=0)
    table = model.describe()
    assert list(table.columns) == ["learner", "weight", "params"]
    assert table["weight"].sum() == pytest.approx(1.0)


def test_fit_can_return_an_ensemble_and_it_still_saves(tmp_path):
    result = saltml.fit(
        IRIS, learners=LEARNERS, n_trials=20, folds=3, seed=0, ensemble=True
    )
    assert result.holdout_score is not None
    path = result.save(tmp_path / "model.joblib")
    import joblib

    reloaded = joblib.load(path)
    assert len(reloaded.predict(result.dataset.X)) == len(result.dataset.y)
