import numpy as np
import optuna
import pandas as pd
import pytest

import saltml
from saltml.data import load
from saltml.metrics import resolve_metric
from saltml.search import search
from saltml.task import Task

optuna.logging.set_verbosity(optuna.logging.WARNING)

IRIS = "data/classification/iris.arff"


def test_search_finds_a_workable_model_on_iris():
    result = search(load(IRIS), n_trials=12, folds=3, seed=0)
    assert result.task is Task.CLASSIFICATION
    assert result.metric == "balanced_accuracy"
    assert result.records
    assert result.best.score > 0.8  # iris is easy; anything less means a real fault
    assert result.n_failed == 0


def test_leaderboard_is_ranked_and_capped():
    result = search(load(IRIS), n_trials=10, folds=3, seed=0)
    board = result.leaderboard(top=3)
    assert len(board) <= 3
    assert list(board["rank"]) == sorted(board["rank"])
    assert board["balanced_accuracy"].is_monotonic_decreasing


def test_best_per_learner_has_one_row_each():
    result = search(load(IRIS), n_trials=15, folds=3, seed=0)
    board = result.best_per_learner()
    assert board["learner"].is_unique
    assert board["trials"].sum() == len(result.records)


def test_restricting_learners_is_respected():
    result = search(load(IRIS), learners=["decision_tree"], n_trials=5, folds=3, seed=0)
    assert {r.learner for r in result.records} == {"decision_tree"}


def test_folds_shrink_to_smallest_class():
    # Three samples in one class cannot support 5 stratified folds.
    frame = pd.DataFrame(
        {"a": list(range(23)), "b": list(range(23)),
         "y": ["rare"] * 3 + ["common"] * 20}
    )
    result = search(load(frame), learners=["decision_tree"], n_trials=2, folds=5, seed=0)
    assert result.records


def test_fit_reports_an_untouched_holdout_score():
    result = saltml.fit(IRIS, n_trials=10, folds=3, holdout=0.3, seed=0)
    assert result.holdout_score is not None
    assert 0.0 <= result.holdout_score <= 1.0
    assert result.learner in [r.learner for r in result.search.records]


def test_fit_without_holdout_reports_none():
    result = saltml.fit(IRIS, n_trials=6, folds=3, holdout=0.0, seed=0)
    assert result.holdout_score is None


def test_fitted_model_predicts():
    result = saltml.fit(IRIS, n_trials=6, folds=3, seed=0)
    predictions = result.predict(result.dataset.X.head(5))
    assert len(predictions) == 5


def test_model_round_trips_through_disk(tmp_path):
    import joblib

    result = saltml.fit(IRIS, n_trials=6, folds=3, seed=0)
    path = result.save(tmp_path / "model.joblib")
    reloaded = joblib.load(path)
    assert list(reloaded.predict(result.dataset.X.head(3))) == list(
        result.predict(result.dataset.X.head(3))
    )


def test_search_is_reproducible_for_a_seed():
    a = search(load(IRIS), n_trials=8, folds=3, seed=7)
    b = search(load(IRIS), n_trials=8, folds=3, seed=7)
    assert [r.score for r in a.records] == [r.score for r in b.records]


def test_regression_path_end_to_end():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    frame["site"] = rng.choice(["x", "y"], 120)  # categorical feature
    frame["target"] = frame["a"] * 3 - frame["b"] + rng.normal(scale=0.1, size=120)
    result = saltml.fit(frame, n_trials=10, folds=3, seed=0)
    assert result.search.task is Task.REGRESSION
    assert result.metric == "r2"
    assert result.cv_score > 0.5


def test_missing_values_are_imputed_not_fatal():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(80, 3)), columns=list("abc"))
    frame.loc[rng.choice(80, 20, replace=False), "a"] = np.nan
    frame["y"] = rng.choice(["p", "q"], 80)
    result = search(load(frame), learners=["decision_tree"], n_trials=3, folds=3, seed=0)
    assert result.n_failed == 0


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="Unknown metric"):
        resolve_metric(Task.CLASSIFICATION, "not_a_metric")


def test_unknown_sampler_is_rejected():
    with pytest.raises(ValueError, match="Unknown sampler"):
        search(load(IRIS), n_trials=1, sampler="magic")


def test_hypercube_sampler_reports_that_it_is_not_ported_yet():
    with pytest.raises(ValueError, match="not been ported"):
        search(load(IRIS), n_trials=1, sampler="hypercube")
