"""Stopping hopeless trials early.

Every trial used to run all its folds, including configurations that were
plainly losing after the first one. These check that early stopping happens,
that it does not cost quality, and — the part most easily got wrong — that a
trial stopped on purpose is not filed alongside one that crashed.
"""

import optuna
import pytest

from saltml.data import load
from saltml.search import _make_pruner, search

optuna.logging.set_verbosity(optuna.logging.ERROR)

IRIS = "data/classification/iris.arff"
LEARNERS = ["svm", "random_forest", "knn", "logistic_regression"]


def test_no_pruner_runs_every_trial_to_completion():
    result = search(load(IRIS), learners=LEARNERS, n_trials=25, folds=5, seed=0, pruner=None)
    assert result.n_pruned == 0
    assert len(result.records) == 25


def test_median_pruner_stops_trials_early():
    result = search(load(IRIS), learners=LEARNERS, n_trials=40, folds=5, seed=0, pruner="median")
    assert result.n_pruned > 0
    # A pruned trial has only a partial score, so it earns no record.
    assert len(result.records) == 40 - result.n_pruned


def test_pruning_does_not_cost_accuracy_on_an_easy_dataset():
    kept = search(load(IRIS), learners=LEARNERS, n_trials=40, folds=5, seed=0, pruner=None)
    cut = search(load(IRIS), learners=LEARNERS, n_trials=40, folds=5, seed=0, pruner="median")
    assert cut.best.score >= kept.best.score - 0.02


def test_pruned_trials_are_not_counted_as_failures():
    """TrialPruned is an ordinary exception, so the failure handler will claim
    a deliberately stopped trial unless it is re-raised ahead of it."""
    result = search(load(IRIS), learners=LEARNERS, n_trials=40, folds=5, seed=0, pruner="median")
    assert result.n_pruned > 0
    assert result.n_failed == 0
    assert not result.never_worked


def test_survey_phase_is_never_pruned():
    """The survey's fixed slice per learner is what measures cost. Cutting it
    short would settle the algorithm choice on partial evidence."""
    result = search(load(IRIS), learners=LEARNERS, timeout=6.0, folds=5, seed=0, pruner="median")
    surveyed = [r for r in result.records if r.phase == "survey"]
    assert {r.learner for r in surveyed} == set(LEARNERS)


@pytest.mark.parametrize("name", ["median", "asha", "hyperband"])
def test_named_pruners_resolve(name):
    assert isinstance(_make_pruner(name), optuna.pruners.BasePruner)


def test_none_disables_pruning():
    assert isinstance(_make_pruner(None), optuna.pruners.NopPruner)
    assert isinstance(_make_pruner("none"), optuna.pruners.NopPruner)


def test_unknown_pruner_is_rejected():
    with pytest.raises(ValueError, match="Unknown pruner"):
        _make_pruner("magic")


def test_a_pruner_instance_passes_through():
    given = optuna.pruners.MedianPruner(n_startup_trials=3)
    assert _make_pruner(given) is given
