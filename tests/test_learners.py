"""Every learner must actually fit.

The search space and the estimator's own constraints are declared in different
places, so an illegal combination (a solver that cannot do multiclass, a
penalty a solver rejects) only shows up at fit time. These tests draw many
configurations from each space and fit them, on both binary and multiclass
data, so such a mismatch fails here rather than as a skipped trial.
"""

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from saltml.data import load
from saltml.learners import REGISTRY, for_task, resolve
from saltml.search import build_pipeline
from saltml.task import Task

optuna.logging.set_verbosity(optuna.logging.WARNING)

DRAWS = 12


def _dataset(task: Task, n_classes: int = 3):
    if task is Task.CLASSIFICATION:
        X, y = make_classification(
            n_samples=120, n_features=6, n_informative=4,
            n_classes=n_classes, n_clusters_per_class=1, random_state=0,
        )
    else:
        X, y = make_regression(n_samples=120, n_features=6, noise=0.2, random_state=0)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    frame["target"] = y
    return load(frame, task=task)


def _draw_and_fit(learner, dataset, seed):
    """Sample one configuration and fit it, returning the built params."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    captured = {}

    def objective(trial):
        params = learner.sample(trial, seed=0)
        captured.update(params)
        build_pipeline(dataset, learner, params).fit(dataset.X, dataset.y)
        return 0.0

    study.optimize(objective, n_trials=1, catch=())
    return captured


@pytest.mark.parametrize("name", sorted(REGISTRY[Task.CLASSIFICATION]))
@pytest.mark.parametrize("n_classes", [2, 3])
def test_classifier_configurations_fit(name, n_classes):
    learner = REGISTRY[Task.CLASSIFICATION][name]
    dataset = _dataset(Task.CLASSIFICATION, n_classes)
    for seed in range(DRAWS):
        _draw_and_fit(learner, dataset, seed)


@pytest.mark.parametrize("name", sorted(REGISTRY[Task.REGRESSION]))
def test_regressor_configurations_fit(name):
    learner = REGISTRY[Task.REGRESSION][name]
    dataset = _dataset(Task.REGRESSION)
    for seed in range(DRAWS):
        _draw_and_fit(learner, dataset, seed)


def test_registry_names_are_unique_per_task():
    for task in Task:
        names = [learner.name for learner in for_task(task)]
        assert len(names) == len(set(names))


def test_resolve_defaults_to_everything():
    assert len(resolve(Task.CLASSIFICATION, None)) == len(REGISTRY[Task.CLASSIFICATION])


def test_resolve_reports_unknown_names():
    with pytest.raises(KeyError, match="banana"):
        resolve(Task.CLASSIFICATION, ["banana"])


def test_parameter_names_are_namespaced_by_learner():
    """Two learners sharing a parameter name must not collide in one study."""
    dataset = _dataset(Task.CLASSIFICATION)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))

    def objective(trial):
        for learner in for_task(Task.CLASSIFICATION):
            learner.sample(trial, seed=0)
        return 0.0

    study.optimize(objective, n_trials=1, catch=())
    names = list(study.trials[0].params)
    assert all(name.count("__") >= 1 for name in names)
    # 'C' appears in both logistic_regression and svm; both must survive.
    assert "logistic_regression__C" in names
    assert "svm__C" in names
