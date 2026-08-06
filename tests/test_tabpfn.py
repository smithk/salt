"""TabPFN as a learner, and the size limits that decide when it can be used.

TabPFN raises rather than degrading past its pre-training limits, so it must
be excluded before the search rather than failing every trial. These tests
mostly exercise that boundary, which needs no model weights; only the two
marked slow actually run the network.
"""

import logging

import numpy as np
import pandas as pd
import pytest

import saltml
from saltml.data import load
from saltml.learners import REGISTRY, applicable, resolve
from saltml.learners.tabpfn import (
    MAX_CLASSES,
    MAX_FEATURES,
    MAX_SAMPLES,
    TABPFN_LEARNERS,
    tabpfn_available,
)
from saltml.task import Task

needs_tabpfn = pytest.mark.skipif(not tabpfn_available(), reason="tabpfn not installed")

CLASSIFIER = next(l for l in TABPFN_LEARNERS if l.task is Task.CLASSIFICATION)


def _frame(n_samples=60, n_features=3, n_classes=2, n_categories=0):
    rng = np.random.default_rng(0)
    data = {f"f{i}": rng.normal(size=n_samples) for i in range(n_features)}
    for i in range(n_categories):
        data[f"c{i}"] = rng.choice([f"v{j}" for j in range(5)], n_samples)
    data["y"] = np.tile(np.arange(n_classes), n_samples // n_classes + 1)[:n_samples]
    return pd.DataFrame(data)


@needs_tabpfn
def test_tabpfn_is_registered_for_both_tasks():
    assert "tabpfn" in REGISTRY[Task.CLASSIFICATION]
    assert "tabpfn" in REGISTRY[Task.REGRESSION]


def test_small_dataset_is_within_limits():
    assert CLASSIFIER.excluded_for(load(_frame(), warn_suspicious=False)) is None


def test_too_many_samples_is_excluded():
    dataset = load(_frame(n_samples=60), warn_suspicious=False)
    object.__setattr__  # noqa: B018 - Dataset is a plain dataclass; mutate directly
    dataset.X = pd.concat([dataset.X] * 200, ignore_index=True)
    dataset.y = pd.concat([dataset.y] * 200, ignore_index=True)
    assert dataset.n_samples > MAX_SAMPLES
    reason = CLASSIFIER.excluded_for(dataset)
    assert reason is not None and "samples" in reason


def test_too_many_classes_is_excluded():
    dataset = load(_frame(n_samples=200, n_classes=MAX_CLASSES + 5), warn_suspicious=False)
    reason = CLASSIFIER.excluded_for(dataset)
    assert reason is not None and "classes" in reason


def test_too_many_features_is_excluded():
    dataset = load(_frame(n_features=MAX_FEATURES + 10), warn_suspicious=False)
    reason = CLASSIFIER.excluded_for(dataset)
    assert reason is not None and "features" in reason


def test_feature_limit_counts_encoded_width_not_raw_columns():
    """One-hot encoding is what TabPFN sees, so a few wide categoricals count."""
    # 120 categorical columns of 5 levels each -> 600 encoded columns.
    dataset = load(_frame(n_features=1, n_categories=120), warn_suspicious=False)
    assert dataset.n_features < MAX_FEATURES          # raw columns look fine
    reason = CLASSIFIER.excluded_for(dataset)
    assert reason is not None and "encoded features" in reason


def test_excluded_learner_is_reported_loudly_when_asked_for(caplog):
    dataset = load(_frame(n_samples=200, n_classes=MAX_CLASSES + 5), warn_suspicious=False)
    with caplog.at_level(logging.INFO, logger="saltml"):
        usable, excluded = applicable([CLASSIFIER], dataset, requested=True)
    assert usable == []
    assert "tabpfn" in excluded
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_classical_learners_have_no_limits():
    dataset = load(_frame(n_samples=200, n_classes=MAX_CLASSES + 5), warn_suspicious=False)
    for learner in resolve(Task.CLASSIFICATION, ["random_forest", "decision_tree"]):
        assert learner.excluded_for(dataset) is None


@needs_tabpfn
def test_search_excludes_tabpfn_and_carries_on(caplog):
    """A dataset past the limits must still search, just without TabPFN."""
    dataset = load(_frame(n_samples=300, n_classes=MAX_CLASSES + 5), warn_suspicious=False)
    result = saltml.search(dataset, learners=["tabpfn", "decision_tree"],
                         n_trials=2, folds=3, seed=0)
    assert {r.learner for r in result.records} == {"decision_tree"}


def test_search_errors_when_nothing_is_applicable():
    dataset = load(_frame(n_samples=200, n_classes=MAX_CLASSES + 5), warn_suspicious=False)
    with pytest.raises(ValueError, match="No learner can be used"):
        saltml.search(dataset, learners=["tabpfn"], n_trials=1, folds=3, seed=0)


@needs_tabpfn
@pytest.mark.slow
def test_tabpfn_fits_and_scores_on_iris():
    result = saltml.search(
        load("data/standard_ml_sets/classification/datasets/standard/iris.arff"),
        learners=["tabpfn"], n_trials=2, folds=3, seed=0,
    )
    assert result.n_failed == 0
    assert result.best.score > 0.85


@needs_tabpfn
@pytest.mark.slow
def test_tabpfn_regression_runs():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(60, 3)), columns=list("abc"))
    frame["y"] = frame["a"] * 2 + rng.normal(scale=0.1, size=60)
    result = saltml.search(load(frame), learners=["tabpfn"], n_trials=1, folds=3, seed=0)
    assert result.task is Task.REGRESSION
    assert result.n_failed == 0
