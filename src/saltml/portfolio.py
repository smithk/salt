"""Configurations worth trying before the search has learned anything.

A search that begins from the prior spends its first trials rediscovering that
gradient boosting usually wants a few hundred trees and a small learning rate.
That is knowledge the field already has, and on a sixty-second budget those
trials are a meaningful fraction of everything the tool will ever do.

So the study starts with a short queue of configurations that are strong across
a wide range of tabular data — library defaults where those are sensible, and
the settings that repeatedly win otherwise. Optuna evaluates a queued trial as
an ordinary trial, so nothing here is trusted: a portfolio entry that does
badly on *this* dataset is simply a trial that scored badly, and the sampler
moves on. The gain is a better starting point, not a shortcut past measurement.

This is the cheap half of what auto-sklearn calls meta-learning. The expensive
half — picking entries by dataset similarity — needs a meta-feature model and a
corpus of prior runs; the plain fixed portfolio captures much of the benefit at
a fraction of the machinery.

Entries are keyed by learner and must name parameters exactly as the learner's
space declares them, within the declared ranges; ``tests/test_portfolio.py``
enforces both against the real search spaces.
"""

from __future__ import annotations

import logging
from typing import Any

import optuna

from .task import Task

__all__ = ["PORTFOLIO", "enqueue_portfolio"]

log = logging.getLogger("saltml")


#: Per-learner starting configurations, best-known-general first.
PORTFOLIO: dict[Task, dict[str, list[dict[str, Any]]]] = {
    Task.CLASSIFICATION: {
        "hist_gradient_boosting": [
            {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31,
             "min_samples_leaf": 20, "l2_regularization": 1e-8},
            {"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 63,
             "min_samples_leaf": 10, "l2_regularization": 1.0},
        ],
        "random_forest": [
            {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 1,
             "class_weight": None, "limit_depth": False},
            {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 3,
             "class_weight": "balanced", "limit_depth": False},
        ],
        "logistic_regression": [
            {"penalty": "l2", "C": 1.0, "class_weight": None},
            {"penalty": "l2", "C": 10.0, "class_weight": "balanced"},
        ],
        "svm": [
            # Scaling is applied for this learner, so unit-ish C and a small
            # gamma are the usual starting point rather than a guess.
            {"kernel": "rbf", "C": 1.0, "gamma": 0.1, "class_weight": None},
            {"kernel": "linear", "C": 1.0, "class_weight": None},
        ],
        "lightgbm": [
            {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31,
             "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
             "reg_lambda": 1e-8},
        ],
        "xgboost": [
            {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
             "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0,
             "reg_lambda": 1.0},
        ],
        "knn": [
            {"n_neighbors": 5, "weights": "uniform", "p": 2},
        ],
    },
    Task.REGRESSION: {
        "hist_gradient_boosting": [
            {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31,
             "min_samples_leaf": 20, "l2_regularization": 1e-8},
            {"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 63,
             "min_samples_leaf": 10, "l2_regularization": 1.0},
        ],
        "random_forest": [
            {"n_estimators": 500, "max_features": None, "min_samples_leaf": 1,
             "limit_depth": False},
        ],
        "ridge": [
            {"alpha": 1.0},
        ],
        "elastic_net": [
            {"alpha": 0.01, "l1_ratio": 0.5},
        ],
        "svr": [
            {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": 0.1},
        ],
        "lightgbm": [
            {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31,
             "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
             "reg_lambda": 1e-8},
        ],
        "xgboost": [
            {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
             "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0,
             "reg_lambda": 1.0},
        ],
        "knn": [
            {"n_neighbors": 5, "weights": "uniform", "p": 2},
        ],
    },
}


def entries(task: Task, offered: list[str]) -> list[dict[str, Any]]:
    """Portfolio trials for the learners on offer, as Optuna parameter dicts.

    Ordered breadth-first — every learner's first entry before any learner's
    second — so that a budget too small to exhaust the queue still covers the
    range of algorithms rather than tuning one of them.
    """
    by_learner = PORTFOLIO.get(task, {})
    available = [name for name in offered if name in by_learner]
    if not available:
        return []

    queued: list[dict[str, Any]] = []
    for rank in range(max(len(by_learner[name]) for name in available)):
        for name in available:
            if rank >= len(by_learner[name]):
                continue
            params = {"learner": name}
            params.update({f"{name}__{key}": value for key, value in by_learner[name][rank].items()})
            queued.append(params)
    return queued


def enqueue_portfolio(study: optuna.Study, task: Task, offered: list[str], *,
                      limit: int | None = None) -> int:
    """Queue portfolio configurations on a study. Returns how many were queued."""
    queued = entries(task, offered)
    if limit is not None:
        queued = queued[:limit]
    for params in queued:
        # skip_if_exists guards the time-budget path, where the focus phase can
        # be handed learners the survey already tried these settings on.
        study.enqueue_trial(params, skip_if_exists=True)
    if queued:
        log.info("Portfolio: queued %d starting configuration(s).", len(queued))
    return len(queued)
