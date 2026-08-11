"""Regression algorithms and their search spaces."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ..task import Task
from . import spaces
from .base import Learner, Space

__all__ = ["REGRESSORS"]


def _ridge(s: Space) -> dict[str, Any]:
    return {"alpha": s.float("alpha", 1e-4, 1e4, log=True)}


def _lasso(s: Space) -> dict[str, Any]:
    return {"alpha": s.float("alpha", 1e-4, 1e2, log=True), "max_iter": 5000}


def _elastic_net(s: Space) -> dict[str, Any]:
    return {
        "alpha": s.float("alpha", 1e-4, 1e2, log=True),
        "l1_ratio": s.float("l1_ratio", 0.01, 1.0),
        "max_iter": 5000,
    }


def _forest(s: Space) -> dict[str, Any]:
    return spaces.forest(s)


def _hist_gradient_boosting(s: Space) -> dict[str, Any]:
    return spaces.hist_gradient_boosting(s)


def _decision_tree(s: Space) -> dict[str, Any]:
    return spaces.decision_tree(s)


def _svr(s: Space) -> dict[str, Any]:
    # epsilon is the width of the band inside which errors cost nothing, which
    # only means something when the target is continuous.
    return {**spaces.kernel_machine(s), "epsilon": s.float("epsilon", 1e-3, 1.0, log=True)}


def _knn(s: Space) -> dict[str, Any]:
    return spaces.knn(s)


def _mlp(s: Space) -> dict[str, Any]:
    return spaces.mlp(s)


REGRESSORS: list[Learner] = [
    Learner("ridge", Task.REGRESSION, Ridge, _ridge, needs_scaling=True),
    Learner("lasso", Task.REGRESSION, Lasso, _lasso, needs_scaling=True),
    Learner("elastic_net", Task.REGRESSION, ElasticNet, _elastic_net, needs_scaling=True),
    Learner("random_forest", Task.REGRESSION, RandomForestRegressor, _forest),
    Learner("extra_trees", Task.REGRESSION, ExtraTreesRegressor, _forest),
    Learner("hist_gradient_boosting", Task.REGRESSION,
            HistGradientBoostingRegressor, _hist_gradient_boosting),
    Learner("decision_tree", Task.REGRESSION, DecisionTreeRegressor, _decision_tree),
    Learner("svr", Task.REGRESSION, SVR, _svr, needs_scaling=True, seedable=False),
    Learner("knn", Task.REGRESSION, KNeighborsRegressor, _knn,
            needs_scaling=True, seedable=False),
    Learner("mlp", Task.REGRESSION, MLPRegressor, _mlp, needs_scaling=True),
]
