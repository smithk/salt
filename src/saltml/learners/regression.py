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
    params: dict[str, Any] = {
        "n_estimators": s.int("n_estimators", 100, 800, log=True),
        "max_features": s.cat("max_features", ["sqrt", "log2", None]),
        "min_samples_leaf": s.int("min_samples_leaf", 1, 20),
        "n_jobs": 1,
    }
    if s.cat("limit_depth", [False, True]):
        params["max_depth"] = s.int("max_depth", 2, 32)
    return params


def _hist_gradient_boosting(s: Space) -> dict[str, Any]:
    return {
        "learning_rate": s.float("learning_rate", 0.01, 0.5, log=True),
        "max_iter": s.int("max_iter", 50, 500, log=True),
        "max_leaf_nodes": s.int("max_leaf_nodes", 15, 255, log=True),
        "min_samples_leaf": s.int("min_samples_leaf", 5, 100, log=True),
        "l2_regularization": s.float("l2_regularization", 1e-8, 1.0, log=True),
    }


def _decision_tree(s: Space) -> dict[str, Any]:
    return {
        "max_depth": s.int("max_depth", 2, 32),
        "min_samples_leaf": s.int("min_samples_leaf", 1, 20),
    }


def _svr(s: Space) -> dict[str, Any]:
    kernel = s.cat("kernel", ["rbf", "linear"])
    params: dict[str, Any] = {
        "kernel": kernel,
        "C": s.float("C", 1e-3, 1e3, log=True),
        "epsilon": s.float("epsilon", 1e-3, 1.0, log=True),
        "cache_size": 500,
    }
    if kernel == "rbf":
        params["gamma"] = s.float("gamma", 1e-5, 1e1, log=True)
    return params


def _knn(s: Space) -> dict[str, Any]:
    return {
        "n_neighbors": s.int("n_neighbors", 1, 50, log=True),
        "weights": s.cat("weights", ["uniform", "distance"]),
        "p": s.cat("p", [1, 2]),
        "n_jobs": 1,
    }


#: Same shapes as the classifier; see the note there on why architectures are
#: drawn from a named list rather than searched as width and depth.
_MLP_SHAPES = ["64", "128", "256", "64x64", "128x64", "256x128"]


def _mlp_shape(name: str) -> tuple[int, ...]:
    return tuple(int(width) for width in name.split("x"))


def _mlp(s: Space) -> dict[str, Any]:
    return {
        "hidden_layer_sizes": _mlp_shape(s.cat("hidden_layer_sizes", _MLP_SHAPES)),
        "alpha": s.float("alpha", 1e-6, 1e1, log=True),
        "learning_rate_init": s.float("learning_rate_init", 1e-4, 1e-1, log=True),
        "batch_size": s.cat("batch_size", [32, 128, "auto"]),
        "solver": "adam",
        "early_stopping": True,
        "n_iter_no_change": 10,
        "max_iter": 500,
    }


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
