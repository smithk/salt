"""Search spaces shared by both tasks.

scikit-learn splits its estimators by task — ``RandomForestClassifier`` and
``RandomForestRegressor`` are different classes — and this package mirrors that
split, because the loss, the metric, the fold policy and whether
``predict_proba`` exists all genuinely differ.

The *search space* usually does not. A forest wants the same number of trees
and the same leaf size whichever target it is fitted against, so those ranges
live here once rather than in both task modules. What a task adds on top —
``class_weight`` for a classifier, ``epsilon`` for an SVR — stays in the module
that owns it.

Anything defined here is drawn before the task-specific additions, so the
parameter names are identical across tasks but the draw order differs slightly
from the hand-written versions these replaced. That changes which
configurations a given seed happens to visit; it does not change the space.
"""

from __future__ import annotations

from typing import Any

from .base import Space

__all__ = [
    "MLP_SHAPES",
    "decision_tree",
    "forest",
    "hist_gradient_boosting",
    "kernel_machine",
    "knn",
    "mlp",
    "mlp_shape",
]


def forest(s: Space) -> dict[str, Any]:
    """Random forest and extra trees, minus anything task-specific."""
    params: dict[str, Any] = {
        "n_estimators": s.int("n_estimators", 100, 800, log=True),
        "max_features": s.cat("max_features", ["sqrt", "log2", None]),
        "min_samples_leaf": s.int("min_samples_leaf", 1, 20),
        "n_jobs": 1,  # parallelism is spent on folds, not inside one fit
    }
    # Unlimited depth is the useful default, so only draw a depth when limiting.
    if s.cat("limit_depth", [False, True]):
        params["max_depth"] = s.int("max_depth", 2, 32)
    return params


def hist_gradient_boosting(s: Space) -> dict[str, Any]:
    """Identical for both tasks: the loss differs, nothing searched does."""
    return {
        "learning_rate": s.float("learning_rate", 0.01, 0.5, log=True),
        "max_iter": s.int("max_iter", 50, 500, log=True),
        "max_leaf_nodes": s.int("max_leaf_nodes", 15, 255, log=True),
        "min_samples_leaf": s.int("min_samples_leaf", 5, 100, log=True),
        "l2_regularization": s.float("l2_regularization", 1e-8, 1.0, log=True),
    }


def decision_tree(s: Space) -> dict[str, Any]:
    """Shape only. The split criterion is named differently per task."""
    return {
        "max_depth": s.int("max_depth", 2, 32),
        "min_samples_leaf": s.int("min_samples_leaf", 1, 20),
    }


def kernel_machine(s: Space) -> dict[str, Any]:
    """SVC and SVR, minus ``class_weight`` and ``epsilon`` respectively."""
    kernel = s.cat("kernel", ["rbf", "linear"])
    params: dict[str, Any] = {
        "kernel": kernel,
        "C": s.float("C", 1e-3, 1e3, log=True),
        "cache_size": 500,
    }
    if kernel == "rbf":
        params["gamma"] = s.float("gamma", 1e-5, 1e1, log=True)
    return params


def knn(s: Space) -> dict[str, Any]:
    return {
        "n_neighbors": s.int("n_neighbors", 1, 50, log=True),
        "weights": s.cat("weights", ["uniform", "distance"]),
        "p": s.cat("p", [1, 2]),
        "n_jobs": 1,
    }


#: Architectures rather than a free width/depth search. Two numbers that
#: interact this strongly would spend the budget on combinations that are
#: obviously too small or too slow; this is the range worth trying on tabular
#: data, from a single narrow layer to a modest two-layer net.
#:
#: Named as strings rather than tuples because Optuna warns that a categorical
#: choice should be a plain scalar — tuples work in memory but do not survive a
#: persistent study.
MLP_SHAPES = ["64", "128", "256", "64x64", "128x64", "256x128"]


def mlp_shape(name: str) -> tuple[int, ...]:
    return tuple(int(width) for width in name.split("x"))


def mlp(s: Space) -> dict[str, Any]:
    return {
        "hidden_layer_sizes": mlp_shape(s.cat("hidden_layer_sizes", MLP_SHAPES)),
        # The two that decide whether an MLP works at all on tabular data: too
        # little regularisation and it memorises, too high a learning rate and
        # it never settles.
        "alpha": s.float("alpha", 1e-6, 1e1, log=True),
        "learning_rate_init": s.float("learning_rate_init", 1e-4, 1e-1, log=True),
        "batch_size": s.cat("batch_size", [32, 128, "auto"]),
        # Adam only. The search compares learners, and lbfgs/sgd here would
        # mostly measure which optimiser suits the budget rather than whether a
        # neural network suits the data.
        "solver": "adam",
        # Stop when a held-out slice stops improving rather than burning the
        # full iteration count on a net that converged long ago — the same
        # bargain the pruner makes at the level of trials.
        "early_stopping": True,
        "n_iter_no_change": 10,
        "max_iter": 500,
    }
