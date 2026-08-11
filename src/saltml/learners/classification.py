"""Classification algorithms and their search spaces."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ..task import Task
from . import spaces
from .base import Learner, Space

__all__ = ["CLASSIFIERS"]


def _sklearn_at_least(major: int, minor: int) -> bool:
    import sklearn

    parts = sklearn.__version__.split(".")
    try:
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except (IndexError, ValueError):  # pragma: no cover - dev builds
        return True


#: scikit-learn 1.8 deprecated ``penalty`` in favour of a continuous
#: ``l1_ratio``, and pairing the old argument with the new default silently
#: produces a model that is neither. Detect once, translate at construction.
_LOGISTIC_USES_L1_RATIO = _sklearn_at_least(1, 8)


def _logistic_factory(
    *, penalty: str, C: float, class_weight: Any, max_iter: int, random_state: Any = None
) -> LogisticRegression:
    """Build a LogisticRegression with the argument spelling this version wants.

    The search space stays the same across versions — a categorical l1/l2
    choice — so study results remain comparable.
    """
    common = {"C": C, "class_weight": class_weight, "max_iter": max_iter,
              "random_state": random_state}
    if _LOGISTIC_USES_L1_RATIO:
        return LogisticRegression(
            l1_ratio=1.0 if penalty == "l1" else 0.0,
            solver="saga" if penalty == "l1" else "lbfgs",
            **common,
        )
    return LogisticRegression(
        penalty=penalty, solver="saga" if penalty == "l1" else "lbfgs", **common
    )


def _logistic_regression(s: Space) -> dict[str, Any]:
    # Both solvers used here handle multiclass natively; liblinear does not,
    # so it is deliberately absent.
    return {
        "penalty": s.cat("penalty", ["l1", "l2"]),
        "C": s.float("C", 1e-4, 1e4, log=True),
        "class_weight": s.cat("class_weight", [None, "balanced"]),
        "max_iter": 5000,
    }


def _ridge_classifier(s: Space) -> dict[str, Any]:
    return {
        "alpha": s.float("alpha", 1e-4, 1e4, log=True),
        "class_weight": s.cat("class_weight", [None, "balanced"]),
    }


def _class_weight(s: Space) -> Any:
    """Every classifier here can rebalance; no regressor has the notion."""
    return s.cat("class_weight", [None, "balanced"])


def _forest(s: Space) -> dict[str, Any]:
    return {**spaces.forest(s), "class_weight": _class_weight(s)}


def _hist_gradient_boosting(s: Space) -> dict[str, Any]:
    return spaces.hist_gradient_boosting(s)


def _decision_tree(s: Space) -> dict[str, Any]:
    return {
        **spaces.decision_tree(s),
        "criterion": s.cat("criterion", ["gini", "entropy"]),
        "class_weight": _class_weight(s),
    }


def _svc(s: Space) -> dict[str, Any]:
    return {**spaces.kernel_machine(s), "class_weight": _class_weight(s)}


def _knn(s: Space) -> dict[str, Any]:
    return spaces.knn(s)


def _gaussian_nb(s: Space) -> dict[str, Any]:
    return {"var_smoothing": s.float("var_smoothing", 1e-12, 1e-4, log=True)}


class _MLPClassifierWithLabels(MLPClassifier):
    """MLPClassifier that survives ``early_stopping`` with non-numeric labels.

    With ``early_stopping=True`` scikit-learn scores a held-out slice each
    iteration through ``_score_with_function``, which guards against a
    diverged net with ``np.isnan(y_pred)``. On a classifier ``y_pred`` holds
    class labels, so when those are strings — as they are in most of the ARFF
    corpus — the guard raises ``TypeError: ufunc 'isnan' not supported`` and
    every fold fails. Encoding the target to integers inside ``fit`` avoids it
    and keeps the labels the caller supplied on the way out, the same trick
    ``_XGBClassifierWithLabels`` uses for a different library's version of the
    same problem.
    """

    def fit(self, X: Any, y: Any) -> "_MLPClassifierWithLabels":
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder().fit(y)
        super().fit(X, encoder.transform(y))
        self._encoder = encoder
        # Set after fitting: the parent leaves the encoded labels here, and
        # predict_proba's columns follow this order either way, since
        # LabelEncoder sorts exactly as the parent does.
        self.classes_ = encoder.classes_
        return self

    def predict(self, X: Any) -> Any:
        return self._encoder.inverse_transform(super().predict(X))


def _mlp(s: Space) -> dict[str, Any]:
    return spaces.mlp(s)


CLASSIFIERS: list[Learner] = [
    Learner("logistic_regression", Task.CLASSIFICATION, _logistic_factory,
            _logistic_regression, needs_scaling=True),
    Learner("ridge_classifier", Task.CLASSIFICATION, RidgeClassifier,
            _ridge_classifier, needs_scaling=True),
    Learner("random_forest", Task.CLASSIFICATION, RandomForestClassifier, _forest),
    Learner("extra_trees", Task.CLASSIFICATION, ExtraTreesClassifier, _forest),
    Learner("hist_gradient_boosting", Task.CLASSIFICATION,
            HistGradientBoostingClassifier, _hist_gradient_boosting),
    Learner("decision_tree", Task.CLASSIFICATION, DecisionTreeClassifier, _decision_tree),
    Learner("svm", Task.CLASSIFICATION, SVC, _svc, needs_scaling=True),
    Learner("knn", Task.CLASSIFICATION, KNeighborsClassifier, _knn,
            needs_scaling=True, seedable=False),
    Learner("gaussian_nb", Task.CLASSIFICATION, GaussianNB, _gaussian_nb, seedable=False),
    Learner("mlp", Task.CLASSIFICATION, _MLPClassifierWithLabels, _mlp, needs_scaling=True),
]
