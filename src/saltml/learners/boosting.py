"""Gradient-boosted trees.

The three dedicated boosting libraries are what usually win on tabular data,
which is the whole point of pointing SALT at a table. They are optional
dependencies (``pip install 'salt-ml[boost]'``) because each is large and needs
a working OpenMP runtime, so a missing one shows up as an absent learner rather
than an import error.

Every one of them is single-threaded here (``n_jobs=1``, ``thread_count=1``):
parallelism is spent across cross-validation folds instead, and letting both
levels grab every core oversubscribes the machine and makes trials slower, not
faster.
"""

from __future__ import annotations

from typing import Any

from ..task import Task
from .base import Learner, Space

__all__ = ["BOOSTING_LEARNERS", "available"]


def _has(module: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def available() -> dict[str, bool]:
    """Which boosting libraries can be imported."""
    return {name: _has(name) for name in ("lightgbm", "xgboost", "catboost")}


# --- search spaces -------------------------------------------------------
# The ranges below are the usual competition-tuning ranges: enough trees to
# matter, learning rates spanning two decades, and both row and column
# subsampling, which is where most of the regularisation comes from.

def _lightgbm_space(s: Space) -> dict[str, Any]:
    return {
        "n_estimators": s.int("n_estimators", 50, 1000, log=True),
        "learning_rate": s.float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": s.int("num_leaves", 15, 255, log=True),
        "min_child_samples": s.int("min_child_samples", 5, 100, log=True),
        "subsample": s.float("subsample", 0.5, 1.0),
        "subsample_freq": 1,          # subsample is ignored unless this is set
        "colsample_bytree": s.float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": s.float("reg_lambda", 1e-8, 10.0, log=True),
        "n_jobs": 1,
        "verbose": -1,
    }


def _xgboost_space(s: Space) -> dict[str, Any]:
    return {
        "n_estimators": s.int("n_estimators", 50, 1000, log=True),
        "learning_rate": s.float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": s.int("max_depth", 2, 12),
        "min_child_weight": s.float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": s.float("subsample", 0.5, 1.0),
        "colsample_bytree": s.float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": s.float("reg_lambda", 1e-8, 10.0, log=True),
        "tree_method": "hist",
        "n_jobs": 1,
        "verbosity": 0,
    }


def _catboost_space(s: Space) -> dict[str, Any]:
    return {
        "iterations": s.int("iterations", 100, 1000, log=True),
        "learning_rate": s.float("learning_rate", 0.01, 0.3, log=True),
        "depth": s.int("depth", 4, 10),
        "l2_leaf_reg": s.float("l2_leaf_reg", 1.0, 30.0, log=True),
        "thread_count": 1,
        "verbose": 0,
        "allow_writing_files": False,   # CatBoost writes catboost_info/ otherwise
    }


# --- factories -----------------------------------------------------------
# Imported inside the factory so that an absent library costs nothing at
# import time and the learner simply never registers.

def _lightgbm_classifier(**params: Any) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**params)


def _lightgbm_regressor(**params: Any) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**params)


class _XGBClassifierWithLabels:
    """XGBClassifier that accepts arbitrary class labels.

    XGBoost 3 requires the target to be exactly ``[0..n_classes-1]`` and
    rejects string labels outright, while every other classifier here takes
    them as they come. Encoding is done inside fit/predict so the rest of the
    system does not have to know, and so the labels reported back to the user
    are the ones they supplied.

    Implements get_params/set_params because scikit-learn clones estimators
    for every cross-validation fold.
    """

    def __init__(self, **params: Any) -> None:
        self._params = params

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return dict(self._params)

    def set_params(self, **params: Any) -> "_XGBClassifierWithLabels":
        self._params.update(params)
        return self

    def fit(self, X: Any, y: Any) -> "_XGBClassifierWithLabels":
        from sklearn.preprocessing import LabelEncoder
        from xgboost import XGBClassifier

        self._encoder = LabelEncoder().fit(y)
        self.classes_ = self._encoder.classes_
        self._model = XGBClassifier(**self._params)
        self._model.fit(X, self._encoder.transform(y))
        return self

    def predict(self, X: Any) -> Any:
        return self._encoder.inverse_transform(self._model.predict(X))

    def predict_proba(self, X: Any) -> Any:
        return self._model.predict_proba(X)

    def __sklearn_tags__(self):  # scikit-learn >= 1.6 estimator introspection
        from sklearn.utils import Tags, TargetTags

        return Tags(
            estimator_type="classifier",
            target_tags=TargetTags(required=True),
            classifier_tags=None,
            regressor_tags=None,
            transformer_tags=None,
        )


def _xgboost_classifier(**params: Any) -> Any:
    return _XGBClassifierWithLabels(**params)


def _xgboost_regressor(**params: Any) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(**params)


def _catboost_classifier(**params: Any) -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(**params)


def _catboost_regressor(**params: Any) -> Any:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(**params)


def _build() -> list[Learner]:
    found = available()
    learners: list[Learner] = []

    if found["lightgbm"]:
        learners += [
            Learner("lightgbm", Task.CLASSIFICATION, _lightgbm_classifier, _lightgbm_space),
            Learner("lightgbm", Task.REGRESSION, _lightgbm_regressor, _lightgbm_space),
        ]
    if found["xgboost"]:
        learners += [
            Learner("xgboost", Task.CLASSIFICATION, _xgboost_classifier, _xgboost_space),
            Learner("xgboost", Task.REGRESSION, _xgboost_regressor, _xgboost_space),
        ]
    if found["catboost"]:
        # CatBoost's ordered target statistics beat one-hot encoding, and
        # decisively so on high-cardinality columns where one-hot explodes the
        # feature count. It is the one learner given the raw columns.
        learners += [
            Learner("catboost", Task.CLASSIFICATION, _catboost_classifier,
                    _catboost_space, handles_categorical=True),
            Learner("catboost", Task.REGRESSION, _catboost_regressor,
                    _catboost_space, handles_categorical=True),
        ]
    return learners


BOOSTING_LEARNERS: list[Learner] = _build()
