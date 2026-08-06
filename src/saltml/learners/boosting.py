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


class _Adapter:
    """Base for the two estimators that need adapting to scikit-learn.

    scikit-learn clones an estimator for every cross-validation fold, and
    ``clone`` insists that ``type(est)(**est.get_params())`` return the very
    same parameter objects — an identity check, not equality. Keeping the
    parameters in one dict and handing back a shallow copy satisfies that,
    which the wrapped libraries' own ``get_params`` do not always do.
    """

    _estimator_kind = "classifier"

    def __init__(self, **params: Any) -> None:
        self._params = params

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return dict(self._params)

    def set_params(self, **params: Any) -> "_Adapter":
        self._params.update(params)
        return self

    def __sklearn_tags__(self):  # scikit-learn >= 1.6 estimator introspection
        from sklearn.utils import Tags, TargetTags

        return Tags(
            estimator_type=self._estimator_kind,
            target_tags=TargetTags(required=True),
            classifier_tags=None,
            regressor_tags=None,
            transformer_tags=None,
        )

    def __sklearn_is_fitted__(self) -> bool:
        # scikit-learn otherwise infers fittedness from trailing-underscore
        # attributes, which a wrapper holding its model privately has none of;
        # without this a fitted regressor is reported as unfitted.
        return hasattr(self, "_model")


class _XGBClassifierWithLabels(_Adapter):
    """XGBClassifier that accepts arbitrary class labels.

    XGBoost 3 requires the target to be exactly ``[0..n_classes-1]`` and
    rejects string labels outright, while every other classifier here takes
    them as they come. Encoding happens inside fit/predict so the rest of the
    system does not have to know, and so the labels reported back to the user
    are the ones they supplied.
    """

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


def _xgboost_classifier(**params: Any) -> Any:
    return _XGBClassifierWithLabels(**params)


class _CatBoost(_Adapter):
    """CatBoost with ``cat_features`` supplied at fit time.

    CatBoost's own ``get_params`` returns a fresh list for ``cat_features``
    each call, so scikit-learn's identity-based clone check rejects it and
    the learner fails on every fold while working fine in a direct ``fit``.
    Holding the value here and passing it to ``fit`` sidesteps the round trip
    entirely.
    """

    def fit(self, X: Any, y: Any) -> "_CatBoost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        params = dict(self._params)
        categorical = params.pop("cat_features", None)
        model_type = (
            CatBoostClassifier if self._estimator_kind == "classifier" else CatBoostRegressor
        )
        self._model = model_type(**params)
        self._model.fit(X, y, cat_features=categorical or None)
        if self._estimator_kind == "classifier":
            self.classes_ = self._model.classes_
        return self

    def predict(self, X: Any) -> Any:
        prediction = self._model.predict(X)
        # CatBoost returns a column vector for classification; sklearn's
        # scorers expect one dimension.
        return prediction.ravel() if hasattr(prediction, "ravel") else prediction

    def predict_proba(self, X: Any) -> Any:
        return self._model.predict_proba(X)


class _CatBoostRegressorAdapter(_CatBoost):
    _estimator_kind = "regressor"


def _xgboost_regressor(**params: Any) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(**params)


def _catboost_classifier(**params: Any) -> Any:
    return _CatBoost(**params)


def _catboost_regressor(**params: Any) -> Any:
    return _CatBoostRegressorAdapter(**params)


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
