"""TabPFN — a transformer that classifies tabular data in one forward pass.

Unlike everything else in the registry, TabPFN is not trained on your data: a
network pre-trained on synthetic datasets takes the training rows as context
and predicts in a single pass. That makes it very strong on small tables and
very unlike the other learners in two ways worth knowing about.

It has almost no hyperparameters to search — the ensemble size and a
temperature, not a space with structure — so the search spends few trials on
it by design.

And it has hard size limits inherited from pre-training, beyond which it
refuses to run. Rather than let every trial fail, the learner declares those
limits and is excluded up front with a reason.

Optional dependency. ``pip install 'salt-ml[tabpfn]'``. Version 2.x downloads
its weights without an account; 8.x requires a licence token in
``TABPFN_TOKEN``. Both expose the same constructor arguments, so this module
works with either.
"""

from __future__ import annotations

import logging
from typing import Any

from ..task import Task
from .base import Learner, Space

__all__ = ["TABPFN_LEARNERS", "tabpfn_available"]

log = logging.getLogger("saltml")

# Pre-training limits for the v2 line. Exceeding them raises rather than
# degrading, so they are treated as hard applicability bounds.
MAX_SAMPLES = 10_000
MAX_FEATURES = 500
MAX_CLASSES = 10


def tabpfn_available() -> bool:
    try:
        import tabpfn  # noqa: F401
    except Exception:  # pragma: no cover - depends on install
        return False
    return True


def _encoded_width(dataset: Any) -> int:
    """Feature count after one-hot encoding, which is what TabPFN will see."""
    width = len(dataset.numeric_columns)
    for column in dataset.categorical_columns:
        width += max(1, int(dataset.X[column].nunique(dropna=True)))
    return width


def _applies(dataset: Any) -> str | None:
    if dataset.n_samples > MAX_SAMPLES:
        return f"{dataset.n_samples:,} samples exceeds TabPFN's {MAX_SAMPLES:,} limit"
    width = _encoded_width(dataset)
    if width > MAX_FEATURES:
        return f"{width} encoded features exceeds TabPFN's {MAX_FEATURES} limit"
    if dataset.task is Task.CLASSIFICATION:
        n_classes = int(dataset.y.nunique())
        if n_classes > MAX_CLASSES:
            return f"{n_classes} classes exceeds TabPFN's {MAX_CLASSES} limit"
    return None


def _space(s: Space) -> dict[str, Any]:
    # Deliberately small. More ensemble members cost close to linear time on
    # CPU for a modest gain, so the upper bound stays low enough that TabPFN
    # does not swallow a whole trial budget.
    return {
        "n_estimators": s.int("n_estimators", 1, 8, log=True),
        "softmax_temperature": s.float("softmax_temperature", 0.5, 1.5),
    }


def _regressor_space(s: Space) -> dict[str, Any]:
    return {
        "n_estimators": s.int("n_estimators", 1, 8, log=True),
        "softmax_temperature": s.float("softmax_temperature", 0.5, 1.5),
    }


def _classifier_factory(**params: Any) -> Any:
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(**params)


def _regressor_factory(**params: Any) -> Any:
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(**params)


TABPFN_LEARNERS: list[Learner] = [
    Learner("tabpfn", Task.CLASSIFICATION, _classifier_factory, _space, applies=_applies),
    Learner("tabpfn", Task.REGRESSION, _regressor_factory, _regressor_space, applies=_applies),
]
