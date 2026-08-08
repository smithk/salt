"""Combining the best trials instead of keeping only the winner.

A search fits hundreds of models and then throws all but one away. The runner
up is usually almost as good and wrong about different rows, which is exactly
the condition under which averaging helps. Building an ensemble from trials
that have *already been paid for* is close to free next to the search itself.

Selection is Caruana's greedy procedure: start empty, repeatedly add whichever
candidate most improves the ensemble's score, allowing the same candidate to be
picked again. Choosing with replacement is what turns a selection method into a
weighting one — a model picked three times out of ten carries weight 0.3 — and
the greedy step means a candidate only joins if it improves *the ensemble*,
which a plain top-k average does not check.

Scoring during selection uses out-of-fold predictions, never the training rows
a model was fitted on. Selecting on in-sample predictions would hand the whole
ensemble to whichever candidate overfits hardest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_predict

from .data import Dataset
from .task import Task

__all__ = ["EnsembleModel", "build_ensemble"]

log = logging.getLogger("saltml")

#: Trials considered for membership, best-scoring first. Beyond this the
#: candidates are too weak to earn a place and only cost fits.
DEFAULT_CANDIDATES = 10
#: Greedy rounds. Also the finest weight the ensemble can express: 1/size.
DEFAULT_SIZE = 10


@dataclass
class EnsembleModel:
    """A weighted vote over fitted pipelines.

    Deliberately not a scikit-learn estimator: it is built from an existing
    search rather than fitted from scratch, so ``fit`` would have nothing
    sensible to do.
    """

    task: Task
    models: list[Any]
    weights: list[float]
    members: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    classes_: np.ndarray | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task is not Task.CLASSIFICATION:
            raise AttributeError("predict_proba is classification-only")
        total = None
        for model, weight in zip(self.models, self.weights):
            # Same treatment as during selection: a member with no
            # probabilities votes for its predicted class at full confidence.
            if hasattr(model, "predict_proba"):
                proba = np.asarray(model.predict_proba(X), dtype=float)
            else:
                proba = _one_hot(np.asarray(model.predict(X)), self.classes_)
            total = proba * weight if total is None else total + proba * weight
        return total

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.task is Task.CLASSIFICATION:
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        total = None
        for model, weight in zip(self.models, self.weights):
            pred = np.asarray(model.predict(X), dtype=float) * weight
            total = pred if total is None else total + pred
        return total

    def describe(self) -> pd.DataFrame:
        """Who is in the ensemble and how much say each one has."""
        return pd.DataFrame(
            [
                {"learner": name, "weight": round(weight, 3), "params": params}
                for (name, params), weight in zip(self.members, self.weights)
            ]
        ).sort_values("weight", ascending=False, ignore_index=True)


class _Precomputed:
    """Adapter presenting stored predictions through the estimator API.

    Lets the same scikit-learn scorer that graded the search grade a candidate
    ensemble, rather than re-deriving each metric's conventions by hand.
    """

    def __init__(self, task: Task, proba: np.ndarray | None, pred: np.ndarray,
                 classes: np.ndarray | None) -> None:
        self._proba = proba
        self._pred = pred
        self.classes_ = classes
        self._estimator_type = (
            "classifier" if task is Task.CLASSIFICATION else "regressor"
        )

    def predict(self, X: Any) -> np.ndarray:
        return self._pred

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._proba is None:
            raise AttributeError("no probabilities available")
        return self._proba


def _one_hot(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Hard labels as a probability matrix, so every candidate blends alike."""
    index = {label: i for i, label in enumerate(classes)}
    wide = np.zeros((len(labels), len(classes)), dtype=float)
    for row, label in enumerate(labels):
        wide[row, index[label]] = 1.0
    return wide


def _blend(task: Task, stacked: np.ndarray, picks: list[int],
           classes: np.ndarray | None) -> _Precomputed:
    """Average the chosen candidates' out-of-fold predictions."""
    mean = stacked[picks].mean(axis=0)
    if task is Task.CLASSIFICATION:
        return _Precomputed(task, mean, classes[np.argmax(mean, axis=1)], classes)
    return _Precomputed(task, None, mean, None)


def build_ensemble(
    dataset: Dataset,
    result: Any,
    *,
    size: int = DEFAULT_SIZE,
    candidates: int = DEFAULT_CANDIDATES,
    folds: int = 5,
    seed: int | None = 0,
    n_jobs: int = -1,
) -> EnsembleModel | None:
    """Build a greedy ensemble from a completed search.

    Returns ``None`` when there is nothing to combine — fewer than two distinct
    configurations survived, so the single best model already is the answer.
    """
    from .learners import resolve
    from .search import build_pipeline
    from .search import _splitter  # the search's own fold policy, reused as-is

    ranked = sorted(result.records, key=lambda r: r.score, reverse=True)
    seen: set[tuple[str, str]] = set()
    chosen: list[Any] = []
    for record in ranked:
        key = (record.learner, repr(sorted(record.params.items(), key=str)))
        if key in seen:
            continue
        seen.add(key)
        chosen.append(record)
        if len(chosen) >= candidates:
            break

    if len(chosen) < 2:
        log.info("Ensemble skipped: only %d distinct configuration(s).", len(chosen))
        return None

    splitter = _splitter(dataset, folds, seed)
    scorer = get_scorer(result.metric)
    classification = dataset.task is Task.CLASSIFICATION
    classes = np.unique(dataset.y) if classification else None

    out_of_fold: list[np.ndarray] = []
    usable: list[Any] = []
    for record in chosen:
        learner = resolve(dataset.task, [record.learner])[0]
        pipeline = build_pipeline(dataset, learner, record.params)
        # Not every classifier offers probabilities — an SVC is fitted with
        # `probability=False` and a ridge classifier has none at all. Those
        # still deserve a place, so their hard labels are widened into a
        # one-hot matrix and they vote at full confidence.
        soft = classification and hasattr(pipeline, "predict_proba")
        try:
            predictions = cross_val_predict(
                pipeline,
                dataset.X,
                dataset.y,
                cv=splitter,
                n_jobs=n_jobs,
                method="predict_proba" if soft else "predict",
            )
        except Exception as exc:  # a candidate that will not cooperate is skipped
            log.debug("Ensemble candidate %s skipped: %s", record.learner, exc)
            continue
        predictions = np.asarray(predictions)
        if classification and not soft:
            predictions = _one_hot(predictions, classes)
        out_of_fold.append(np.asarray(predictions, dtype=float))
        usable.append(record)

    if len(usable) < 2:
        return None

    stacked = np.stack(out_of_fold)
    # Every round adds the candidate that most improves the blend, duplicates
    # allowed — that is what expresses weight. The rounds always run to the
    # end and the best-scoring prefix wins, rather than stopping at the first
    # round that fails to improve: re-picking the leader is score-neutral, so
    # an improvement test would halt on it immediately and never find the
    # combinations that lie one step past a plateau.
    picks: list[int] = []
    history: list[tuple[float, int]] = []
    for _ in range(size):
        scores = [
            scorer(_blend(dataset.task, stacked, picks + [i], classes), dataset.X, dataset.y)
            for i in range(len(usable))
        ]
        best = int(np.argmax(scores))
        picks.append(best)
        history.append((float(scores[best]), len(picks)))

    best_so_far, cut = max(history, key=lambda pair: (pair[0], -pair[1]))
    picks = picks[:cut]

    counts = {i: picks.count(i) for i in sorted(set(picks))}
    total = float(sum(counts.values()))

    models, weights, members = [], [], []
    for index, count in counts.items():
        record = usable[index]
        learner = resolve(dataset.task, [record.learner])[0]
        model = clone(build_pipeline(dataset, learner, record.params))
        model.fit(dataset.X, dataset.y)
        models.append(model)
        weights.append(count / total)
        members.append((record.learner, record.params))

    log.info(
        "Ensemble: %d member(s) from %d candidate(s), out-of-fold %s %.4f.",
        len(models), len(usable), result.metric, best_so_far,
    )
    return EnsembleModel(
        task=dataset.task,
        models=models,
        weights=weights,
        members=members,
        classes_=classes,
    )
