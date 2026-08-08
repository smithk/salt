"""SALT — suggest a machine learning model and hyperparameters for your dataset.

    >>> import salt
    >>> result = salt.fit("data.csv", target="label", timeout=60)
    >>> print(result.search.leaderboard())
    >>> result.model.predict(new_rows)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import Dataset, load
from .learners import REGISTRY, Learner, for_task, resolve
from .metrics import DEFAULT_METRIC, describe_metric, resolve_metric
from .runner import LocalRunner, Runner
from .search import SearchResult, TrialRecord, build_pipeline, search
from .task import Task

__version__ = "0.2.0.dev0"

__all__ = [
    "Dataset",
    "FitResult",
    "Learner",
    "SearchResult",
    "Task",
    "TrialRecord",
    "fit",
    "for_task",
    "load",
    "search",
]

log = logging.getLogger("saltml")


@dataclass
class FitResult:
    """The chosen model, how it was chosen, and how it did on unseen data."""

    model: Pipeline
    learner: str
    params: dict[str, Any]
    search: SearchResult
    dataset: Dataset = field(repr=False)
    holdout_score: float | None = None

    @property
    def metric(self) -> str:
        return self.search.metric

    @property
    def cv_score(self) -> float:
        return self.search.best.score

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def save(self, path: str | Path) -> Path:
        import joblib

        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, destination)
        return destination

    def summary(self) -> str:
        lines = [
            self.dataset.describe(),
            f"metric: {describe_metric(self.metric)}",
            f"best:   {self.learner}  cv={self.cv_score:.4f}",
        ]
        if self.holdout_score is not None:
            lines.append(f"holdout: {self.holdout_score:.4f}  (untouched during search)")
        budget = self.search.budget_seconds
        if budget is not None:
            over = "" if self.search.elapsed_seconds <= budget * 1.1 else "  (over: a trial in flight cannot be cut short)"
            # Pruned trials are absent from `records`, so reporting only those
            # would understate the work by however much pruning saved.
            pruned = (
                f" (+{self.search.n_pruned} stopped early)" if self.search.n_pruned else ""
            )
            lines.append(
                f"spent:  {self.search.elapsed_seconds:.0f}s of a {budget:.0f}s budget"
                f" across {len(self.search.records)} trials{pruned}{over}"
            )
        members = getattr(self.model, "members", None)
        if members is not None:
            # An ensemble was fitted, so the single best above describes the
            # strongest member rather than what will actually do the predicting.
            names = ", ".join(sorted({name for name, _ in members}))
            lines.append(f"model:  ensemble of {len(members)} ({names})")
        else:
            lines.append(f"params: {self.params}")
        return "\n".join(lines)


def fit(
    source: str | Path | pd.DataFrame,
    target: str | int | None = None,
    *,
    task: Task | str | None = None,
    categorical: Sequence[str] | None = None,
    learners: Sequence[str] | None = None,
    metric: str | None = None,
    n_trials: int | None = None,
    timeout: float | None = None,
    folds: int = 5,
    holdout: float = 0.25,
    sampler: str = "tpe",
    pruner: str | None = "median",
    warm_start: bool = True,
    ensemble: bool = False,
    n_jobs: int = -1,
    seed: int | None = 0,
    runner: Runner | None = None,
    progress: Any = None,
) -> FitResult:
    """Search for the best model for a dataset and fit it.

    A holdout fraction is set aside before the search begins and scored only
    once at the end. Cross-validation scores are optimistic precisely because
    they were optimised against; the holdout number is the honest one.

    :param source: dataset path or DataFrame.
    :param target: target column; defaults to the last one.
    :param categorical: feature columns to treat as labels rather than
        quantities, for integer-coded categories a file cannot describe.
    :param holdout: fraction withheld from the search. Set to 0 to search on
        everything and forgo an independent estimate.
    """
    dataset = load(source, target, task=task, categorical=categorical)
    log.info("Loaded %s", dataset.describe())

    train, held_out = _split_holdout(dataset, holdout, seed)

    result = search(
        train,
        learners=learners,
        metric=metric,
        n_trials=n_trials,
        timeout=timeout,
        folds=folds,
        sampler=sampler,
        pruner=pruner,
        warm_start=warm_start,
        runner=runner or LocalRunner(n_jobs=n_jobs),
        seed=seed,
        progress=progress,
    )

    best = result.best
    learner = resolve(dataset.task, [best.learner])[0]
    model = build_pipeline(train, learner, best.params)
    model.fit(train.X, train.y)

    if ensemble:
        from .ensemble import build_ensemble

        # Membership is decided on out-of-fold predictions, never on the
        # holdout: choosing against the holdout would spend the one honest
        # estimate the tool has.
        combined = build_ensemble(train, result, folds=folds, seed=seed, n_jobs=n_jobs)
        if combined is not None:
            model = combined

    holdout_score = None
    if held_out is not None:
        from sklearn.metrics import get_scorer

        holdout_score = float(get_scorer(result.metric)(model, held_out.X, held_out.y))

    return FitResult(
        model=model,
        learner=best.learner,
        params=best.params,
        search=result,
        dataset=dataset,
        holdout_score=holdout_score,
    )


def _split_holdout(
    dataset: Dataset, fraction: float, seed: int | None
) -> tuple[Dataset, Dataset | None]:
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"holdout must be in [0, 1), got {fraction}")
    if fraction == 0.0:
        return dataset, None

    stratify = dataset.y if dataset.task is Task.CLASSIFICATION else None
    if stratify is not None and int(dataset.y.value_counts().min()) < 2:
        log.warning("A class has a single sample; splitting without stratification.")
        stratify = None

    X_train, X_hold, y_train, y_hold = train_test_split(
        dataset.X, dataset.y, test_size=fraction, random_state=seed, stratify=stratify
    )
    make = lambda X, y: Dataset(  # noqa: E731 - trivial local constructor
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        task=dataset.task,
        name=dataset.name,
        forced_categorical=list(dataset.forced_categorical),
    )
    return make(X_train, y_train), make(X_hold, y_hold)
