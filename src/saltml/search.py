"""The search itself: draw a learner and a configuration, cross-validate, repeat."""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .data import Dataset
from .learners import Learner, applicable, resolve
from .metrics import resolve_metric
from .preprocess import make_preprocessor
from .runner import LocalRunner, Runner
from .task import Task

__all__ = ["SearchResult", "TrialRecord", "search", "build_pipeline"]

log = logging.getLogger("saltml")


@dataclass
class TrialRecord:
    number: int
    learner: str
    params: dict[str, Any]
    score: float
    std: float
    seconds: float


@dataclass
class SearchResult:
    dataset: str
    task: Task
    metric: str
    records: list[TrialRecord]
    study: optuna.Study = field(repr=False)
    n_failed: int = 0
    #: Learners ruled out before the search, mapped to why.
    excluded: dict[str, str] = field(default_factory=dict)
    #: Learners that were tried but never once succeeded, mapped to the first error.
    never_worked: dict[str, str] = field(default_factory=dict)

    @property
    def best(self) -> TrialRecord:
        if not self.records:
            raise RuntimeError("No trial completed successfully.")
        return max(self.records, key=lambda r: r.score)

    def leaderboard(self, top: int | None = 15) -> pd.DataFrame:
        rows = sorted(self.records, key=lambda r: r.score, reverse=True)
        if top is not None:
            rows = rows[:top]
        return pd.DataFrame(
            [
                {
                    "rank": i + 1,
                    "learner": r.learner,
                    self.metric: round(r.score, 4),
                    "std": round(r.std, 4),
                    "seconds": round(r.seconds, 2),
                    "params": _short_params(r.params),
                }
                for i, r in enumerate(rows)
            ]
        )

    def best_per_learner(self) -> pd.DataFrame:
        best: dict[str, TrialRecord] = {}
        for record in self.records:
            current = best.get(record.learner)
            if current is None or record.score > current.score:
                best[record.learner] = record
        rows = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return pd.DataFrame(
            [
                {
                    "learner": r.learner,
                    self.metric: round(r.score, 4),
                    "std": round(r.std, 4),
                    "trials": sum(1 for x in self.records if x.learner == r.learner),
                }
                for r in rows
            ]
        )


def _short_params(params: dict[str, Any], limit: int = 60) -> str:
    text = ", ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()
    )
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_pipeline(dataset: Dataset, learner: Learner, params: dict[str, Any]) -> Pipeline:
    """Assemble the full candidate: preprocessing plus estimator."""
    encode = not learner.handles_categorical
    params = dict(params)
    if not encode:
        # The learner encodes categoricals itself and must be told which
        # columns they are. Names survive the transformer, so pass names.
        params["cat_features"] = list(dataset.categorical_columns)
    return Pipeline(
        [
            (
                "prepare",
                make_preprocessor(
                    dataset, scale=learner.needs_scaling, encode_categorical=encode
                ),
            ),
            ("model", learner.build(params)),
        ]
    )


def _splitter(dataset: Dataset, folds: int, seed: int | None):
    """Cross-validation splitter, reduced if a class is too small to spread."""
    if dataset.task is Task.REGRESSION:
        return KFold(n_splits=folds, shuffle=True, random_state=seed)

    smallest_class = int(dataset.y.value_counts().min())
    usable = max(2, min(folds, smallest_class))
    if usable < folds:
        log.warning(
            "Smallest class has %d samples; using %d folds instead of %d.",
            smallest_class, usable, folds,
        )
    return StratifiedKFold(n_splits=usable, shuffle=True, random_state=seed)


def search(
    dataset: Dataset,
    *,
    learners: Sequence[str] | None = None,
    metric: str | None = None,
    n_trials: int | None = None,
    timeout: float | None = None,
    folds: int = 5,
    sampler: str | optuna.samplers.BaseSampler = "tpe",
    runner: Runner | None = None,
    seed: int | None = 0,
    progress: Callable[[int, TrialRecord | None], None] | None = None,
) -> SearchResult:
    """Search over learners and their hyperparameters.

    All learners share a single study with the algorithm itself as a top-level
    categorical, so the sampler can concentrate the budget on whichever family
    is working rather than dividing it evenly in advance.
    """
    if n_trials is None and timeout is None:
        n_trials = 100

    requested = list(learners) if learners else None
    candidates, excluded = applicable(
        resolve(dataset.task, requested), dataset, requested=requested is not None
    )
    if not candidates:
        raise ValueError(
            "No learner can be used on this dataset. "
            + "; ".join(f"{name}: {reason}" for name, reason in excluded.items())
        )
    by_name = {learner.name: learner for learner in candidates}
    names = list(by_name)
    metric_name = resolve_metric(dataset.task, metric)
    active_runner = runner or LocalRunner()
    splitter = _splitter(dataset, folds, seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=_make_sampler(sampler, seed),
        study_name=f"saltml:{dataset.name}",
    )

    failures: list[str] = []
    failures_by_learner: dict[str, str] = {}

    def objective(trial: optuna.Trial) -> float:
        name = trial.suggest_categorical("learner", names)
        learner = by_name[name]
        params = learner.sample(trial, seed=seed)
        pipeline = build_pipeline(dataset, learner, params)

        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                # Failed convergence is information the score already carries;
                # emitting it per fold would bury the actual output.
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    pipeline,
                    dataset.X,
                    dataset.y,
                    scoring=metric_name,
                    cv=splitter,
                    n_jobs=active_runner.n_jobs,
                    error_score="raise",
                )
        except Exception as exc:  # a bad configuration must not end the search
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            failures_by_learner.setdefault(name, f"{type(exc).__name__}: {exc}")
            log.debug("Trial %d (%s) failed: %s", trial.number, name, exc)
            raise optuna.TrialPruned() from exc

        elapsed = time.perf_counter() - started
        trial.set_user_attr("learner", name)
        trial.set_user_attr("params", params)
        trial.set_user_attr("std", float(np.std(scores)))
        trial.set_user_attr("seconds", elapsed)
        return float(np.mean(scores))

    def _on_trial(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if progress is not None:
            progress(trial.number, _to_record(trial))

    active_runner.optimize(
        study, objective, n_trials=n_trials, timeout=timeout, callbacks=[_on_trial]
    )

    records = [
        record
        for trial in study.trials
        if (record := _to_record(trial)) is not None
    ]
    if failures:
        log.debug("%d trial(s) failed. First: %s", len(failures), failures[0])

    # A learner that failed every single trial is a systematic problem — a
    # missing model download, an incompatible dependency — not an unlucky
    # configuration. Say so rather than leaving it as a silent absence.
    succeeded = {record.learner for record in records}
    never_worked = {n: r for n, r in failures_by_learner.items() if n not in succeeded}
    for name, reason in never_worked.items():
        log.warning("%s failed on every attempt and produced no result. %s", name, reason)

    return SearchResult(
        dataset=dataset.name,
        task=dataset.task,
        metric=metric_name,
        records=records,
        study=study,
        n_failed=len(failures),
        excluded=excluded,
        never_worked=never_worked,
    )


def _to_record(trial: optuna.trial.FrozenTrial) -> TrialRecord | None:
    if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
        return None
    return TrialRecord(
        number=trial.number,
        learner=trial.user_attrs.get("learner", "?"),
        params=trial.user_attrs.get("params", {}),
        score=float(trial.value),
        std=float(trial.user_attrs.get("std", 0.0)),
        seconds=float(trial.user_attrs.get("seconds", 0.0)),
    )


def _make_sampler(
    sampler: str | optuna.samplers.BaseSampler, seed: int | None
) -> optuna.samplers.BaseSampler:
    if isinstance(sampler, optuna.samplers.BaseSampler):
        return sampler
    key = sampler.lower()
    if key == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if key == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if key == "hypercube":
        try:
            from .hypercube import ShrinkingHypercubeSampler
        except ImportError as exc:  # pragma: no cover - until the port lands
            raise ValueError(
                "The shrinking-hypercube sampler has not been ported yet. "
                "Use --sampler tpe or --sampler random."
            ) from exc
        return ShrinkingHypercubeSampler(seed=seed)
    raise ValueError(f"Unknown sampler {sampler!r}. Choose from: tpe, random, hypercube.")
