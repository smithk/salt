"""The search itself: draw a learner and a configuration, cross-validate, repeat."""

from __future__ import annotations

import itertools
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
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
    #: Mean seconds to fit one fold. What the search costs.
    fit_seconds: float = 0.0
    #: Milliseconds to score 1,000 rows. What deployment costs — a different
    #: number from fit time, and sometimes the opposite one: TabPFN barely
    #: fits at all and then predicts slowly.
    predict_ms_per_1k: float = 0.0
    #: Which phase produced this trial, for reporting where the budget went.
    phase: str = "focus"


@dataclass
class SearchResult:
    dataset: str
    task: Task
    metric: str
    records: list[TrialRecord]
    study: optuna.Study = field(repr=False)
    n_failed: int = 0
    #: Trials stopped early by the pruner. Distinct from ``n_failed``: these
    #: were working, just not well enough to be worth finishing.
    n_pruned: int = 0
    #: Learners ruled out before the search, mapped to why.
    excluded: dict[str, str] = field(default_factory=dict)
    #: Learners that were tried but never once succeeded, mapped to the first error.
    never_worked: dict[str, str] = field(default_factory=dict)
    #: Budget asked for, and wall-clock actually spent. These differ: a trial
    #: already running cannot be interrupted, so the budget is a target rather
    #: than a ceiling, and one slow learner can overshoot it substantially.
    budget_seconds: float | None = None
    elapsed_seconds: float = 0.0

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

    def best_by_learner(self) -> dict[str, TrialRecord]:
        best: dict[str, TrialRecord] = {}
        for record in self.records:
            current = best.get(record.learner)
            if current is None or record.score > current.score:
                best[record.learner] = record
        return best

    def best_per_learner(self) -> pd.DataFrame:
        rows = sorted(self.best_by_learner().values(), key=lambda r: r.score, reverse=True)
        return pd.DataFrame(
            [
                {
                    "learner": r.learner,
                    self.metric: round(r.score, 4),
                    "std": round(r.std, 4),
                    "predict_ms/1k": round(r.predict_ms_per_1k, 2),
                    "trials": sum(1 for x in self.records if x.learner == r.learner),
                    "spent_s": round(
                        sum(x.seconds for x in self.records if x.learner == r.learner), 1
                    ),
                }
                for r in rows
            ]
        )

    def frontier(self) -> list[TrialRecord]:
        """Configurations not beaten on both accuracy and prediction cost.

        Answers the question a single winner cannot: what does the last
        percent of accuracy actually cost to serve? A record is on the
        frontier when nothing scores higher *and* predicts faster.
        """
        best = sorted(
            self.best_by_learner().values(),
            key=lambda r: (-r.score, r.predict_ms_per_1k),
        )
        frontier: list[TrialRecord] = []
        cheapest = float("inf")
        for record in best:
            if record.predict_ms_per_1k < cheapest:
                frontier.append(record)
                cheapest = record.predict_ms_per_1k
        return frontier

    def recommended(self, tolerance: float = 0.01) -> TrialRecord:
        """The cheapest model within ``tolerance`` of the best score.

        Usually the honest answer to "what should I use". Accuracy differences
        below a percent are rarely larger than the noise across folds, while a
        hundredfold difference in prediction cost is real.
        """
        cutoff = self.best.score - abs(self.best.score) * tolerance
        contenders = [r for r in self.best_by_learner().values() if r.score >= cutoff]
        return min(contenders, key=lambda r: (r.predict_ms_per_1k, -r.score))

    def tradeoff(self) -> pd.DataFrame:
        rows = self.frontier()
        return pd.DataFrame(
            [
                {
                    "learner": r.learner,
                    self.metric: round(r.score, 4),
                    "predict_ms/1k": round(r.predict_ms_per_1k, 2),
                    "fit_ms": round(r.fit_seconds * 1000, 1),
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
    if not encode and dataset.categorical_columns:
        # The learner encodes categoricals itself and must be told which
        # columns they are. Names survive the transformer, so pass names.
        #
        # Only when there are some: CatBoost silently drops cat_features=[]
        # in its constructor, which makes sklearn's clone() refuse the
        # estimator — and cross-validation clones for every fold, so the
        # learner fails everywhere while working fine in a direct fit().
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


#: Share of a time budget spent surveying every learner before focusing.
#: Low enough that most of the budget still goes on the contenders, high
#: enough that a slow-but-strong family gets a fair first look.
SURVEY_FRACTION = 0.3
#: No learner gets less than this in the survey, however many there are;
#: a slice too short to fit once tells us nothing.
MIN_SURVEY_SECONDS = 2.0


def _contenders(records: list[TrialRecord], names: list[str]) -> list[str]:
    """Which learners survive the survey.

    Keeps everything within the top quarter of the observed spread, rather
    than a fixed number or an absolute margin: score scales differ between
    metrics (balanced accuracy sits in [0, 1], r2 can be arbitrarily
    negative), so the threshold has to come from the data.
    """
    best: dict[str, float] = {}
    for record in records:
        best[record.learner] = max(best.get(record.learner, -np.inf), record.score)
    if not best:
        return list(names)

    scores = list(best.values())
    leader, worst = max(scores), min(scores)
    spread = leader - worst
    if spread <= 0:
        return list(best)

    cutoff = leader - 0.25 * spread
    survivors = [n for n, s in best.items() if s >= cutoff]
    # Always keep a few, so one lucky survey trial cannot narrow the search
    # to a single family.
    if len(survivors) < 3:
        survivors = [n for n, _ in sorted(best.items(), key=lambda kv: -kv[1])][:3]
    return survivors


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
    pruner: str | optuna.pruners.BasePruner | None = None,
    warm_start: bool = False,
    runner: Runner | None = None,
    seed: int | None = 0,
    progress: Callable[[int, TrialRecord | None], None] | None = None,
) -> SearchResult:
    """Search over learners and their hyperparameters.

    With a time budget the search runs in two phases. A *survey* gives every
    applicable learner an equal slice of wall-clock, which both scores it
    roughly and measures what it costs; then a *focus* phase spends the rest
    on the families still in contention.

    The alternative — one study with the algorithm as a top-level categorical
    and a fixed trial count — treats a decision tree (hundredths of a second)
    and a TabPFN forward pass (seconds) as equal units of budget. That makes
    wall-clock unpredictable and quietly starves the expensive families, which
    on tabular data are often the ones worth having.

    A trial budget still works and still uses the single-study form: it is the
    reproducible mode, where equal trial counts are the point.
    """
    if n_trials is None and timeout is None:
        timeout = 60.0

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

    failures: list[str] = []
    failures_by_learner: dict[str, str] = {}
    pruned_early: list[int] = []  # counted across phases, so not read off one study
    counter = itertools.count()

    def make_objective(
        offered: list[str], phase: str, *, prunable: bool = False
    ) -> Callable[[optuna.Trial], float]:
        def objective(trial: optuna.Trial) -> float:
            name = trial.suggest_categorical("learner", offered)
            learner = by_name[name]
            params = learner.sample(trial, seed=seed)
            pipeline = build_pipeline(dataset, learner, params)

            started = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    # Failed convergence is information the score already
                    # carries; emitting it per fold would bury the output.
                    warnings.simplefilter("ignore")
                    scored = _evaluate(
                        trial,
                        pipeline,
                        dataset,
                        metric_name,
                        splitter,
                        n_jobs=active_runner.n_jobs,
                        prunable=prunable,
                    )
            except optuna.TrialPruned:
                # Stopped early on purpose. Not a failure, and must not be
                # counted as one — the generic handler below would do exactly
                # that, since TrialPruned is an ordinary exception.
                pruned_early.append(1)
                raise
            except Exception as exc:  # a bad configuration must not end the search
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                failures_by_learner.setdefault(name, f"{type(exc).__name__}: {exc}")
                log.debug("Trial %s (%s) failed: %s", trial.number, name, exc)
                raise optuna.TrialPruned() from exc

            scores = scored["test_score"]
            # score_time covers predict plus the metric on one held-out fold.
            # Normalising by fold size makes it comparable across datasets and
            # gives a number that means something at deployment.
            rows_per_fold = max(1, len(dataset.y) // splitter.get_n_splits())
            predict_ms = float(np.mean(scored["score_time"])) * 1000.0 / rows_per_fold * 1000.0

            trial.set_user_attr("learner", name)
            trial.set_user_attr("params", params)
            trial.set_user_attr("std", float(np.std(scores)))
            trial.set_user_attr("seconds", time.perf_counter() - started)
            trial.set_user_attr("fit_seconds", float(np.mean(scored["fit_time"])))
            trial.set_user_attr("predict_ms_per_1k", predict_ms)
            trial.set_user_attr("phase", phase)
            trial.set_user_attr("serial", next(counter))
            return float(np.mean(scores))

        return objective

    def run_phase(
        offered: list[str],
        phase: str,
        *,
        phase_sampler: optuna.samplers.BaseSampler,
        phase_trials: int | None,
        phase_timeout: float | None,
        prunable: bool = False,
        warm: bool = False,
    ) -> tuple[list[TrialRecord], optuna.Study]:
        study = optuna.create_study(
            direction="maximize",
            sampler=phase_sampler,
            pruner=_make_pruner(pruner) if prunable else optuna.pruners.NopPruner(),
            study_name=f"saltml:{dataset.name}:{phase}",
        )
        if warm and warm_start:
            from .portfolio import enqueue_portfolio

            enqueue_portfolio(study, dataset.task, offered)

        def _on_trial(_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            if progress is not None:
                progress(trial.user_attrs.get("serial", trial.number), _to_record(trial))

        active_runner.optimize(
            study,
            make_objective(offered, phase, prunable=prunable),
            n_trials=phase_trials,
            timeout=phase_timeout,
            callbacks=[_on_trial],
        )
        found = [r for t in study.trials if (r := _to_record(t)) is not None]
        return found, study

    records: list[TrialRecord] = []
    search_started = time.perf_counter()
    if timeout is None:
        # Trial-count mode: one study, every learner on the table. Equal trial
        # counts are the point here, so no survey phase.
        records, study = run_phase(
            names, "focus",
            phase_sampler=_make_sampler(sampler, seed),
            phase_trials=n_trials,
            phase_timeout=None,
            prunable=True,
            warm=True,
        )
    else:
        survey_budget = timeout * SURVEY_FRACTION
        slice_each = max(MIN_SURVEY_SECONDS, survey_budget / max(1, len(names)))
        log.info(
            "Survey: %d learners, up to %.0fs each.", len(names), slice_each,
        )
        started = time.perf_counter()
        study = None
        for name in names:
            found, study = run_phase(
                [name], "survey",
                phase_sampler=optuna.samplers.RandomSampler(seed=seed),
                phase_trials=None,
                phase_timeout=slice_each,
                # The survey is where each learner is first seen, so it is
                # where a strong starting configuration is worth most. The
                # focus phase that follows is deliberately not warm-started:
                # it is a separate study, so it would re-run configurations
                # the survey has already measured.
                warm=True,
            )
            records.extend(found)

        remaining = max(0.0, timeout - (time.perf_counter() - started))
        contenders = _contenders(records, names)
        if len(contenders) < len(names):
            log.info(
                "Focus: %s (dropped %s after the survey).",
                ", ".join(contenders),
                ", ".join(n for n in names if n not in contenders) or "none",
            )
        if remaining > 1.0 and contenders:
            found, study = run_phase(
                contenders, "focus",
                phase_sampler=_make_sampler(sampler, seed),
                phase_trials=None,
                phase_timeout=remaining,
                # The survey deliberately does not prune: its fixed slice per
                # learner is what measures cost, and cutting it short would
                # decide the algorithm choice on partial evidence.
                prunable=True,
            )
            records.extend(found)

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
        n_pruned=len(pruned_early),
        excluded=excluded,
        never_worked=never_worked,
        budget_seconds=timeout,
        elapsed_seconds=time.perf_counter() - search_started,
    )


def _to_record(trial: optuna.trial.FrozenTrial) -> TrialRecord | None:
    if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
        return None
    return TrialRecord(
        number=trial.user_attrs.get("serial", trial.number),
        learner=trial.user_attrs.get("learner", "?"),
        params=trial.user_attrs.get("params", {}),
        score=float(trial.value),
        std=float(trial.user_attrs.get("std", 0.0)),
        seconds=float(trial.user_attrs.get("seconds", 0.0)),
        fit_seconds=float(trial.user_attrs.get("fit_seconds", 0.0)),
        predict_ms_per_1k=float(trial.user_attrs.get("predict_ms_per_1k", 0.0)),
        phase=trial.user_attrs.get("phase", "focus"),
    )


#: Folds evaluated between pruning decisions. Not one at a time: fold-level
#: parallelism is how a trial gets its speed, so a chunk is still evaluated in
#: parallel and the gap between chunks is where a hopeless configuration is
#: dropped. Two buys a decision after 40% of a 5-fold trial while keeping most
#: of the parallelism.
PRUNE_CHUNK = 2


def _evaluate(
    trial: optuna.Trial,
    pipeline: Pipeline,
    dataset: Dataset,
    metric_name: str,
    splitter: Any,
    *,
    n_jobs: int,
    prunable: bool,
) -> dict[str, np.ndarray]:
    """Cross-validate, optionally stopping early on a hopeless configuration.

    Without pruning this is one ``cross_validate`` call, exactly as before.
    With it, folds are evaluated in chunks and the running mean is reported to
    Optuna between them, which is what a pruner needs to compare this trial
    against the ones already finished.
    """
    splits = list(splitter.split(dataset.X, dataset.y))
    step = PRUNE_CHUNK if prunable else len(splits)
    scores: list[float] = []
    fit_times: list[float] = []
    score_times: list[float] = []

    for start in range(0, len(splits), step):
        part = cross_validate(
            pipeline,
            dataset.X,
            dataset.y,
            scoring=metric_name,
            cv=splits[start : start + step],
            n_jobs=n_jobs,
            error_score="raise",
        )
        scores.extend(part["test_score"])
        fit_times.extend(part["fit_time"])
        score_times.extend(part["score_time"])
        if prunable and start + step < len(splits):
            trial.report(float(np.mean(scores)), step=len(scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

    return {
        "test_score": np.asarray(scores),
        "fit_time": np.asarray(fit_times),
        "score_time": np.asarray(score_times),
    }


def _make_pruner(
    pruner: str | optuna.pruners.BasePruner | None,
) -> optuna.pruners.BasePruner:
    """Resolve a pruner. ``None`` means run every trial to completion."""
    if pruner is None or pruner == "none":
        return optuna.pruners.NopPruner()
    if isinstance(pruner, optuna.pruners.BasePruner):
        return pruner
    key = pruner.lower()
    if key == "median":
        # Warm-up matters more here than in a typical Optuna study: the first
        # trials of each learner are the only evidence that learner has, and
        # pruning them on the strength of a different learner's head start
        # would decide the algorithm choice before it has been measured.
        return optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    if key == "asha":
        return optuna.pruners.SuccessiveHalvingPruner()
    if key == "hyperband":
        return optuna.pruners.HyperbandPruner()
    raise ValueError(
        f"Unknown pruner {pruner!r}. Choose from: none, median, asha, hyperband."
    )


def _make_sampler(
    sampler: str | optuna.samplers.BaseSampler, seed: int | None
) -> optuna.samplers.BaseSampler:
    if isinstance(sampler, optuna.samplers.BaseSampler):
        return sampler
    key = sampler.lower()
    if key == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if key == "tpe-mv":
        # Models parameters jointly instead of one at a time, and groups them
        # by which trials actually defined them — which is what a space like
        # this one is, where each learner contributes its own parameters and
        # they are absent from every trial that chose a different learner.
        return optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    if key == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if key == "hypercube":
        from .hypercube import ShrinkingHypercubeSampler

        return ShrinkingHypercubeSampler(seed=seed)
    raise ValueError(
        f"Unknown sampler {sampler!r}. Choose from: tpe, tpe-mv, random, hypercube."
    )
