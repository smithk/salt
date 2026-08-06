"""Budget allocation and the accuracy/cost tradeoff.

A fixed trial count treats a decision tree and a TabPFN forward pass as equal
units of budget, which they are not by three orders of magnitude. With a time
budget the search surveys every learner on an equal slice of wall-clock, then
spends the rest on the families still in contention.
"""

import numpy as np
import optuna
import pandas as pd
import pytest

import saltml
from saltml.data import load
from saltml.search import (
    MIN_SURVEY_SECONDS,
    SURVEY_FRACTION,
    TrialRecord,
    SearchResult,
    _contenders,
    search,
)
from saltml.task import Task

optuna.logging.set_verbosity(optuna.logging.WARNING)

IRIS = "data/classification/iris.arff"


def record(learner, score, predict_ms=1.0, phase="focus"):
    return TrialRecord(
        number=0, learner=learner, params={}, score=score, std=0.0,
        seconds=0.1, fit_seconds=0.01, predict_ms_per_1k=predict_ms, phase=phase,
    )


def result_from(records):
    return SearchResult(
        dataset="t", task=Task.CLASSIFICATION, metric="balanced_accuracy",
        records=records, study=None,
    )


# --- contender selection -------------------------------------------------

def test_contenders_keep_the_leaders_and_drop_the_hopeless():
    # Needs more than three candidates: the floor below keeps everything when
    # there are only three, on the grounds that there is nothing to save.
    records = [
        record("good", 0.90), record("also_good", 0.89), record("fine", 0.87),
        record("poor", 0.40), record("awful", 0.20),
    ]
    kept = _contenders(records, ["good", "also_good", "fine", "poor", "awful"])
    assert "good" in kept and "also_good" in kept
    assert "awful" not in kept
    assert "poor" not in kept


def test_contenders_do_not_filter_a_field_of_three():
    records = [record("a", 0.9), record("b", 0.5), record("c", 0.1)]
    assert len(_contenders(records, ["a", "b", "c"])) == 3


def test_contenders_keep_everything_when_scores_are_tied():
    records = [record(n, 0.5) for n in ("a", "b", "c", "d")]
    assert set(_contenders(records, ["a", "b", "c", "d"])) == {"a", "b", "c", "d"}


def test_contenders_never_narrow_below_three():
    # One clear leader must not collapse the search onto a single family.
    records = [record("win", 0.99)] + [record(n, 0.10) for n in ("a", "b", "c")]
    assert len(_contenders(records, ["win", "a", "b", "c"])) >= 3


def test_contenders_survive_negative_scores():
    # r2 is unbounded below; the threshold has to come from the spread.
    records = [record("a", -0.1), record("b", -5.0), record("c", -50.0)]
    kept = _contenders(records, ["a", "b", "c"])
    assert "a" in kept


def test_contenders_fall_back_to_all_when_nothing_scored():
    assert _contenders([], ["a", "b"]) == ["a", "b"]


# --- the two phases actually run -----------------------------------------

@pytest.mark.slow
def test_time_budget_surveys_every_learner():
    """Every applicable learner must get at least one trial, unlike a trial budget."""
    result = search(load(IRIS), timeout=25, folds=3, seed=0)
    surveyed = {r.learner for r in result.records if r.phase == "survey"}
    registered = {learner.name for learner in saltml.for_task(Task.CLASSIFICATION)}
    missed = registered - surveyed - set(result.excluded) - set(result.never_worked)
    assert not missed, f"survey skipped {missed}"


@pytest.mark.slow
def test_time_budget_is_respected():
    import time

    started = time.perf_counter()
    search(load(IRIS), timeout=20, folds=3, seed=0)
    elapsed = time.perf_counter() - started
    # Optuna only checks the clock between trials, so allow slop for one
    # long trial overshooting; the point is that it is bounded at all.
    assert elapsed < 20 * 3, f"took {elapsed:.0f}s for a 20s budget"


def test_trial_budget_skips_the_survey():
    result = search(load(IRIS), learners=["decision_tree"], n_trials=4, folds=3, seed=0)
    assert all(r.phase == "focus" for r in result.records)
    assert len(result.records) == 4


def test_default_budget_is_time_not_trials():
    """An unbudgeted call must be bounded in wall-clock, not in trial count."""
    import inspect

    source = inspect.getsource(search)
    assert "timeout = 60.0" in source


# --- cost is measured ----------------------------------------------------

def test_cost_is_recorded_per_trial():
    result = search(load(IRIS), learners=["decision_tree"], n_trials=3, folds=3, seed=0)
    for entry in result.records:
        assert entry.fit_seconds > 0
        assert entry.predict_ms_per_1k > 0


def test_expensive_learner_is_measured_as_expensive():
    """A forest must cost more to predict than a single tree."""
    result = search(
        load(IRIS), learners=["decision_tree", "random_forest"],
        n_trials=8, folds=3, seed=0,
    )
    by_learner = result.best_by_learner()
    if len(by_learner) == 2:
        assert (
            by_learner["random_forest"].predict_ms_per_1k
            > by_learner["decision_tree"].predict_ms_per_1k
        )


# --- the tradeoff report -------------------------------------------------

def test_frontier_excludes_dominated_options():
    # 'bloated' is both worse and slower than 'lean': it cannot be on the front.
    records = [
        record("lean", 0.90, predict_ms=1.0),
        record("bloated", 0.80, predict_ms=50.0),
        record("heavy", 0.95, predict_ms=100.0),
    ]
    names = [r.learner for r in result_from(records).frontier()]
    assert "bloated" not in names
    assert "heavy" in names and "lean" in names


def test_frontier_is_ordered_best_first():
    records = [
        record("a", 0.95, predict_ms=100.0),
        record("b", 0.90, predict_ms=1.0),
    ]
    front = result_from(records).frontier()
    assert [r.learner for r in front] == ["a", "b"]


def test_recommended_prefers_cheap_when_accuracy_is_a_wash():
    records = [
        record("slow", 0.900, predict_ms=500.0),
        record("fast", 0.895, predict_ms=1.0),   # within 1%
    ]
    assert result_from(records).recommended(tolerance=0.01).learner == "fast"


def test_recommended_keeps_accuracy_when_the_gap_is_real():
    records = [
        record("accurate", 0.95, predict_ms=500.0),
        record("fast", 0.60, predict_ms=1.0),    # far outside 1%
    ]
    assert result_from(records).recommended(tolerance=0.01).learner == "accurate"


def test_recommended_handles_a_single_learner():
    assert result_from([record("only", 0.9)]).recommended().learner == "only"


def test_tradeoff_table_has_the_columns_a_user_needs():
    table = result_from([record("a", 0.9, 2.0), record("b", 0.8, 1.0)]).tradeoff()
    assert list(table.columns) == ["learner", "balanced_accuracy", "predict_ms/1k", "fit_ms"]


def test_best_per_learner_reports_spend():
    result = search(load(IRIS), learners=["decision_tree"], n_trials=3, folds=3, seed=0)
    table = result.best_per_learner()
    assert "spent_s" in table.columns
    assert "predict_ms/1k" in table.columns
