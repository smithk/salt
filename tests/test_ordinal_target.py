"""Telling the user when the task was a coin flip.

SALT already warns when a *feature* looks like an integer-coded category. The
target has the same ambiguity and matters more — it decides the metric, the
learners and the fold policy — but was decided silently.

A rating scale is the case: 1-5 stars, 0-20 grades. Classification keeps the
levels distinct and scores every mistake alike; regression uses the ordering,
so being one level out counts as a smaller error than being five out. Both are
defensible and they do not score the same, which is exactly when the user
should be told rather than have a heuristic decide for them.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from saltml.data import load, looks_ordinal
from saltml.task import Task

RNG = np.random.default_rng(0)


def _frame(target):
    frame = pd.DataFrame(RNG.normal(size=(200, 3)), columns=list("abc"))
    frame["target"] = target
    return frame


def _warnings(caplog):
    return [r.message for r in caplog.records if r.levelno >= logging.WARNING]


def _load_capturing(caplog, target, **kwargs):
    with caplog.at_level(logging.WARNING, logger="saltml"):
        dataset = load(_frame(target), "target", **kwargs)
    return dataset, " ".join(_warnings(caplog))


def test_rating_scale_is_flagged(caplog):
    dataset, said = _load_capturing(caplog, RNG.integers(1, 6, 200))
    assert dataset.task is Task.CLASSIFICATION
    assert "rating scale" in said
    assert "--task regression" in said  # names the alternative, not just the choice


def test_a_scale_past_the_label_limit_is_flagged_the_other_way(caplog):
    """25 consecutive levels falls out of the heuristic as regression, and is
    just as arguably classification."""
    dataset, said = _load_capturing(caplog, RNG.integers(0, 25, 200))
    assert dataset.task is Task.REGRESSION
    assert "--task classification" in said


@pytest.mark.parametrize(
    "target,why",
    [
        (RNG.integers(0, 2, 200), "binary is not a scale"),
        (RNG.integers(0, 3, 200), "three labels carry too little ordering"),
        (RNG.normal(size=200), "continuous is unambiguous"),
        (RNG.choice([10, 20, 30, 40, 50], 200), "gapped codes are labels, not a scale"),
        (RNG.choice(list("abcde"), 200), "text is unambiguous"),
    ],
)
def test_unambiguous_targets_stay_quiet(caplog, target, why):
    _, said = _load_capturing(caplog, target)
    assert "rating scale" not in said, why


def test_an_explicit_task_silences_it(caplog):
    """The warning exists to prompt a decision. Once made, repeating it is noise."""
    _, said = _load_capturing(caplog, RNG.integers(1, 6, 200), task="regression")
    assert "rating scale" not in said


def test_warn_suspicious_false_silences_it(caplog):
    _, said = _load_capturing(caplog, RNG.integers(1, 6, 200), warn_suspicious=False)
    assert "rating scale" not in said


@pytest.mark.parametrize(
    "values,expected",
    [
        ([1, 2, 3, 4, 5], True),
        ([0, 1], False),
        ([1, 2, 3, 4], False),          # below the level floor
        ([1, 2, 3, 5, 6, 7], False),    # a gap breaks the scale
        ([1.5, 2.5, 3.5, 4.5, 5.5], False),
    ],
)
def test_looks_ordinal_directly(values, expected):
    series = pd.Series(values * 20)
    assert looks_ordinal(series) is expected
