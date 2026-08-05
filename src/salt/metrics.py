"""Scoring defaults.

Every metric here is oriented so that larger is better, which lets the search
maximise unconditionally.
"""

from __future__ import annotations

from sklearn.metrics import get_scorer

from .task import Task

__all__ = ["DEFAULT_METRIC", "resolve_metric", "describe_metric"]

# balanced_accuracy rather than accuracy: it does not flatter a model that
# only ever predicts the majority class, which matters because the tool is
# meant to be pointed at arbitrary data.
DEFAULT_METRIC: dict[Task, str] = {
    Task.CLASSIFICATION: "balanced_accuracy",
    Task.REGRESSION: "r2",
}

_HINTS: dict[str, str] = {
    "balanced_accuracy": "chance level is 1/n_classes",
    "accuracy": "compare against the majority-class rate",
    "r2": "0.0 is no better than predicting the mean",
    "neg_mean_squared_error": "negated so larger is better",
    "neg_mean_absolute_error": "negated so larger is better",
}


def resolve_metric(task: Task, metric: str | None) -> str:
    name = metric or DEFAULT_METRIC[task]
    try:
        get_scorer(name)
    except (ValueError, KeyError) as exc:
        raise ValueError(f"Unknown metric {name!r} for {task}.") from exc
    return name


def describe_metric(name: str) -> str:
    hint = _HINTS.get(name)
    return f"{name} ({hint})" if hint else name
