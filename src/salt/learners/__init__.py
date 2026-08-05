"""Registry of searchable algorithms."""

from __future__ import annotations

from ..task import Task
from .base import Learner, Space
from .classification import CLASSIFIERS
from .regression import REGRESSORS

__all__ = ["Learner", "Space", "REGISTRY", "for_task", "resolve"]

REGISTRY: dict[Task, dict[str, Learner]] = {
    Task.CLASSIFICATION: {learner.name: learner for learner in CLASSIFIERS},
    Task.REGRESSION: {learner.name: learner for learner in REGRESSORS},
}


def for_task(task: Task) -> list[Learner]:
    """Every learner applicable to ``task``."""
    return list(REGISTRY[task].values())


def resolve(task: Task, names: list[str] | None) -> list[Learner]:
    """Select learners by name, defaulting to all of them for the task."""
    available = REGISTRY[task]
    if not names:
        return list(available.values())

    unknown = [n for n in names if n not in available]
    if unknown:
        raise KeyError(
            f"Unknown learner(s) for {task}: {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(available))}"
        )
    return [available[n] for n in names]
