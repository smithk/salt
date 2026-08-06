"""Registry of searchable algorithms."""

from __future__ import annotations

import logging

from ..task import Task
from .base import Learner, Space
from .classification import CLASSIFIERS
from .regression import REGRESSORS
from .tabpfn import TABPFN_LEARNERS, tabpfn_available

__all__ = ["Learner", "Space", "REGISTRY", "for_task", "resolve", "applicable"]

log = logging.getLogger("saltml")

REGISTRY: dict[Task, dict[str, Learner]] = {
    Task.CLASSIFICATION: {learner.name: learner for learner in CLASSIFIERS},
    Task.REGRESSION: {learner.name: learner for learner in REGRESSORS},
}

# Optional learners join the registry only when their dependency is present,
# so a missing extra shows up as one absent name rather than an import error.
if tabpfn_available():
    for _learner in TABPFN_LEARNERS:
        REGISTRY[_learner.task][_learner.name] = _learner


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


def applicable(
    learners: list[Learner], dataset, *, requested: bool
) -> tuple[list[Learner], dict[str, str]]:
    """Split learners into those this dataset supports and those it does not.

    :param requested: whether the caller named these learners explicitly. If
        so, exclusions are surfaced as warnings rather than quiet information,
        because the user asked for something they will not get.
    """
    usable: list[Learner] = []
    excluded: dict[str, str] = {}
    for learner in learners:
        reason = learner.excluded_for(dataset)
        if reason is None:
            usable.append(learner)
        else:
            excluded[learner.name] = reason

    for name, reason in excluded.items():
        log.log(
            logging.WARNING if requested else logging.INFO,
            "Skipping %s: %s.", name, reason,
        )
    return usable, excluded
