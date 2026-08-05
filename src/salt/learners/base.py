"""How a candidate algorithm and its search space are described.

The 2014 version spent ~150 lines of class boilerplate per algorithm. Here a
learner is a small record: how to construct the estimator, and a function that
draws one configuration from an Optuna trial. Conditional hyperparameters —
where one choice unlocks others — are ordinary Python control flow, which is
what the old ``param.py`` parameter tree existed to express.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import optuna

from ..task import Task

__all__ = ["Learner", "Space"]


class Space:
    """Namespaced view of an Optuna trial.

    Every learner shares one study, so parameter names must not collide: an
    ``SVC``'s ``C`` and a ``LogisticRegression``'s ``C`` have different ranges
    and Optuna would reject the second definition. Prefixing by learner keeps
    them independent.
    """

    def __init__(self, trial: optuna.Trial, prefix: str) -> None:
        self._trial = trial
        self._prefix = prefix

    def float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        return self._trial.suggest_float(self._prefix + name, low, high, log=log)

    def int(self, name: str, low: int, high: int, *, log: bool = False) -> int:
        return self._trial.suggest_int(self._prefix + name, low, high, log=log)

    def cat(self, name: str, choices: Sequence[Any]) -> Any:
        return self._trial.suggest_categorical(self._prefix + name, list(choices))


@dataclass(frozen=True)
class Learner:
    """One searchable algorithm."""

    name: str
    task: Task
    factory: Callable[..., Any]
    space: Callable[[Space], dict[str, Any]]
    #: Distance- and margin-based methods need standardised features; trees do not.
    needs_scaling: bool = False
    #: Whether the estimator accepts ``random_state``.
    seedable: bool = True

    def sample(self, trial: optuna.Trial, *, seed: int | None = None) -> dict[str, Any]:
        params = self.space(Space(trial, f"{self.name}__"))
        if self.seedable and seed is not None:
            params["random_state"] = seed
        return params

    def build(self, params: dict[str, Any]) -> Any:
        return self.factory(**params)
