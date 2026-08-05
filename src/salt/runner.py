"""Where trials execute.

Only a local implementation exists. The seam is here so that a Ray or Dask
backend can be added later without restructuring the search: the engine asks a
runner to drive the study and to say how much fold-level parallelism it may
use, and knows nothing else about execution.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import optuna

__all__ = ["Runner", "LocalRunner"]


@runtime_checkable
class Runner(Protocol):
    #: Fold-level parallelism available to a single trial.
    n_jobs: int

    def optimize(
        self,
        study: optuna.Study,
        objective: Callable[[optuna.Trial], float],
        *,
        n_trials: int | None,
        timeout: float | None,
        callbacks: list[Any] | None = None,
    ) -> None: ...


class LocalRunner:
    """Run trials in this process, parallelising across cross-validation folds.

    Fold-level rather than trial-level parallelism: Optuna's ``n_jobs`` uses
    threads, which serialises on the GIL for the pure-Python parts of a fit,
    whereas scikit-learn's fold parallelism uses processes.
    """

    def __init__(self, n_jobs: int = -1) -> None:
        self.n_jobs = n_jobs

    def optimize(
        self,
        study: optuna.Study,
        objective: Callable[[optuna.Trial], float],
        *,
        n_trials: int | None,
        timeout: float | None,
        callbacks: list[Any] | None = None,
    ) -> None:
        study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=callbacks)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"LocalRunner(n_jobs={self.n_jobs})"
