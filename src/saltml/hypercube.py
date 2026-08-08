"""The shrinking-hypercube optimiser, ported from the 2014 version of SALT.

The idea, and the only genuinely non-textbook one in the original codebase:
keep a box around the best configuration seen so far and sample inside it.
When a draw improves on the incumbent, the box grows (1.2x) and re-centres on
the winner — the neighbourhood is paying off, so look further. When a draw
fails to improve, the box shrinks (0.97x) around the unchanged incumbent —
tighten until something turns up. Growth is fast and decay is slow, so a run
of failures is needed to undo one success.

Crucially there is not one box but one per *categorical signature*. Continuous
hyperparameters mean different things under different discrete choices — an
SVM's ``C`` behaves differently under an RBF kernel than a linear one — so the
boxes are kept apart and each converges on its own.

One correction is not a port detail but a fix. The original centred a box on
the *first* completed trial under a signature — a single random draw — and then
confined sampling to 5% of each range from the second trial onward, shrinking
further on every draw that failed to beat it. With no exploration at all, a
mediocre first draw was polished for the rest of the budget, and the search
performed *worse than uniform random sampling* on 10 of 11 OpenML-CC18 tasks.
``STARTUP_TRIALS`` draws from the prior first and builds the box around the
best of them; that alone beat the faithful port on 10 of 12 measured cells and
closed most of the gap to TPE. Pass ``n_startup_trials=0`` for the 2014
behaviour.

Delegating the *categorical* choices to TPE — letting it concentrate on
learners that are working — looks like the obvious next win and measured worse,
both alone and on top of the warm-up. The likely reason: TPE commits to a
learner on the evidence of early trials whose continuous configurations are
still poor, and cannot back out. It is available as ``categorical="tpe"`` and
is off by default.

Two further deviations from the 2014 code, both deliberate:

* The collapse threshold is relative (a fraction of the prior range) rather
  than the original's absolute ``1e-4``. Absolute made collapse depend on the
  units a hyperparameter happened to be measured in: a box on ``[0, 1]`` was
  declared collapsed a hundred times sooner than the same proportional box on
  ``[0, 100]``. That is a bug, not part of the idea being tested.
* The first result for a signature establishes the incumbent without also
  expanding. The original compared against ``None`` and, under Python 2's
  permissive ordering, always took the improvement branch.

The port targets Optuna's sampler interface, which differs from the original
in one way that matters. The 2014 optimiser drew a whole configuration at once
and so always knew the complete categorical signature. Optuna asks for one
parameter at a time, so when a float is drawn the categoricals after it in the
sampling order do not exist yet. Each box is therefore keyed by the categorical
*context* present at the moment its parameter was drawn — reconstructable
exactly, because Optuna's ``distributions`` preserves sampling order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers import BaseSampler, RandomSampler
from optuna.trial import FrozenTrial, TrialState

__all__ = ["ShrinkingHypercubeSampler"]

#: Draws from the prior, per signature, before that signature's box is used.
#: Not in the 2014 design, and the single largest correction to it.
STARTUP_TRIALS = 20
#: Initial box edge, as a fraction of the prior range.
STARTING_WIDTH = 0.05
#: Growth on improvement. Deliberately larger than the decay is small.
EXPAND_RATE = 1.2
#: Decay on failure to improve.
SHRINK_RATE = 0.97
#: A box narrower than this fraction of the prior range has converged; the
#: signature is reset to the prior so the search can escape a local optimum.
COLLAPSE_FRACTION = 1e-4

#: Context key: the categorical choices fixed before a parameter was drawn.
Context = tuple[tuple[str, Any], ...]


@dataclass
class _Cube:
    """One box, covering every numeric parameter drawn under one context."""

    #: Completed trials seen under this context, for the warm-up count.
    trials: int = 0
    #: Current edge length per parameter, in sampling space.
    widths: dict[str, float] = field(default_factory=dict)
    #: The incumbent's coordinates, in sampling space.
    centre: dict[str, float] = field(default_factory=dict)
    #: Prior range per parameter, for initial width and collapse detection.
    spans: dict[str, float] = field(default_factory=dict)
    #: Best objective value seen under this context.
    best: float | None = None

    def collapsed(self) -> bool:
        return bool(self.widths) and all(
            width <= COLLAPSE_FRACTION * self.spans[name] for name, width in self.widths.items()
        )


def _bounds(distribution: BaseDistribution) -> tuple[float, float] | None:
    """Bounds in sampling space, or None if this is not a numeric parameter.

    Log-scaled parameters are handled in log space throughout, so that a box
    is proportionally sized rather than swamped by the top of the range.
    """
    if not isinstance(distribution, (FloatDistribution, IntDistribution)):
        return None
    if distribution.low <= 0 and distribution.log:
        return None  # cannot take logs; leave it to the prior
    if getattr(distribution, "step", None) not in (None, 1):
        return None  # a value off the grid would be silently snapped elsewhere
    if distribution.log:
        return math.log(distribution.low), math.log(distribution.high)
    return float(distribution.low), float(distribution.high)


def _context(trial: FrozenTrial, before: str | None = None) -> Context:
    """Categorical choices fixed before ``before`` was drawn.

    ``before=None`` yields the full signature. Order comes from Optuna, which
    records distributions as they are requested.
    """
    context: list[tuple[str, Any]] = []
    for name, distribution in trial.distributions.items():
        if name == before:
            break
        if isinstance(distribution, CategoricalDistribution) and name in trial.params:
            context.append((name, trial.params[name]))
    return tuple(context)


class ShrinkingHypercubeSampler(BaseSampler):
    """Sampler implementing the shrinking-hypercube search.

    Falls back to uniform sampling from the prior whenever there is no box to
    draw from — before a signature has produced its first result, for
    categorical parameters, and after a box collapses and is reset.
    """

    def __init__(
        self,
        seed: int | None = None,
        *,
        n_startup_trials: int = STARTUP_TRIALS,
        starting_width: float = STARTING_WIDTH,
        categorical: str = "random",
    ) -> None:
        """
        ``n_startup_trials`` draws from the prior before a signature's box is
        used, so the box forms around the best of a sample rather than around
        whichever draw happened to land first. Pass ``0`` for the 2014
        behaviour, which is measurably worse — see the module docstring.

        ``categorical`` chooses who picks the discrete parts of the space,
        including which learner to try. Both the original and the default here
        leave it to the prior (``"random"``). ``"tpe"`` lets Optuna concentrate
        on learners that are working; it sounds like the obvious win and
        measured worse, also discussed in the module docstring.
        """
        if categorical not in ("random", "tpe"):
            raise ValueError(f"categorical must be 'random' or 'tpe', not {categorical!r}")
        self._random = RandomSampler(seed=seed)
        self._categorical = (
            optuna.samplers.TPESampler(seed=seed) if categorical == "tpe" else self._random
        )
        self._rng = np.random.default_rng(seed)
        self._startup = n_startup_trials
        self._starting_width = starting_width
        self._cubes: dict[Context, _Cube] = {}

    def reseed_rng(self) -> None:
        self._random.reseed_rng()
        if self._categorical is not self._random:
            self._categorical.reseed_rng()
        self._rng = np.random.default_rng()

    def infer_relative_search_space(
        self, study: optuna.Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        # Every parameter is drawn independently: a box is per-parameter and
        # needs the categorical context, which only exists parameter by
        # parameter.
        return {}

    def sample_relative(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if isinstance(param_distribution, CategoricalDistribution):
            return self._categorical.sample_independent(
                study, trial, param_name, param_distribution
            )
        bounds = _bounds(param_distribution)
        if bounds is None:
            return self._random.sample_independent(study, trial, param_name, param_distribution)

        cube = self._cubes.get(_context(trial, before=param_name))
        if cube is None or param_name not in cube.centre or cube.trials <= self._startup:
            # Still warming up: draw from the prior so the box forms around the
            # best of a sample rather than around the first thing that landed.
            return self._random.sample_independent(study, trial, param_name, param_distribution)

        low, high = bounds
        half = cube.widths[param_name] / 2.0
        centre = cube.centre[param_name]
        # Clamp to the prior. An incumbent near an edge gets a box truncated by
        # it rather than one that wanders outside the declared range.
        lower = max(low, centre - half)
        upper = min(high, centre + half)
        if not upper > lower:
            return self._random.sample_independent(study, trial, param_name, param_distribution)

        value = float(self._rng.uniform(lower, upper))
        if param_distribution.log:
            value = math.exp(value)
        if isinstance(param_distribution, IntDistribution):
            value = int(min(max(round(value), param_distribution.low), param_distribution.high))
        return value

    def after_trial(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
        state: TrialState,
        values: list[float] | None,
    ) -> None:
        if state != TrialState.COMPLETE or not values:
            return  # a failed configuration says nothing about its neighbourhood
        value = float(values[0])
        maximise = study.direction == optuna.study.StudyDirection.MAXIMIZE

        # Group this trial's numeric parameters by the context each was drawn
        # under, so every box is updated with exactly its own coordinates.
        by_context: dict[Context, dict[str, tuple[float, float]]] = {}
        for name, distribution in trial.distributions.items():
            bounds = _bounds(distribution)
            if bounds is None or name not in trial.params:
                continue
            raw = float(trial.params[name])
            coordinate = math.log(raw) if distribution.log and raw > 0 else raw
            span = bounds[1] - bounds[0]
            by_context.setdefault(_context(trial, before=name), {})[name] = (coordinate, span)

        for context, drawn in by_context.items():
            cube = self._cubes.get(context)
            if cube is None:
                cube = self._cubes[context] = _Cube()
            cube.trials += 1
            if cube.best is None:
                # First result under this signature: it is the incumbent, and
                # there is nothing yet to have improved on.
                cube.best = value
                cube.centre = {name: coordinate for name, (coordinate, _) in drawn.items()}
                cube.spans = {name: span for name, (_, span) in drawn.items()}
                cube.widths = {
                    name: self._starting_width * span for name, (_, span) in drawn.items()
                }
                continue

            improved = value > cube.best if maximise else value < cube.best
            if improved:
                cube.best = value
                cube.centre = {name: coordinate for name, (coordinate, _) in drawn.items()}
            # During warm-up the draws come from the prior, so how they compare
            # says nothing about whether this box is the right size. Track the
            # incumbent, but leave the width alone until the box is in use.
            if cube.trials > self._startup:
                # Shrinking is around the unchanged incumbent, not around the
                # draw that just failed.
                self._scale(cube, EXPAND_RATE if improved else SHRINK_RATE)

            if cube.collapsed():
                # Converged as far as it usefully can. Drop the box; sampling
                # reverts to the prior and a fresh box forms from what it finds.
                del self._cubes[context]

    @staticmethod
    def _scale(cube: _Cube, rate: float) -> None:
        for name in cube.widths:
            floor = COLLAPSE_FRACTION * cube.spans[name]
            # Already at the floor: leave it, so collapse is detected rather
            # than a box growing back off a value that never decayed.
            if cube.widths[name] > floor:
                cube.widths[name] = max(cube.widths[name] * rate, floor)
