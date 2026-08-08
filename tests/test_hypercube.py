"""Mechanics of the shrinking-hypercube sampler.

These drive the sampler through a synthetic study with scripted objective
values, so the box arithmetic is checked directly rather than inferred from
whether a search happened to do well.
"""

import math

import optuna
import pytest

from saltml.hypercube import (
    COLLAPSE_FRACTION,
    EXPAND_RATE,
    SHRINK_RATE,
    STARTING_WIDTH,
    STARTUP_TRIALS,
    ShrinkingHypercubeSampler,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run(values, *, seed=0, low=0.0, high=1.0, log=False, kinds=("a",), startup=0):
    """Drive a one-float, one-categorical study through scripted scores.

    Warm-up defaults to off here: these check the box arithmetic, which only
    happens once a box is in use. The warm-up itself is tested separately.
    """
    sampler = ShrinkingHypercubeSampler(seed=seed, n_startup_trials=startup)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        trial.suggest_categorical("kind", list(kinds))
        trial.suggest_float("x", low, high, log=log)
        return values[trial.number]

    study.optimize(objective, n_trials=len(values))
    return sampler, study


def test_first_result_sets_the_incumbent_without_expanding():
    sampler, _ = run([0.5])
    cube = sampler._cubes[(("kind", "a"),)]
    assert cube.best == 0.5
    assert cube.widths["x"] == pytest.approx(STARTING_WIDTH * 1.0)


def test_box_expands_on_improvement_and_shrinks_otherwise():
    # 0.5 establishes the box, 0.4 fails to improve it, 0.9 beats it.
    sampler, _ = run([0.5, 0.4, 0.9])
    cube = sampler._cubes[(("kind", "a"),)]
    assert cube.best == 0.9
    assert cube.widths["x"] == pytest.approx(STARTING_WIDTH * SHRINK_RATE * EXPAND_RATE)


def test_incumbent_survives_a_worse_draw():
    sampler, _ = run([0.9, 0.1])
    cube = sampler._cubes[(("kind", "a"),)]
    assert cube.best == 0.9  # shrinking happens around the winner, not the loser


def test_each_categorical_signature_gets_its_own_box():
    sampler, _ = run([0.5] * 12, kinds=("a", "b", "c"))
    signatures = set(sampler._cubes)
    assert len(signatures) > 1
    assert all(len(sig) == 1 and sig[0][0] == "kind" for sig in signatures)


def test_draws_stay_inside_the_prior():
    _, study = run([0.5, 0.6, 0.4, 0.7, 0.3, 0.8], low=-2.0, high=3.0)
    drawn = [t.params["x"] for t in study.trials]
    assert all(-2.0 <= x <= 3.0 for x in drawn)


def test_log_parameters_are_boxed_in_log_space():
    sampler, _ = run([0.5], low=1e-4, high=1e2, log=True)
    cube = sampler._cubes[(("kind", "a"),)]
    span = math.log(1e2) - math.log(1e-4)
    assert cube.spans["x"] == pytest.approx(span)
    assert cube.widths["x"] == pytest.approx(STARTING_WIDTH * span)


def test_a_collapsed_box_resets_instead_of_vanishing():
    # A constant score never improves, so the box shrinks every trial. Pure
    # decay would put it under the floor well inside this many trials; a width
    # above the floor at the end proves it reset and started over.
    n = int(math.log(COLLAPSE_FRACTION / STARTING_WIDTH) / math.log(SHRINK_RATE)) + 60
    sampler, _ = run([0.5] * n)
    cube = sampler._cubes[(("kind", "a"),)]
    assert cube.widths["x"] > COLLAPSE_FRACTION * 1.0


def test_failed_trials_do_not_move_the_box():
    sampler = ShrinkingHypercubeSampler(seed=0, n_startup_trials=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        trial.suggest_categorical("kind", ["a"])
        trial.suggest_float("x", 0.0, 1.0)
        if trial.number == 1:
            raise optuna.TrialPruned()
        return 0.5

    study.optimize(objective, n_trials=3)
    cube = sampler._cubes[(("kind", "a"),)]
    # Trials 0 and 2 counted; the pruned one in between said nothing.
    assert cube.widths["x"] == pytest.approx(STARTING_WIDTH * SHRINK_RATE)


def test_same_seed_gives_the_same_draws():
    _, first = run([0.5, 0.6, 0.4, 0.7], seed=7)
    _, second = run([0.5, 0.6, 0.4, 0.7], seed=7)
    assert [t.params for t in first.trials] == [t.params for t in second.trials]


def test_minimisation_direction_is_respected():
    sampler = ShrinkingHypercubeSampler(seed=0, n_startup_trials=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    values = [0.5, 0.4]  # lower is better, so 0.4 is an improvement

    def objective(trial):
        trial.suggest_categorical("kind", ["a"])
        trial.suggest_float("x", 0.0, 1.0)
        return values[trial.number]

    study.optimize(objective, n_trials=2)
    cube = sampler._cubes[(("kind", "a"),)]
    assert cube.best == 0.4
    assert cube.widths["x"] == pytest.approx(STARTING_WIDTH * EXPAND_RATE)


def test_warm_up_is_on_by_default():
    # The 2014 behaviour (no warm-up) measured worse than uniform random
    # sampling. The default must not quietly go back to it.
    assert STARTUP_TRIALS > 0
    assert ShrinkingHypercubeSampler()._startup == STARTUP_TRIALS


def test_categorical_delegation_is_off_by_default():
    sampler = ShrinkingHypercubeSampler(seed=0)
    assert sampler._categorical is sampler._random


def test_warm_up_leaves_the_box_at_its_starting_size():
    sampler = ShrinkingHypercubeSampler(seed=0, n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    values = [0.5, 0.4, 0.3, 0.2, 0.1]  # every draw after the first fails to improve

    def objective(trial):
        trial.suggest_categorical("kind", ["a"])
        trial.suggest_float("x", 0.0, 1.0)
        return values[trial.number]

    study.optimize(objective, n_trials=5)
    cube = sampler._cubes[(("kind", "a"),)]
    # Without warm-up these would have shrunk four times over.
    assert cube.widths["x"] == pytest.approx(STARTING_WIDTH * 1.0)
    assert cube.trials == 5


def test_warm_up_centres_the_box_on_the_best_draw_not_the_first():
    sampler = ShrinkingHypercubeSampler(seed=0, n_startup_trials=4)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    values = [0.1, 0.2, 0.9, 0.3]  # the winner is the third draw

    def objective(trial):
        trial.suggest_categorical("kind", ["a"])
        trial.suggest_float("x", 0.0, 1.0)
        return values[trial.number]

    study.optimize(objective, n_trials=4)
    cube = sampler._cubes[(("kind", "a"),)]
    assert cube.best == 0.9
    assert cube.centre["x"] == pytest.approx(study.trials[2].params["x"])


def test_starting_width_is_configurable():
    sampler = ShrinkingHypercubeSampler(seed=0, starting_width=0.4)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        trial.suggest_categorical("kind", ["a"])
        trial.suggest_float("x", 0.0, 1.0)
        return 0.5

    study.optimize(objective, n_trials=1)
    assert sampler._cubes[(("kind", "a"),)].widths["x"] == pytest.approx(0.4)


def test_categorical_choice_can_be_delegated_to_tpe():
    sampler = ShrinkingHypercubeSampler(seed=0, categorical="tpe")
    assert isinstance(sampler._categorical, optuna.samplers.TPESampler)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        kind = trial.suggest_categorical("kind", ["a", "b"])
        trial.suggest_float("x", 0.0, 1.0)
        return 1.0 if kind == "a" else 0.0

    study.optimize(objective, n_trials=40)
    chosen = [t.params["kind"] for t in study.trials[20:]]
    # TPE should have worked out that "a" is the paying choice.
    assert chosen.count("a") > chosen.count("b")


def test_unknown_categorical_mode_is_rejected():
    with pytest.raises(ValueError, match="categorical must be"):
        ShrinkingHypercubeSampler(categorical="magic")


def test_integer_parameters_stay_integral_and_in_range():
    sampler = ShrinkingHypercubeSampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        trial.suggest_categorical("kind", ["a"])
        return float(trial.suggest_int("n", 2, 9))

    study.optimize(objective, n_trials=20)
    drawn = [t.params["n"] for t in study.trials]
    assert all(isinstance(n, int) and 2 <= n <= 9 for n in drawn)
