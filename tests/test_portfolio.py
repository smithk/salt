"""Starting configurations, and the guarantee that they are real.

A portfolio entry is a dict of parameter names typed out by hand, away from the
learner that defines them. Optuna accepts a queued trial without complaint and
simply ignores names the objective never asks for, so a typo does not fail —
it quietly becomes an ordinary random trial, and the warm start does nothing.
These tests check every entry against the actual search spaces instead.
"""

import optuna
import pytest

from saltml.data import load
from saltml.learners import REGISTRY, resolve
from saltml.learners.base import Space
from saltml.portfolio import PORTFOLIO, entries, enqueue_portfolio
from saltml.search import search
from saltml.task import Task

optuna.logging.set_verbosity(optuna.logging.ERROR)

IRIS = "data/classification/iris.arff"


def _declared_space(task, name, samples=40):
    """Every parameter a learner can declare, with its distribution.

    Sampled repeatedly rather than once: conditional parameters — an SVM's
    ``gamma``, a forest's ``max_depth`` — are only declared on the branch that
    uses them, so one pass sees an arbitrary subset of the space.
    """
    learner = resolve(task, [name])[0]
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    declared = {}
    for _ in range(samples):
        trial = study.ask()
        learner.space(Space(trial, f"{name}__"))
        declared.update(trial.distributions)
    return declared


def _within(distribution, value):
    """Whether a value is admissible, without reaching into Optuna internals."""
    if isinstance(distribution, optuna.distributions.CategoricalDistribution):
        return value in distribution.choices
    return distribution.low <= value <= distribution.high


@pytest.mark.parametrize(
    "task,name",
    [(task, name) for task, block in PORTFOLIO.items() for name in block],
)
def test_portfolio_learners_exist(task, name):
    assert name in REGISTRY[task], f"{name} is not a {task} learner"


@pytest.mark.parametrize(
    "task,name",
    [(task, name) for task, block in PORTFOLIO.items() for name in block],
)
def test_portfolio_entries_match_the_declared_space(task, name):
    if name not in REGISTRY[task]:
        pytest.skip(f"{name} unavailable")
    declared = _declared_space(task, name)
    for config in PORTFOLIO[task][name]:
        for key, value in config.items():
            qualified = f"{name}__{key}"
            assert qualified in declared, (
                f"{name} never declares {qualified!r}; a queued trial would "
                f"silently ignore it and warm-starting would do nothing"
            )
            assert _within(declared[qualified], value), (
                f"{qualified}={value!r} is outside {declared[qualified]}"
            )


def test_entries_are_ordered_breadth_first():
    """A budget too small to finish the queue should still see every learner."""
    queued = entries(Task.CLASSIFICATION, ["random_forest", "svm", "knn"])
    first_round = queued[: len({e["learner"] for e in queued})]
    assert len({e["learner"] for e in first_round}) == len(first_round)


def test_unknown_learners_are_ignored():
    assert entries(Task.CLASSIFICATION, ["not_a_learner"]) == []


def test_enqueue_puts_configurations_on_the_study():
    study = optuna.create_study(direction="maximize")
    count = enqueue_portfolio(study, Task.CLASSIFICATION, ["random_forest", "svm"])
    assert count > 0
    assert len(study.get_trials(deepcopy=False)) == count


def test_warm_started_search_actually_runs_the_portfolio():
    result = search(
        load(IRIS), learners=["random_forest", "svm"], n_trials=8, folds=3,
        seed=0, pruner=None, warm_start=True,
    )
    tried = {(r.learner, r.params.get("n_estimators"), r.params.get("C")) for r in result.records}
    # The first portfolio entries are a 500-tree forest and an RBF SVM at C=1.
    assert any(learner == "random_forest" and trees == 500 for learner, trees, _ in tried)
    assert any(learner == "svm" and c == 1.0 for learner, _, c in tried)


def test_warm_start_reaches_the_time_budget_path():
    """The default budget is wall-clock, which runs a survey phase per learner
    rather than the single study `--trials` uses. Warm-starting has to happen
    there too, or it does nothing in the mode most runs actually take."""
    result = search(
        load(IRIS), learners=["random_forest"], timeout=8.0, folds=3,
        seed=0, pruner=None, warm_start=True,
    )
    surveyed = [r for r in result.records if r.phase == "survey"]
    assert any(r.params.get("n_estimators") == 500 for r in surveyed)


def test_warm_start_can_be_turned_off():
    result = search(
        load(IRIS), learners=["random_forest"], n_trials=4, folds=3,
        seed=0, pruner=None, warm_start=False,
    )
    assert all(r.params.get("n_estimators") != 500 for r in result.records)
