# Contributing

## Setup

```bash
git clone https://github.com/smithk/salt.git
cd salt
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

On a machine with no GPU, install the CPU build of PyTorch first — see
[README Install](README.md#install).

## Tests

```bash
pytest                    # everything
pytest -m "not slow"      # skip network and model-weight tests (the usual run)
pytest tests/test_budget.py -k frontier
```

`slow` marks the tests that download OpenML datasets or TabPFN weights. Keep
them marked: a test suite that needs the network is a test suite people stop
running.

## Test data

`data/` holds a small offline corpus the tests depend on — 47 classification
and 2 regression ARFF files, one canonical copy of each. Benchmark suites are
*not* stored here; they are fetched on demand (see
[README Benchmarks](README.md#benchmarks)).

Malformed-input fixtures are not stored either. `tests/test_malformed.py`
generates them: a file whose only purpose is to be broken is cheaper to write
in three lines than to review and carry in git forever, and generating it
documents exactly what is wrong with it.

## Adding a learner

A learner is a small record, not a class hierarchy. In
`src/saltml/learners/classification.py` or `regression.py`:

```python
def _my_space(s: Space) -> dict[str, Any]:
    # Conditional parameters are ordinary control flow.
    kernel = s.cat("kernel", ["rbf", "linear"])
    params = {"kernel": kernel, "C": s.float("C", 1e-3, 1e3, log=True)}
    if kernel == "rbf":
        params["gamma"] = s.float("gamma", 1e-5, 1e1, log=True)
    return params


Learner("my_learner", Task.CLASSIFICATION, MyEstimator, _my_space,
        needs_scaling=True)
```

Then add it to the module's list. The registry picks it up, and
`tests/test_learners.py` will start drawing configurations from its space and
fitting them on binary and multiclass data — which is where you find out that
your solver cannot do multiclass, or that your library rejects string labels.

If the learner exists for both tasks, put the shared part of its space in
`src/saltml/learners/spaces.py` and add only what is task-specific in the task
module — `class_weight` for a classifier, `epsilon` for an SVR. A forest wants
the same number of trees whichever target it is fitted against, and these were
duplicated by hand once and quietly diverged. `test_shared_spaces_do_not_drift_apart`
fails if the two sides stop agreeing.

Flags on `Learner`:

| Flag | Use |
|---|---|
| `needs_scaling` | Distance- and margin-based methods. Trees are invariant and pay only the cost. |
| `seedable` | Whether the estimator accepts `random_state`. |
| `handles_categorical` | Skip one-hot encoding and pass the raw columns. Only worth it where the learner's own handling beats one-hot, as CatBoost's does. |
| `applies` | Return a reason string when a dataset is out of range, or `None`. Used for TabPFN's pre-training size limits, and for its much lower CPU-only limit. |

Optional dependencies belong behind an import guard so a missing library is an
absent learner rather than an import error — see `learners/boosting.py`.

If the learner has a configuration that is strong across most tabular data,
add it to `src/saltml/portfolio.py` so the search starts there instead of
finding its way. `tests/test_portfolio.py` checks every entry against the
learner's declared space: a queued trial with a misspelled parameter is
silently ignored by Optuna rather than rejected, so a typo would turn the warm
start into an ordinary random trial and nothing would look wrong.

## Conventions

- Parameter names are namespaced per learner (`svm__C`). Every learner shares
  one Optuna study, so unprefixed names collide across learners with different
  ranges.
- Comments explain *why*, not *what*. Several in this codebase record a
  library's misbehaviour — CatBoost's `cat_features` not surviving `clone()`,
  XGBoost rejecting string labels — and exist so nobody re-derives them.
- New behaviour needs a test that fails without it. Most bugs found during the
  rewrite were found by writing the test first.

## Design notes

See [docs/design.md](docs/design.md) for how the search allocates its budget,
why cost is measured the way it is, and the library-compatibility adapters.
