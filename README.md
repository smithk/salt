# SALT — Suggest A Learner for Tabular data

Point it at a dataset. It searches over algorithms and their hyperparameters,
and hands back a fitted model plus a ranked table of what worked.

```bash
saltml fit data.csv --target label --time 10m -o model.joblib
```

```python
import saltml

result = saltml.fit("data.csv", target="label", timeout=600)
print(result.summary())
print(result.search.leaderboard())
result.model.predict(new_rows)
```

## What it does

Classification and regression on tabular data. Every candidate is a full
pipeline — imputation, categorical encoding, optional scaling, then the
estimator — so mixed CSV files work without preparation.

The search is a single [Optuna](https://optuna.org) study with the algorithm
itself as a top-level choice, which lets the sampler spend the budget on
whichever family is working instead of dividing it evenly up front.

A holdout fraction (25% by default) is withheld before the search starts and
scored exactly once at the end. Cross-validation scores are optimistic because
they are what the search optimised against; the holdout number is the honest
one, and both are reported.

## Install

Requires Python 3.10+.

```bash
pip install -e .              # core
pip install -e '.[boost]'     # adds LightGBM / XGBoost / CatBoost
pip install -e '.[tabpfn]'    # adds TabPFN
```

`lightgbm` and `xgboost` need an OpenMP runtime (`libgomp1` on Debian/Ubuntu).

### TabPFN

TabPFN is a transformer pre-trained on synthetic tabular data. It is not
trained on your data at all: the training rows are given to the network as
context and it predicts in a single forward pass. On small tables it is often
the strongest thing in the registry.

Two things make it unlike the other learners:

- **It has almost no hyperparameters**, so the search spends few trials on it.
- **It has hard size limits** from pre-training — 10,000 samples, 500 encoded
  features, 10 classes. Past those it is excluded before the search starts,
  with the reason printed, rather than failing every trial.

On CPU it is roughly 25–1000× slower per trial than the classical learners, so
prefer `--time` over `--trials` when it is enabled; a trial budget divides
evenly between learners that do not cost the same.

Version 2.x downloads its weights with no account. Version 8.x requires
registering at [ux.priorlabs.ai](https://ux.priorlabs.ai), accepting the
licence, and setting `TABPFN_TOKEN`. Both work — the pin defaults to 2.x so
the tool runs out of the box. Weights are fetched on first use, so the first
TabPFN run needs network access.

## Usage

```
saltml fit DATA [--target COL] [--categorical COLS] [--time 10m | --trials N]
                [--metric M] [--learners a,b] [--folds K] [--holdout F]
                [--sampler tpe|random|hypercube] [--jobs N] [-o model.joblib]

saltml learners [--task classification|regression]
```

The target defaults to the last column. Task type is detected from the target
and can be forced with `--task`. Defaults: `balanced_accuracy` for
classification, `r2` for regression.

Reads CSV, TSV, ARFF, and Parquet. Parquet is the format to prefer: it keeps
column types, so categorical columns survive a round trip that CSV flattens.

### Integer-coded categories

A column of site IDs — 1, 2, 3 — is indistinguishable from a measurement once
written to a file. Treated as a number, it tells the model that site 3 is three
times site 1, and the result is quietly wrong rather than obviously broken:

```bash
saltml fit sites.parquet --learners ridge                 # r2 = -0.003
saltml fit sites.parquet --learners ridge --categorical site   # r2 = 0.998
```

SALT warns when a numeric column holds few distinct whole numbers and names the
flag that fixes it. Binary 0/1 columns are not flagged — they are already the
encoding a category would receive.

## Datasets

`data/` holds a small offline corpus the test suite depends on — 47
classification and 2 regression ARFF files, one canonical copy of each.

Malformed-input fixtures are not stored here. `tests/test_malformed.py`
generates them: a file whose only purpose is to be broken is cheaper to write
in three lines than to carry in git forever, and generating it documents
exactly what is wrong with it.

For measuring how good SALT actually is, use the benchmark suites. They are
fetched from [OpenML](https://www.openml.org) on demand into a cache outside
the repository, because committing a published suite would put hundreds of
megabytes into git history permanently:

```bash
saltml bench list                        # suites, contents, cache location
saltml bench fetch ctr23-lite            # download (seconds)
saltml bench run smoke --trials 20       # score SALT across a whole suite
saltml bench run cc18-lite --time 2m -o results.csv
```

| Suite | Contents |
|---|---|
| `smoke` | 5 small datasets, both tasks. Seconds to fetch, minutes to run. |
| `cc18-lite` | 12 classification tasks from OpenML-CC18 |
| `ctr23-lite` | 10 regression tasks from OpenML-CTR23 |

Each is a curated sample rather than the full published suite — CC18 is 72
datasets and several gigabytes, which is a benchmarking session, not a check.
The samples favour the shapes that break things: mixed types, all-categorical,
high cardinality, missing values.

The cache lives at `~/.cache/saltml/benchmarks` (override with `SALTML_CACHE`)
and stores Parquet, so column types survive the round trip.

## History

SALT began in 2013–2014 as a research prototype by Roger Bermudez-Chacon,
Kevin Smith, and Peter Horvath — a CASH solver (combined algorithm selection
and hyperparameter optimisation) contemporary with Auto-WEKA. The current
version keeps that two-stage evaluation protocol and its shrinking-hypercube
optimiser, and replaces the hand-built process pool, cluster dispatch, and
parameter-space machinery with Optuna and modern scikit-learn.

The original implementation is preserved in git history on the `master` branch.
