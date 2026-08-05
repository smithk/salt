# SALT — Suggest-A-Learner Toolbox

Point it at a dataset. It searches over algorithms and their hyperparameters,
and hands back a fitted model plus a ranked table of what worked.

```bash
salt fit data.csv --target label --time 10m -o model.joblib
```

```python
import salt

result = salt.fit("data.csv", target="label", timeout=600)
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
```

`lightgbm` and `xgboost` need an OpenMP runtime (`libgomp1` on Debian/Ubuntu).

## Usage

```
salt fit DATA [--target COL] [--categorical COLS] [--time 10m | --trials N]
              [--metric M] [--learners a,b] [--folds K] [--holdout F]
              [--sampler tpe|random|hypercube] [--jobs N] [-o model.joblib]

salt learners [--task classification|regression]
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
salt fit sites.parquet --learners ridge              # r2 = -0.003
salt fit sites.parquet --learners ridge \
    --categorical site                               # r2 =  0.998
```

SALT warns when a numeric column holds few distinct whole numbers and names the
flag that fixes it. Binary 0/1 columns are not flagged — they are already the
encoding a category would receive.

## Datasets

`data/standard_ml_sets/` holds ~40 ARFF benchmark datasets, used as both a test
corpus and the bed for comparing search strategies.

## History

SALT began in 2013–2014 as a research prototype by Roger Bermudez-Chacon,
Kevin Smith, and Peter Horvath — a CASH solver (combined algorithm selection
and hyperparameter optimisation) contemporary with Auto-WEKA. The current
version keeps that two-stage evaluation protocol and its shrinking-hypercube
optimiser, and replaces the hand-built process pool, cluster dispatch, and
parameter-space machinery with Optuna and modern scikit-learn.

The original implementation is preserved in git history on the `master` branch.
