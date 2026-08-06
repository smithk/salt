# SALT — Suggest A Learner for Tabular data

Point it at a table. It searches across algorithms and their hyperparameters
and hands back a fitted model, a ranked table of what worked, and what the
accuracy costs to serve.

```bash
saltml fit data.csv --target label --time 10m -o model.joblib
```

```python
import saltml

result = saltml.fit("data.csv", target="label", timeout=600)
print(result.summary())
print(result.search.tradeoff())        # accuracy vs prediction cost
result.model.predict(new_rows)
```

## What it does

Classification and regression on tabular data. Every candidate is a full
pipeline — imputation, categorical encoding, optional scaling, then the
estimator — so mixed CSV files work without preparation.

**The budget is wall-clock time, not a trial count.** Learner costs span three
orders of magnitude: a decision tree fits in hundredths of a second, a TabPFN
forward pass takes tens. Counting trials treats those as equal units, which
makes runtime unpredictable and quietly starves the expensive families. A time
budget is spent in two phases:

1. **Survey** (30%) — every applicable learner gets an equal slice of
   wall-clock. Guarantees each family is tried at least once, and measures what
   each one costs.
2. **Focus** (70%) — [Optuna](https://optuna.org) TPE over the families still in
   contention, which are those within the top quarter of the observed score
   spread.

`--trials N` still works and skips the survey. That is the reproducible mode,
where equal trial counts are the point.

The budget is a target, not a ceiling: a trial already running cannot be
interrupted, so one slow learner can overshoot. Runs report what they actually
spent against what was asked for.

**A holdout fraction (25% by default) is withheld before the search starts** and
scored exactly once at the end. Cross-validation scores are optimistic because
they are what the search optimised against; the holdout number is the honest
one, and both are reported.

## Learners

13 for each task. All are searched by default; restrict with `--learners`.

| | |
|---|---|
| **Boosted trees** | `lightgbm`, `xgboost`, `catboost`, `hist_gradient_boosting` |
| **Bagged trees** | `random_forest`, `extra_trees`, `decision_tree` |
| **Linear** | `logistic_regression`, `ridge_classifier` (classification); `ridge`, `lasso`, `elastic_net` (regression) |
| **Kernel & instance** | `svm` / `svr`, `knn` |
| **Other** | `gaussian_nb` (classification), `tabpfn` (both) |

`saltml learners` lists them with their preprocessing requirements. The three
dedicated boosting libraries and TabPFN are optional installs; a missing one is
an absent learner, not an import error.

## Accuracy is not the only axis

The best model and the model you should deploy are often different. SALT
measures fit cost and prediction cost separately for every trial and reports
the frontier — the configurations beaten on neither accuracy nor speed:

```
Accuracy vs prediction cost (nothing here is beaten on both):
    learner     r2  predict_ms/1k  fit_ms
        svr 0.5137          29.99     6.0
      lasso 0.5066          23.22     4.4
elastic_net 0.5064          22.80     4.4
```

On that dataset `tabpfn` scored 0.5101 — within 0.8% of the winner — at
**12,000 ms/1k, some 400× the serving cost**. A single-winner answer cannot
express that. `result.search.recommended()` returns the cheapest model within
1% of the best, which is usually the honest answer: differences below a percent
rarely exceed the noise across folds, while a 400× difference in serving cost
is real.

Prediction cost is measured per 1,000 rows and includes fixed per-call
overhead, so on small datasets the absolute figure has a floor of a few tens of
milliseconds. It is meant for comparing learners on one dataset, not as a
deployment latency guarantee.

## Install

Requires Python 3.10+.

```bash
pip install -e .              # core
pip install -e '.[boost]'     # adds LightGBM / XGBoost / CatBoost
pip install -e '.[tabpfn]'    # adds TabPFN
pip install -e '.[dev]'       # pytest
```

`lightgbm` and `xgboost` need an OpenMP runtime (`libgomp1` on Debian/Ubuntu).

## Usage

```
saltml fit DATA [--target COL] [--task classification|regression]
                [--categorical COLS] [--time 10m | --trials N]
                [--metric M] [--learners a,b] [--folds K] [--holdout F]
                [--sampler tpe|random|hypercube] [--jobs N]
                [--top N] [-o model.joblib] [-q]

saltml learners [--task ...]
saltml bench list | fetch SUITE | run SUITE
```

The target defaults to the last column. Task type is detected from the target
and can be forced with `--task`. Defaults: `balanced_accuracy` for
classification, `r2` for regression, a 60 second budget, 5 folds.

Reads CSV, TSV, ARFF, and Parquet. Prefer Parquet: it keeps column types, so
categorical columns survive a round trip that CSV flattens.

### Integer-coded categories

A column of site IDs — 1, 2, 3 — is indistinguishable from a measurement once
written to a file. Treated as a number, it tells the model that site 3 is three
times site 1, and the result is quietly wrong rather than obviously broken:

```bash
saltml fit sites.parquet --learners ridge                       # r2 = -0.003
saltml fit sites.parquet --learners ridge --categorical site    # r2 =  0.998
```

SALT warns when a numeric column holds few distinct whole numbers and names the
flag that fixes it. Binary 0/1 columns are not flagged — they are already the
encoding a category would receive.

## Benchmarks

Suites are fetched from [OpenML](https://www.openml.org) on demand into a cache
outside the repository. Committing a published suite would put hundreds of
megabytes into git history permanently, and history cannot be shrunk afterwards.

| Suite | Contents |
|---|---|
| `smoke` | 5 small datasets, both tasks. Seconds to fetch, minutes to run. |
| `cc18-lite` | 12 classification tasks from OpenML-CC18 |
| `ctr23-lite` | 10 regression tasks from OpenML-CTR23 |

Each is a curated sample rather than the full published suite — CC18 is 72
datasets and several gigabytes, which is a benchmarking session, not a check.
The samples favour the shapes that break things: mixed types, all-categorical,
high cardinality, missing values.

### Worked example: every learner across a whole suite

```bash
# 1. See what is available, and where the cache will go
saltml bench list

# 2. Download one. Seconds, and nothing lands in the repository.
saltml bench fetch cc18-lite

# 3. Run every learner across all 12 datasets, 20 seconds each
saltml bench run cc18-lite --time 20s --folds 3 --jobs 8 -o results.csv
```

Every dataset is fetched, task-detected, preprocessed, searched across all 13
learners, and scored on a holdout it never saw:

```
                         dataset           task    n  features            metric     cv  holdout           best_learner  trials
                        credit-g classification 1000        20 balanced_accuracy 0.7092   0.6810                    svm     296
                        diabetes classification  768         8 balanced_accuracy 0.7396   0.7910          decision_tree     817
                     tic-tac-toe classification  958         9 balanced_accuracy 0.9880   0.9940 hist_gradient_boosting     433
                         vehicle classification  846        18 balanced_accuracy 0.8365   0.8695                 tabpfn     585
                        kr-vs-kp classification 3196        36 balanced_accuracy 0.9937   0.9988               lightgbm     198
                            sick classification 3772        29 balanced_accuracy 0.9640   0.9377          decision_tree     181
                        spambase classification 4601        57 balanced_accuracy 0.9483   0.9619               lightgbm      58
                         phoneme classification 5404         5 balanced_accuracy 0.8678   0.8981            extra_trees     263
         banknote-authentication classification 1372         4 balanced_accuracy 1.0000   1.0000                    svm     217
blood-transfusion-service-center classification  748         4 balanced_accuracy 0.7002   0.6678          random_forest     289
climate-model-simulation-crashes classification  540        20 balanced_accuracy 0.8107   0.8120                    svm     225
                            ilpd classification  583        10 balanced_accuracy 0.7068   0.7072                    svm     366
```

Seven different learners win across twelve datasets, which is the argument for
searching rather than defaulting to one favourite. Note also that a 20-second
budget buys 817 trials on `diabetes` and 58 on `spambase` — the same wall-clock
against very different per-trial costs, which is exactly what a fixed trial
count cannot express.

Each run also prints where the budget went:

```
Focus: logistic_regression, ridge_classifier, extra_trees, decision_tree, svm, gaussian_nb
       (dropped random_forest, hist_gradient_boosting, knn, lightgbm, xgboost, catboost,
        tabpfn after the survey).
```

`results.csv` carries the same columns for further analysis. Add `--learners`
to compare a subset, or `--sampler random` to check how much the TPE sampler is
actually buying — the same harness answers both.

To fetch everything at once: `saltml bench fetch all`. The cache lives at
`~/.cache/saltml/benchmarks` (override with `SALTML_CACHE`) and stores Parquet,
so column types survive the round trip.

## TabPFN

TabPFN is a transformer pre-trained on synthetic tabular data. It is not
trained on your data at all: the training rows are given to the network as
context and it predicts in a single forward pass. On small tables it is often
the strongest thing in the registry.

Two things make it unlike the other learners:

- **It has almost no hyperparameters**, so the search spends few trials on it.
- **It has hard size limits** from pre-training — 10,000 samples, 500 encoded
  features, 10 classes. Past those it is excluded before the search starts,
  with the reason printed, rather than failing every trial.

On CPU it is roughly 25–1000× slower per trial than the classical learners.

Version 2.x downloads its weights with no account. Version 8.x requires
registering at [ux.priorlabs.ai](https://ux.priorlabs.ai), accepting the
licence, and setting `TABPFN_TOKEN`. Both work — the pin defaults to 2.x so the
tool runs out of the box. Weights are fetched on first use, so the first TabPFN
run needs network access.

## Development

```bash
pytest                    # everything
pytest -m "not slow"      # skip network and model-weight tests
```

`data/` holds a small offline corpus the tests depend on — 47 classification
and 2 regression ARFF files, one canonical copy of each.

Malformed-input fixtures are not stored. `tests/test_malformed.py` generates
them: a file whose only purpose is to be broken is cheaper to write in three
lines than to carry in git forever, and generating it documents exactly what is
wrong with it.

## History

SALT began in 2013–2014 as a research prototype by Roger Bermudez-Chacon,
Kevin Smith, and Peter Horvath — a CASH solver (combined algorithm selection
and hyperparameter optimisation) contemporary with Auto-WEKA. It was written
before deep learning reshaped the field, in Python 2, against scikit-learn
0.14.

The current version keeps the ideas worth keeping — the two-stage evaluation
protocol, and searching algorithms and hyperparameters jointly — and replaces
the hand-built process pool, cluster dispatch, Tkinter interface, and
conditional-parameter-space machinery with Optuna and modern scikit-learn.
11,600 lines of Python became about 2,500, plus 1,100 lines of tests where
there were none that ran.

The original **shrinking-hypercube optimiser is not yet ported**; `--sampler
hypercube` says so rather than pretending. Restoring it, and benchmarking it
against TPE across the suites above, is the outstanding piece of work.

The 2014 implementation is preserved at the `v0.1-2014` tag and on the `master`
branch.
