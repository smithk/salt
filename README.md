# SALT — Suggest A Learner for Tabular data

<img src="docs/logo.png" alt="" width="240" align="right">

Point it at a table. It searches across algorithms and their hyperparameters
and hands back a fitted model, a ranked table of what worked, and what the
accuracy costs to serve.

Classification and regression, 14 learners each — from linear models to
gradient-boosted trees to a pre-trained transformer.

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/smithk/salt.git
cd salt
python -m venv .venv && source .venv/bin/activate
pip install -e .
saltml --version
```

On a machine with **no GPU**, install the CPU build of PyTorch first — the
default resolution pulls several gigabytes of CUDA wheels you cannot use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

LightGBM, XGBoost and CatBoost need an OpenMP runtime — `libgomp1` on
Debian/Ubuntu, `sudo apt install libgomp1`. See
[Troubleshooting](#troubleshooting) if a learner is missing.

## Quickstart: run a benchmark

Nothing to prepare and no data of your own needed. Three commands:

```bash
saltml bench list                 # what's available, and where the cache goes
saltml bench fetch smoke          # 5 datasets, a few seconds
saltml bench run smoke --time 15s --folds 3
```

```
    dataset           task   n  features            metric     cv  holdout        best_learner  trials
       iris classification 150         4 balanced_accuracy 0.9729   0.9722                 svm     893
       wine classification 178        13 balanced_accuracy 0.9804   1.0000 logistic_regression     954
   breast-w classification 699         9 balanced_accuracy 0.9799   0.9783         extra_trees     922
  autoPrice     regression 159        15                r2 0.8603   0.9126              tabpfn    1092
cholesterol     regression 303        13                r2 0.0573   0.0546         elastic_net     637
```

That takes **about two and a half minutes** on 16 cores. Add `-o results.csv`
to keep the table.

Every dataset is fetched, task-detected, preprocessed, searched across all 13
learners, and scored on a holdout it never saw. See [Benchmarks](#benchmarks) for the larger suites.

## Your own data

```bash
saltml fit data.csv --target label --time 10m -o model.joblib
```

```python
import saltml

result = saltml.fit("data.csv", target="label", timeout=600)
print(result.summary())
print(result.search.tradeoff())          # accuracy vs prediction cost
print(result.search.leaderboard())       # every configuration tried
```

Using the saved model later:

```python
import joblib, pandas as pd

model = joblib.load("model.joblib")      # a fitted scikit-learn Pipeline
model.predict(pd.read_csv("new_rows.csv"))
```

The target defaults to the last column. Task type is detected from the target
and can be forced with `--task`. Reads CSV, TSV, ARFF and Parquet.

## Benchmarks

Suites are fetched from [OpenML](https://www.openml.org) on demand into a cache
outside the repository, so the datasets stay out of git.

| Suite | Contents | Approximate run time |
|---|---|---|
| `smoke` | 5 small datasets, both tasks | ~2.5 min at `--time 15s` |
| `cc18-lite` | 12 classification tasks from OpenML-CC18 | ~25 min at `--time 20s` |
| `ctr23-lite` | 10 regression tasks from OpenML-CTR23 | similar |

Run times are much longer than `--time` × datasets suggests: the budget is per
dataset, and it overshoots, because a trial already running cannot be
interrupted. See [Budget allocation](docs/design.md#budget-allocation).

Each suite is a curated sample rather than the full published one — CC18 is 72
datasets and several gigabytes. The samples favour the shapes that break
things: mixed types, all-categorical, high cardinality, missing values.

```bash
saltml bench fetch all                                   # every suite
saltml bench run cc18-lite --time 20s --folds 3 --jobs 8 -o results.csv
saltml bench run ctr23-lite --learners lightgbm,ridge    # compare a subset
saltml bench run smoke --sampler random                  # what is TPE buying?
```

A full `cc18-lite` run: 12 classification datasets against every learner,
scored by `balanced_accuracy`. The `task` and `metric` columns the command
prints are dropped here, since they are the same on every row.

```
                         dataset    n  features     cv holdout           best_learner  trials
                        credit-g 1000        20 0.7092  0.6810                    svm     264
                        diabetes  768         8 0.7396  0.7910          decision_tree     832
                     tic-tac-toe  958         9 0.9880  0.9940 hist_gradient_boosting     441
                         vehicle  846        18 0.8365  0.8695                 tabpfn     590
                        kr-vs-kp 3196        36 0.9937  0.9988               lightgbm     197
                            sick 3772        29 0.9640  0.9377          decision_tree     199
                        spambase 4601        57 0.9483  0.9619               lightgbm      91
                         phoneme 5404         5 0.8703  0.9019               catboost     430
         banknote-authentication 1372         4 1.0000  1.0000                    svm     788
blood-transfusion-service-center  748         4 0.7084  0.6643          random_forest     889
climate-model-simulation-crashes  540        20 0.8107  0.8120                    svm     708
                            ilpd  583        10 0.7068  0.7072                    svm     622
```

Seven different learners win across twelve datasets, which is the argument for
searching rather than defaulting to one favourite. Note also that a 20-second
budget buys 832 trials on `diabetes` and 91 on `spambase` — the same wall-clock
against very different per-trial costs.

The cache lives at `~/.cache/saltml/benchmarks` (override with `SALTML_CACHE`)
and stores Parquet, so column types survive the round trip.

## Learners

| Family | Members |
|---|---|
| **Boosted trees** | `lightgbm`, `xgboost`, `catboost`, `hist_gradient_boosting` |
| **Bagged trees** | `random_forest`, `extra_trees`, `decision_tree` |
| **Neural** | `tabpfn` — a pre-trained transformer |
| **Linear** | `logistic_regression`, `ridge_classifier`; `ridge`, `lasso`, `elastic_net` |
| **Kernel & instance** | `svm` / `svr`, `knn` |
| **Probabilistic** | `gaussian_nb` |

All install and are searched by default; restrict with `--learners`. Run
`saltml learners` to list them. See [docs/learners.md](docs/learners.md) for
per-learner notes, including TabPFN's size limits and weight downloads.

## How the search works

**The budget is wall-clock time, not a trial count**, because learner costs
span three orders of magnitude and counting trials treats them as equal. Time
is spent in two phases: a **survey** giving every learner an equal slice, then
a **focus** phase running Optuna TPE over the families still in contention.
`--trials N` skips the survey and is the reproducible mode.

**A holdout fraction (25% by default) is withheld before the search starts**
and scored exactly once at the end. Cross-validation scores are optimistic
because they are what the search optimised against; the holdout number is the
honest one, and both are reported.

**Hopeless trials are stopped early.** Folds are evaluated in chunks, and a
configuration already losing after the first chunk is abandoned rather than
carried to the last fold. Because folds are still evaluated in parallel within
a chunk, this costs none of the parallelism it would seem to. Turn it off with
`--pruner none`.

**`warm_start=True` seeds the search from configurations already known to be
strong** — a few-hundred-tree forest, boosting at a small learning rate —
instead of from the prior. Off by default, and honestly so: across two budgets
and 24 measured runs it never beat starting cold, and at a short budget it was
slightly worse. See [docs/design.md](docs/design.md) for the numbers.

**`--ensemble` combines the best trials instead of keeping only the winner.**
Members are chosen greedily on out-of-fold predictions, never on the holdout,
and a model that adds nothing is simply not selected — on some datasets the
result is a single model, which is the honest answer.

Full rationale, including the contention rule and why the budget overshoots:
[docs/design.md](docs/design.md).

### Which sampler

`--sampler tpe` is the default and the one to use. `tpe-mv` models parameters
jointly rather than independently; `random` is the control that tells you what
the sampler is buying.

`hypercube` is a shrinking-hypercube optimiser — one box per categorical
signature, expanding on improvement and shrinking otherwise. It is offered
because it is unusual rather than because it wins: across `cc18-lite` and
`ctr23-lite` it ranks last of three. In its original 1:1 form it lost to
*uniform random sampling* on 10 of 11 classification tasks, because it built
its box around the first completed trial — one random draw — and then confined
sampling to 5% of each range with no exploration phase at all. Drawing 20
trials from the prior first and building the box around the best of them beats
that on 10 of 12 measured cells and closes most of the gap to TPE without
overtaking it. That warm-up is on by default; `n_startup_trials=0` restores the
original behaviour.

One variant that sounds obviously right and measured worse: letting TPE choose
the learner while the box tunes its continuous parameters. TPE commits to a
learner on the evidence of early trials whose configurations are still poor,
and cannot back out. Available as `categorical="tpe"`, off by default.

Mechanism and measurements in
[docs/design.md](docs/design.md#the-shrinking-hypercube).

## Accuracy vs prediction cost

The best model and the model you should deploy are often different. SALT
measures fit cost and prediction cost separately for every trial and reports
the frontier — configurations beaten on neither accuracy nor speed:

| learner | r2 | predict_ms/1k | on the frontier |
|---|---|---|---|
| `svr` | 0.5137 | 29.99 | yes — most accurate |
| `lasso` | 0.5066 | 23.22 | yes |
| `elastic_net` | 0.5064 | 22.80 | yes — cheapest |
| `tabpfn` | 0.5101 | ~12,000 | **no** — less accurate than `svr` *and* ~400× dearer to serve |

The command prints only the frontier rows (`svr`, `lasso`, `elastic_net`) plus
a `fit_ms` column; `tabpfn` is shown here because being beaten on both counts
is the interesting part. It lands within 0.8% of the winner on accuracy, which
a leaderboard alone would make look like a close second, and costs some 400×
as much to serve. A single-winner answer cannot express that.
`result.search.recommended()` returns the cheapest model within 1% of the best,
which is usually the honest answer.

Prediction cost is per 1,000 rows and includes fixed per-call overhead, so on
small datasets it has a floor of a few tens of milliseconds. It compares
learners on one dataset; it is not a latency guarantee.

## Data formats

Reads CSV, TSV, ARFF and Parquet. **Prefer Parquet where you have the choice.**
Parquet stores each column's type alongside the data, so a column of category
labels is still a category column when it is read back. CSV stores only text,
so those types have to be guessed on load — and a category coded as `1`, `2`,
`3` is indistinguishable from a measurement, which is how a categorical column
ends up silently treated as a quantity. See
[Integer-coded categories](#integer-coded-categories) for the CSV workaround.

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

## Command reference

```
saltml fit DATA [--target COL] [--task classification|regression]
                [--categorical COLS] [--time 10m | --trials N]
                [--metric M] [--learners a,b] [--folds K] [--holdout F]
                [--sampler tpe|tpe-mv|random|hypercube]
                [--pruner median|asha|hyperband|none] [--ensemble] [--jobs N]
                [--top N] [-o model.joblib] [-q]

saltml learners [--task ...]

saltml bench list
saltml bench fetch SUITE|all [--refresh]
saltml bench run SUITE [--task ...] [--time D | --trials N] [--learners a,b]
                       [--folds K] [--sampler S] [--pruner P] [--jobs N]
                       [-o results.csv]
```

Defaults: `balanced_accuracy` for classification, `r2` for regression, a
60 second budget, 5 folds, 25% holdout, all cores, `--sampler tpe`,
and `--pruner median`. Warm-starting and ensembling are opt-in.

## Troubleshooting

**A learner is missing from `saltml learners`.** Its library failed to import.
For `lightgbm`, `xgboost` or `catboost` this is almost always a missing OpenMP
runtime: `sudo apt install libgomp1`. Check with
`python -c "import lightgbm"`.

**`pip install` pulled gigabytes of NVIDIA packages.** TabPFN brings PyTorch,
which resolves to the CUDA build by default. Install the CPU build first (see
[Install](#install)).

**TabPFN asks for a licence token.** You have version 8.x, which requires
registering at [ux.priorlabs.ai](https://ux.priorlabs.ai) and setting
`TABPFN_TOKEN`. The pinned 2.x line needs no account —
`pip install 'tabpfn>=2.2,<3'`.

**A run took far longer than `--time`.** Expected. The budget is per dataset,
and a trial already running cannot be interrupted, so one slow learner
overshoots. Runs print elapsed against requested.

**Scores look impossibly good.** Check for a leaked identifier column, and
compare the cross-validation score against the holdout: a large gap between
them is the signal.

## Citing

If you use SALT in published work, please cite the repository. The
2013–2014 prototype is preserved at the `v0.1-2014` tag.

```bibtex
@software{salt,
  title  = {SALT: Suggest A Learner for Tabular data},
  author = {Smith, Kevin},
  url    = {https://github.com/smithk/salt},
  year   = {2026}
}
```

## Licence

[MIT](LICENSE). Use it, change it, ship it; just keep the copyright notice.

The dependencies are all permissively licensed too (scikit-learn BSD-3-Clause,
LightGBM MIT, XGBoost and CatBoost Apache-2.0), so nothing here obliges you to
open-source what you build with it.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
