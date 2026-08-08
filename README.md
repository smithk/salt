# SALT — Suggest A Learner for Tabular data

<img src="docs/logo.png" alt="" width="240" align="right">

Point it at a table. It searches across algorithms and their hyperparameters
and hands back a fitted model, a ranked table of what worked, and what the
accuracy costs to serve.

Classification and regression, 13 learners each — from linear models to
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
outside the repository. Committing a published suite would put hundreds of
megabytes into git history permanently, and history cannot be shrunk afterwards.

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

A full `cc18-lite` run, 12 datasets against all 13 learners:

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
against very different per-trial costs.

> This run predates early stopping and warm-starting, both now on by default.
> The point it makes — that learners win on different data, and that per-trial
> cost varies by an order of magnitude — is unaffected, but the trial counts
> are no longer what you would see today. Due a refresh.

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

**The search starts from configurations already known to be strong** — a
few-hundred-tree forest, boosting at a small learning rate — instead of from
the prior. They are ordinary trials that must earn their score like any other;
the gain is a better starting point, not a shortcut past measurement.

**`--ensemble` combines the best trials instead of keeping only the winner.**
Members are chosen greedily on out-of-fold predictions, never on the holdout,
and a model that adds nothing is simply not selected — on some datasets the
result is a single model, which is the honest answer.

Full rationale, including the contention rule and why the budget overshoots:
[docs/design.md](docs/design.md).

## Accuracy vs prediction cost

The best model and the model you should deploy are often different. SALT
measures fit cost and prediction cost separately for every trial and reports
the frontier — configurations beaten on neither accuracy nor speed:

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
1% of the best, which is usually the honest answer.

Prediction cost is per 1,000 rows and includes fixed per-call overhead, so on
small datasets it has a floor of a few tens of milliseconds. It compares
learners on one dataset; it is not a latency guarantee.

## Data formats

Reads CSV, TSV, ARFF and Parquet. **Prefer Parquet**: it keeps column types, so
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
`--pruner median`, and warm-starting on. Ensembling is opt-in.

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
  author = {Bermudez-Chacon, Roger and Smith, Kevin and Horvath, Peter},
  url    = {https://github.com/smithk/salt},
  year   = {2026}
}
```

## History

SALT began in 2013–2014 as a research prototype by Roger Bermudez-Chacon,
Kevin Smith, and Peter Horvath — a CASH solver (combined algorithm selection
and hyperparameter optimisation) contemporary with Auto-WEKA. It was written
before deep learning reshaped the field, in Python 2, against scikit-learn 0.14.

The current version keeps the ideas worth keeping — the two-stage evaluation
protocol, and searching algorithms and hyperparameters jointly — and replaces
the hand-built process pool, cluster dispatch, Tkinter interface, and
conditional-parameter-space machinery with Optuna and modern scikit-learn.
11,600 lines of Python became about 2,500, plus 1,100 lines of tests where
there were none that ran.

The original **shrinking-hypercube optimiser has been ported** and is available
as `--sampler hypercube`. Benchmarking it was the point of building the suites
above, and the answer is no: TPE beats it. Across `cc18-lite` and `ctr23-lite`
it ranks last of three, and in its faithful 1:1 form it lost to *uniform random
sampling* on 10 of 11 classification tasks.

The reason turned out to be a defect rather than the idea. The original built
its box around the **first** completed trial under a signature — one random
draw — and then confined sampling to 5% of each range, shrinking further on
every draw that failed to beat it. There was no exploration phase at all, so a
mediocre first draw was polished for the rest of the budget. Drawing 20 trials
from the prior first and building the box around the best of them beats the
faithful port on 10 of 12 measured cells and closes most of the gap to TPE,
though it does not overtake it. That warm-up is on by default; `--sampler
hypercube` gives you the fixed version, and `n_startup_trials=0` gives you 2014.

One idea that sounds obvious and measured worse: letting TPE pick the learner
while the box tunes its continuous parameters. It seems to combine each
method's strength, and it loses to warm-up alone — TPE commits to a learner on
the evidence of early trials whose configurations are still poor, and cannot
back out. Kept as an option, off by default.

Details and the deviations from 2014 in
[docs/design.md](docs/design.md#the-shrinking-hypercube).

The 2014 implementation is preserved at the `v0.1-2014` tag and on the `master`
branch.

## Licence

**Not yet determined.** `pyproject.toml` currently declares `Proprietary`,
inherited from the 2014 prototype, and there is no LICENSE file — which means
default copyright applies and no reuse is permitted. If you intend this to be
usable by others, add a licence.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
