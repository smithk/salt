# Learners

Thirteen for each task, all searched by default. `saltml learners` lists them
with their preprocessing requirements. To add one, see
[CONTRIBUTING](../CONTRIBUTING.md#adding-a-learner).

| Family | Classification | Regression |
|---|---|---|
| Boosted trees | `lightgbm`, `xgboost`, `catboost`, `hist_gradient_boosting` | same |
| Bagged trees | `random_forest`, `extra_trees`, `decision_tree` | same |
| Neural | `tabpfn` | `tabpfn` |
| Linear | `logistic_regression`, `ridge_classifier` | `ridge`, `lasso`, `elastic_net` |
| Kernel & instance | `svm`, `knn` | `svr`, `knn` |
| Probabilistic | `gaussian_nb` | — |

Each registers only if its library imports, so an environment where one cannot
load — a missing OpenMP runtime being the usual cause — loses that learner
rather than failing to start. If a learner is missing from `saltml learners`,
that is why.

## TabPFN

TabPFN is a transformer pre-trained on synthetic tabular data. It is not
trained on your data at all: the training rows are given to the network as
context and it predicts in a single forward pass. On small tables it is often
the strongest thing in the registry — it won `vehicle` outright in the
`cc18-lite` run.

Three things make it unlike the other learners.

**It has almost no hyperparameters.** The search spends few trials on it
because there is little to search.

**It has hard size limits** inherited from pre-training — 10,000 samples, 500
encoded features, 10 classes. Past those it is excluded before the search
starts, with the reason printed, rather than failing every trial. That is what
`Learner.applies` exists for.

**Without a GPU the binding limit is far lower: 1,000 samples**, and TabPFN
refuses outright rather than merely running slowly. That limit is checked too,
because it fails in a way the pre-training one does not. Cross-validation fits
on a fraction of the data, so on a 1,372-row dataset every fold trained on ~915
rows, stayed under the limit, and TabPFN scored 1.0 and *won* — and then the
final refit on the full 1,029-row training split crossed the limit and raised,
losing the entire run. A learner that wins and then cannot be delivered is
worse than one that is excluded up front. Set
`TABPFN_ALLOW_CPU_LARGE_DATASET=1` to accept the speed instead.

**It is slow on CPU** — roughly 25–1000× slower per trial than the classical
learners, and slow at *prediction* specifically, which is the cost that matters
at deployment. A GPU changes this substantially; nothing else in the registry
benefits from one.

### Versions and weights

Weights are downloaded on first use, so the first TabPFN run needs network
access.

| Version | Weights |
|---|---|
| **2.x** (pinned default) | Downloaded from HuggingFace with no account |
| 8.x | Requires registering at [ux.priorlabs.ai](https://ux.priorlabs.ai), accepting the licence, and setting `TABPFN_TOKEN` |

The pin is `tabpfn>=2.2,<3` so the tool runs out of the box. Both lines work;
if you want 8.x, install it and set the token — the learner adapts.

There is also a hosted `tabpfn-client` that sends your data to PriorLabs'
servers. SALT does not use it, and for anything confidential you should not
either.

## Preprocessing per learner

Two flags change what a learner receives:

- **`needs_scaling`** — numeric features are standardised. Set for the linear,
  kernel and instance-based learners; tree ensembles are invariant to scaling
  and would pay only the cost.
- **`handles_categorical`** — the learner receives raw categorical columns
  instead of one-hot encoded ones. Only CatBoost sets this: its ordered target
  statistics beat one-hot decisively at high cardinality, where one-hot
  explodes the feature count.

Everything else gets the same pipeline: median imputation for numeric columns,
most-frequent for categorical, one-hot encoding, then optional scaling.
