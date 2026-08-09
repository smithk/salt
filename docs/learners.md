# Learners

Fourteen for each task, all searched by default. `saltml learners` lists them
with their preprocessing requirements. To add one, see
[CONTRIBUTING](../CONTRIBUTING.md#adding-a-learner).

| Family | Classification | Regression |
|---|---|---|
| Boosted trees | `lightgbm`, `xgboost`, `catboost`, `hist_gradient_boosting` | same |
| Bagged trees | `random_forest`, `extra_trees`, `decision_tree` | same |
| Neural | `tabpfn`, `mlp` | `tabpfn`, `mlp` |
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

## The MLP

A plain multi-layer perceptron, `sklearn.neural_network.MLP*`. It is here with
modest expectations, and it is worth being explicit about them.

**It will usually lose to boosting.** That is the consistent finding of the
tabular deep-learning literature — Grinsztajn et al. (2022), Shwartz-Ziv &
Armon (2022) — and the reasons are structural rather than a matter of tuning:
an MLP is biased toward smooth functions while tabular targets tend to be
irregular, it suffers more from uninformative features, and its rotation
invariance is a poor fit for data whose axes are individually meaningful.

**It is here anyway** because SALT picks a winner per dataset, so a learner
needs to win *sometimes*, not usually — the same argument that justifies
keeping `gaussian_nb`. It also fills a real gap: `tabpfn` is the other neural
option and is unusable above 1,000 rows without a GPU, which otherwise leaves
no neural model at all on a CPU-only machine for anything but small data.

**Measured, and worth being blunt about.** Across four datasets and three
seeds at a 60-second budget, making the MLP available moved the holdout score
by +0.001 on average — three wins, five exact ties, four losses. It won the
leaderboard once in twelve runs.

The obvious argument for including it was that a neural network's errors are
decorrelated from a tree ensemble's, so it should pull its weight inside
`--ensemble` even when it never wins. That did not happen. Greedy selection
chose it exactly once in twelve, and in that same run it had *already won* the
leaderboard outright, joining at weight 0.9 — so it was the ensemble rather
than a diversifying member. On the evidence here it contributes as an
occasional winner or not at all.

That is a weak case, honestly stated. It is kept because it costs no new
dependency, because one win in twelve is the same standing several other
learners have, and because the neural gap above 1,000 rows on CPU is real. It
is not kept because it was shown to help.

**Architecture is drawn from a named list** (`64`, `128`, `256`, `64x64`,
`128x64`, `256x128`) rather than searched as free width and depth. Two numbers
that interact this strongly would spend the budget on combinations that are
obviously too small or too slow. Weight decay and initial learning rate are
searched properly, because those are what decide whether an MLP works on
tabular data at all. Early stopping is always on, which is the same bargain the
pruner makes one level up.

Expect convergence warnings on some configurations. They are suppressed during
the search: a net that has not converged simply scores badly, and the score
already carries that information.

## Preprocessing per learner

Two flags change what a learner receives:

- **`needs_scaling`** — numeric features are standardised. Set for the linear,
  kernel, instance-based and neural learners; tree ensembles are invariant to
  scaling and would pay only the cost. For the MLP it is not an optimisation
  but a requirement: gradient descent on unscaled features is dominated by
  whichever column happens to have the largest units.
- **`handles_categorical`** — the learner receives raw categorical columns
  instead of one-hot encoded ones. Only CatBoost sets this: its ordered target
  statistics beat one-hot decisively at high cardinality, where one-hot
  explodes the feature count.

Everything else gets the same pipeline: median imputation for numeric columns,
most-frequent for categorical, one-hot encoding, then optional scaling.
