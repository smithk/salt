# Design notes

Why the search works the way it does. For usage, see the
[README](../README.md).

## Budget allocation

**The budget is wall-clock time, not a trial count.** Learner costs span three
orders of magnitude: a decision tree fits in hundredths of a second, a TabPFN
forward pass takes tens. Counting trials treats those as equal units, which
makes runtime unpredictable and quietly starves the expensive families — which
on tabular data are often the ones worth having.

The evidence is in any benchmark run. A 20-second budget bought 817 trials on
`diabetes` and 58 on `spambase`: the same wall-clock against very different
per-trial costs, which no fixed trial count can express. (Those counts were
measured before early stopping was added, so both are higher today; the ratio
between them is the point and it has not moved.)

A time budget is spent in two phases:

1. **Survey** (`SURVEY_FRACTION`, 30%) — every applicable learner gets an equal
   slice of wall-clock, with a floor of `MIN_SURVEY_SECONDS`. This guarantees
   each family is tried at least once, which a trial budget does not, and it
   measures what each one costs before deciding where to spend.
2. **Focus** (70%) — Optuna TPE restricted to the families still in contention.

`--trials N` skips the survey and runs a single study. That is the reproducible
mode, where equal trial counts are the point.

### Not finishing hopeless trials

Every trial used to run all its folds, including configurations that were
plainly losing after the first one. Under a time budget that is the most
expensive habit the search had: the budget buys folds, and a fold spent
confirming that a bad configuration is bad buys nothing.

Folds are therefore evaluated in chunks of `PRUNE_CHUNK`, with the running mean
reported to Optuna between chunks so a pruner can compare this trial against
the ones already finished. Chunks rather than single folds because fold-level
parallelism is where a trial gets its speed — a chunk still runs in parallel,
and the gap between chunks is simply where a decision becomes possible. Two
buys a decision after 40% of a five-fold trial while keeping most of the
parallelism.

The survey phase is deliberately never pruned. Its fixed slice per learner is
what measures cost, and cutting a learner short would settle the algorithm
choice on partial evidence — the opposite of what the survey is for.

One trap worth recording: Optuna signals "stop this trial" by raising
`TrialPruned`, and the search already caught every exception around
cross-validation to keep a bad configuration from ending the run. A pruned
trial therefore looked exactly like a crashed one and was counted as a failure.
`TrialPruned` is re-raised ahead of the general handler, and `n_pruned` is
reported separately from `n_failed`.

### Starting from something known to work

A search that begins at the prior spends its first trials rediscovering that
gradient boosting usually wants a few hundred trees and a small learning rate.
On a sixty-second budget those trials are a meaningful share of everything the
tool will ever do.

`portfolio.py` holds a short queue of configurations that are strong across a
wide range of tabular data, enqueued before the study starts. Optuna runs a
queued trial as an ordinary trial, so nothing is trusted: an entry that suits
this dataset badly is just a trial that scored badly. Entries are ordered
breadth-first — every learner's first entry before any learner's second — so a
budget too small to drain the queue still spans the algorithms instead of
tuning one.

This is the cheap half of what auto-sklearn calls meta-learning. The expensive
half, choosing entries by dataset similarity, needs meta-features and a corpus
of prior runs.

Only the phase that *first* sees a learner is warm-started: the survey under a
time budget, the single study under `--trials`. The focus phase is a separate
study, so warm-starting it too would re-run configurations the survey had
already measured.

## Combining trials instead of discarding them

A search fits hundreds of models and keeps one. The runner-up is usually almost
as good and wrong about different rows, which is the condition under which
averaging helps, and both have already been paid for.

`--ensemble` uses Caruana's greedy selection: start empty, repeatedly add
whichever candidate most improves the blend, with replacement. Choosing with
replacement is what turns selection into weighting — picked three times out of
ten means weight 0.3 — and the greedy step means a member joins only if it
improves *the ensemble*, which a plain top-k average never checks.

Two details that decide whether this works at all:

- **Selection scores out-of-fold predictions**, never the rows a model was
  fitted on, and never the holdout. In-sample selection would hand the
  ensemble to whichever candidate overfits hardest; holdout selection would
  spend the one honest estimate the tool has.
- **The rounds always run to the end and the best-scoring prefix wins**, rather
  than stopping at the first round that fails to improve. Re-picking the
  current leader is score-neutral, so an improvement test halts immediately and
  never finds the combinations that lie one step past a plateau.

Not every classifier offers probabilities — an SVC is fitted with
`probability=False`, a ridge classifier has none at all. Their hard labels are
widened into one-hot rows so they vote at full confidence rather than being
excluded, which on some datasets is the difference between an ensemble and no
ensemble at all.

### Choosing contenders

Contenders are the learners whose survey best falls within the top quarter of
the observed score *spread*, not within a fixed absolute margin and not a fixed
top-k. Metric scales differ — balanced accuracy sits in `[0, 1]`, r² is
unbounded below — so the threshold has to come from the data rather than from a
constant.

There is a floor of three survivors, so one lucky survey trial cannot narrow
the search to a single family. With three or fewer candidates in total, nothing
is dropped: there is nothing to save.

### The budget is a target, not a ceiling

Optuna checks the clock *between* trials, so a trial already running cannot be
interrupted. One slow learner overshoots — a single 18-second TabPFN trial
blows a 2-second survey slice. Runs report elapsed against requested time and
say when they went over.

Fixing this properly needs per-trial interruption, which needs process
isolation. That belongs behind the `Runner` interface (`runner.py`), alongside
a future cluster backend, not bolted onto the search loop.

## Measuring cost

Fit cost and prediction cost are recorded separately for every trial, from
`cross_validate`'s `fit_time` and `score_time`. No extra fitting is needed —
the numbers were already there and were previously discarded.

They are different questions. Fit cost is what the *search* spends; prediction
cost is what *deployment* spends. They diverge sharply, and sometimes invert:
TabPFN barely fits at all — its "training" is a forward pass over context rows
— and then predicts slowly.

Prediction cost is normalised to milliseconds per 1,000 rows so it is
comparable across datasets. It includes fixed per-call overhead, so on small
data the absolute figure has a floor of a few tens of milliseconds. It is a
comparison between learners on one dataset, not a latency guarantee.

### The frontier

`SearchResult.frontier()` returns configurations beaten on neither accuracy nor
prediction cost. `recommended()` returns the cheapest within 1% of the best
score, which is usually the honest answer to "what should I use": differences
below a percent rarely exceed the noise across folds, while a 400× difference
in serving cost is real.

## Search space

Conditional hyperparameters — where one choice unlocks others — are ordinary
Python control flow in a space function:

```python
if s.cat("penalty", ["l1", "l2"]) == "l2":
    s.cat("dual", [True, False])
```

The 2014 codebase had a 591-line `param.py` building a tree of
`CategoricalParameter` objects with weighted branches to express exactly this,
plus a second competing implementation in `parameters.py` that the migration
never finished. Optuna's define-by-run API replaces both.

Every learner shares one study, so parameter names are namespaced by learner
(`svm__C`, `logistic_regression__C`). Without that, two learners' `C` — with
different ranges — collide and Optuna rejects the second definition.

## Library compatibility

Three adapters exist because a library misbehaves under scikit-learn's
contract. All three failure modes appeared *only* under cross-validation and
not in a direct `fit()`, because `cross_validate` clones the estimator for
every fold.

**XGBoost** requires the target to be exactly `[0..n_classes-1]` and rejects
string labels, unlike every other classifier here. `_XGBClassifierWithLabels`
encodes inside fit/predict so the labels reported back are the user's own.

**CatBoost** cannot take `cat_features` through its constructor at all.
`cat_features=[]` is silently dropped, and even a non-empty value does not
survive `get_params()` by *identity*, which `clone` requires. It is supplied at
fit time instead. Symptom before the fix: CatBoost failed every fold while
working fine standalone, and would have been quietly absent from every result
table had `never_worked` reporting not existed.

**Both wrappers** hold their parameters in one dict and return a shallow copy
from `get_params`, satisfying `clone`'s identity check, and implement
`__sklearn_is_fitted__` — scikit-learn otherwise infers fittedness from
trailing-underscore attributes, which a wrapper holding its model privately
does not have.

## Preprocessing

Every candidate is a full pipeline: impute, encode, optionally scale, then the
estimator. The 2014 tool accepted only fully numeric ARFF and left everything
else to the user.

Scaling is per-learner (`needs_scaling`): distance- and margin-based methods
need it, tree ensembles are invariant and would pay only the cost.

Categorical encoding is one-hot except where a learner declares
`handles_categorical`, in which case it receives the raw columns. Only CatBoost
does, because its ordered target statistics beat one-hot decisively at high
cardinality, where one-hot explodes the feature count.

## The shrinking hypercube

The one idea in the 2014 codebase that is not a textbook algorithm, ported in
`hypercube.py` as an `optuna.samplers.BaseSampler` and reachable as
`--sampler hypercube`.

It keeps a box around the best configuration seen so far and samples inside it.
A draw that improves on the incumbent expands the box 1.2× and re-centres it on
the winner; a draw that does not shrinks it 0.97× around the unchanged
incumbent. Growth is fast and decay is slow, so one success outweighs a run of
failures. Boxes are clamped to the prior bounds, and a box that collapses below
a fraction of the prior range is discarded so sampling reverts to the prior —
the escape hatch from a local optimum.

There is one box per **categorical signature**, because a continuous
hyperparameter means different things under different discrete choices: an
SVM's `C` behaves differently under an RBF kernel than a linear one.

Two deliberate deviations from the original:

- The collapse threshold is a fraction of the prior range rather than an
  absolute `1e-4`. Absolute made collapse depend on the units a hyperparameter
  happened to use — a box on `[0, 1]` was declared collapsed a hundred times
  sooner than the same proportional box on `[0, 100]`.
- The first result under a signature establishes the incumbent without also
  expanding. The original compared against `None` and, under Python 2's
  permissive ordering, always took the improvement branch.

One structural difference is forced by Optuna. The 2014 optimiser drew a whole
configuration at once and always knew the complete signature. Optuna asks for
one parameter at a time, so when a float is drawn, the categoricals that come
after it in the sampling order do not exist yet. Each box is therefore keyed by
the categorical *context* present when its parameter was drawn, reconstructed
from `trial.distributions`, which preserves sampling order.

A consequence worth stating plainly: the categorical parts of the space —
including which learner to try — are still drawn from the prior, exactly as in
the original. The hypercube concentrates the *continuous* search only. It has
no mechanism for spending more trials on a promising learner, which is where
TPE's advantage would be expected to come from.
