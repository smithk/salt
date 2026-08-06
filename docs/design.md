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
per-trial costs, which no fixed trial count can express.

A time budget is spent in two phases:

1. **Survey** (`SURVEY_FRACTION`, 30%) — every applicable learner gets an equal
   slice of wall-clock, with a floor of `MIN_SURVEY_SECONDS`. This guarantees
   each family is tried at least once, which a trial budget does not, and it
   measures what each one costs before deciding where to spend.
2. **Focus** (70%) — Optuna TPE restricted to the families still in contention.

`--trials N` skips the survey and runs a single study. That is the reproducible
mode, where equal trial counts are the point.

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

## What is not ported

The **shrinking-hypercube optimiser** from the 2014 version. It keeps one
hypercube per categorical signature, expands 1.2× on improvement, shrinks
0.97× otherwise, clamps to prior bounds, and resets when it collapses below
`1e-4`. It is the most distinctive idea in the original codebase and it is not
a textbook algorithm.

Porting it means implementing `optuna.samplers.BaseSampler`; its one
requirement, the categorical *signature*, is available as the tuple of
categorical values in `trial.params`. `--sampler hypercube` currently raises a
message saying it is unported rather than silently falling back.

Benchmarking it against TPE across `cc18-lite` and `ctr23-lite` is the point of
having the benchmark harness.
