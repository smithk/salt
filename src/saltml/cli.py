"""Command line interface."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import optuna

from . import __version__, fit
from .learners import REGISTRY
from .metrics import DEFAULT_METRIC, describe_metric
from .task import Task

__all__ = ["main"]

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*$", re.IGNORECASE)
_MULTIPLIER = {"": 1, "s": 1, "m": 60, "h": 3600}


def parse_duration(text: str) -> float:
    """Turn ``30s`` / ``10m`` / ``1h`` / ``90`` into seconds."""
    match = _DURATION.match(text)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid duration {text!r}. Use forms like 90, 30s, 10m, 1h."
        )
    value, unit = match.groups()
    return float(value) * _MULTIPLIER[unit.lower()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saltml",
        description="Suggest a machine learning model and hyperparameters for your dataset.",
    )
    parser.add_argument("--version", action="version", version=f"saltml {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("fit", help="search for the best model and fit it")
    run.add_argument("data", help="CSV, TSV, ARFF, or Parquet file")
    run.add_argument("-t", "--target", help="target column (default: last column)")
    run.add_argument(
        "--task", choices=[t.value for t in Task], help="override task detection"
    )
    run.add_argument(
        "--categorical",
        metavar="COLS",
        help="comma-separated feature columns to treat as labels rather than "
             "numbers (for integer-coded categories such as site or region IDs)",
    )

    budget = run.add_argument_group("budget")
    budget.add_argument("--time", type=parse_duration, metavar="DURATION",
                        help="wall-clock budget, e.g. 10m")
    budget.add_argument("--trials", type=int, help="number of configurations to try")

    tuning = run.add_argument_group("search")
    tuning.add_argument("--metric", help="scoring metric (default depends on task)")
    tuning.add_argument("--learners", help="comma-separated subset to consider")
    tuning.add_argument("--folds", type=int, default=5, help="cross-validation folds")
    tuning.add_argument("--holdout", type=float, default=0.25,
                        help="fraction withheld for a final honest score (0 to disable)")
    tuning.add_argument("--sampler", default="tpe",
                        choices=["tpe", "random", "hypercube"],
                        help="search strategy")
    tuning.add_argument("--jobs", type=int, default=-1,
                        help="parallel cross-validation folds (-1 for all cores)")
    tuning.add_argument("--seed", type=int, default=0)

    output = run.add_argument_group("output")
    output.add_argument("-o", "--out", help="write the fitted model here (joblib)")
    output.add_argument("--top", type=int, default=10, help="leaderboard rows to show")
    output.add_argument("-q", "--quiet", action="store_true", help="only print the result")

    listing = subcommands.add_parser("learners", help="list available algorithms")
    listing.add_argument("--task", choices=[t.value for t in Task])

    bench = subcommands.add_parser(
        "bench", help="fetch and run public benchmark suites"
    )
    bench_actions = bench.add_subparsers(dest="bench_command", required=True)

    bench_actions.add_parser("list", help="show available suites and the cache location")

    grab = bench_actions.add_parser("fetch", help="download a suite into the cache")
    grab.add_argument("suite", help="suite name, or 'all'")
    grab.add_argument("--refresh", action="store_true", help="re-download even if cached")

    trial = bench_actions.add_parser("run", help="score SALT across a whole suite")
    trial.add_argument("suite")
    trial.add_argument("--task", choices=[t.value for t in Task],
                       help="restrict to one task type")
    trial.add_argument("--time", type=parse_duration, metavar="DURATION",
                       help="budget per dataset, e.g. 2m")
    trial.add_argument("--trials", type=int, help="trials per dataset")
    trial.add_argument("--learners", help="comma-separated subset to consider")
    trial.add_argument("--sampler", default="tpe",
                       choices=["tpe", "random", "hypercube"])
    trial.add_argument("--folds", type=int, default=5)
    trial.add_argument("--jobs", type=int, default=-1)
    trial.add_argument("--seed", type=int, default=0)
    trial.add_argument("-o", "--out", help="write per-dataset results as CSV")

    return parser


def _split_list(text: str | None) -> list[str] | None:
    """Turn ``a, b ,c`` into ``['a', 'b', 'c']``."""
    if not text:
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def _list_learners(task: str | None) -> int:
    tasks = [Task(task)] if task else list(Task)
    for entry in tasks:
        print(f"\n{entry.value} (default metric: {describe_metric(DEFAULT_METRIC[entry])})")
        for name, learner in sorted(REGISTRY[entry].items()):
            marker = " *" if learner.needs_scaling else ""
            print(f"  {name}{marker}")
    print("\n  * features are standardised for this learner")
    return 0


def _bench_list() -> int:
    from .benchmarks import SUITES, cache_dir

    for suite in SUITES.values():
        tasks = ", ".join(sorted(str(t) for t in suite.tasks))
        print(f"\n{suite.name}  ({len(suite.entries)} datasets: {tasks})")
        print(f"  {suite.description}")
        print(f"  {suite.reference}")
        print("  " + ", ".join(e.name for e in suite.entries))
    print(f"\ncache: {cache_dir()}")
    print("override with SALTML_CACHE. Nothing fetched is stored in the repository.")
    return 0


def _bench_fetch(args: argparse.Namespace) -> int:
    from .benchmarks import SUITES, fetch, resolve_suite

    suites = list(SUITES.values()) if args.suite == "all" else [resolve_suite(args.suite)]
    total = failed = 0
    for suite in suites:
        for entry in suite.entries:
            total += 1
            try:
                path = fetch(entry, refresh=args.refresh)
                size = path.stat().st_size / 1024
                print(f"  {entry.name:<36} {size:>8.0f} KB")
            except Exception as exc:
                failed += 1
                print(f"  {entry.name:<36} FAILED: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
    print(f"\n{total - failed}/{total} datasets cached.")
    return 1 if failed else 0


def _bench_run(args: argparse.Namespace) -> int:
    import pandas as pd

    from . import fit
    from .benchmarks import iter_suite, resolve_suite

    suite = resolve_suite(args.suite)
    task = Task(args.task) if args.task else None
    if args.time is None and args.trials is None:
        args.trials = 50

    rows = []
    for dataset in iter_suite(suite, task=task):
        print(f"\n=== {dataset.name} ({dataset.n_samples}x{dataset.n_features}) ===",
              file=sys.stderr)
        try:
            result = fit(
                dataset.X.assign(**{"__target__": dataset.y}),
                "__target__",
                task=dataset.task,
                learners=_split_list(args.learners),
                n_trials=args.trials,
                timeout=args.time,
                folds=args.folds,
                sampler=args.sampler,
                n_jobs=args.jobs,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            rows.append({"dataset": dataset.name, "task": str(dataset.task),
                         "error": f"{type(exc).__name__}: {exc}"})
            continue

        rows.append({
            "dataset": dataset.name,
            "task": str(dataset.task),
            "n": dataset.n_samples,
            "features": dataset.n_features,
            "metric": result.metric,
            "cv": round(result.cv_score, 4),
            "holdout": None if result.holdout_score is None else round(result.holdout_score, 4),
            "best_learner": result.learner,
            "trials": len(result.search.records),
        })
        print(f"  {result.learner}: cv={result.cv_score:.4f}", file=sys.stderr)

    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nWritten to {args.out}")
    return 0


def _run_bench(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if args.bench_command == "list":
        return _bench_list()
    if args.bench_command == "fetch":
        return _bench_fetch(args)
    return _bench_run(args)


def _run_fit(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.time is None and args.trials is None:
        args.trials = 100

    best_so_far = float("-inf")

    def progress(number, record):
        nonlocal best_so_far
        if args.quiet or record is None:
            return
        if record.score > best_so_far:
            best_so_far = record.score
            print(
                f"  trial {number:>4}  {record.learner:<24} {record.score:.4f}  (best)",
                file=sys.stderr,
            )

    result = fit(
        args.data,
        args.target,
        task=args.task,
        categorical=_split_list(args.categorical),
        learners=_split_list(args.learners),
        metric=args.metric,
        n_trials=args.trials,
        timeout=args.time,
        folds=args.folds,
        holdout=args.holdout,
        sampler=args.sampler,
        n_jobs=args.jobs,
        seed=args.seed,
        progress=progress,
    )

    print()
    print(result.summary())
    recommended = result.search.recommended()
    if recommended.learner != result.learner:
        ratio = result.search.best.predict_ms_per_1k / max(recommended.predict_ms_per_1k, 1e-9)
        print(
            f"\nCheaper alternative within 1%: {recommended.learner} "
            f"({recommended.score:.4f}, {ratio:.0f}x faster to predict)"
        )

    print()
    print("Accuracy vs prediction cost (nothing here is beaten on both):")
    print(result.search.tradeoff().to_string(index=False))

    print()
    print("Best per learner:")
    print(result.search.best_per_learner().to_string(index=False))
    if args.top:
        print()
        print(f"Top {args.top} configurations:")
        print(result.search.leaderboard(args.top).to_string(index=False))
    for name, reason in result.search.excluded.items():
        print(f"\nnot used  {name}: {reason}", file=sys.stderr)
    for name, reason in result.search.never_worked.items():
        print(f"\nfailed    {name}: never succeeded. {reason}", file=sys.stderr)
    if result.search.n_failed:
        print(f"\n{result.search.n_failed} trial(s) failed and were skipped.", file=sys.stderr)

    if args.out:
        written = result.save(args.out)
        print(f"\nModel written to {written}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "learners":
            return _list_learners(args.task)
        if args.command == "bench":
            return _run_bench(args)
        return _run_fit(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (ValueError, KeyError, OSError) as exc:
        # OSError covers a missing file, an unreadable one, and the OSError
        # subclasses that data readers raise; none should reach a user as a
        # traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
