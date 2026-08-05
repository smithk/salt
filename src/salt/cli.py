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
        prog="salt",
        description="Suggest a machine learning model and hyperparameters for your dataset.",
    )
    parser.add_argument("--version", action="version", version=f"salt {__version__}")
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
        return _run_fit(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
