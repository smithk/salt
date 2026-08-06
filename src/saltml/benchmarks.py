"""Public benchmark suites, fetched on demand.

Benchmark data is not vendored into the repository. Committing a suite would
put hundreds of megabytes into git history permanently, and git history cannot
be made smaller after the fact. OpenML is a stable, citable host, so suites are
fetched into a cache outside version control and reused from there.

The datasets under ``data/`` are a different thing: a small offline corpus the
test suite depends on. These suites are for measuring how good SALT actually
is, across many datasets at once.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from .data import Dataset, load
from .task import Task

__all__ = ["Suite", "SUITES", "cache_dir", "fetch", "iter_suite", "resolve_suite"]

log = logging.getLogger("saltml")


@dataclass(frozen=True)
class Entry:
    """One dataset within a suite, identified by its OpenML id."""

    name: str
    openml_id: int
    task: Task


@dataclass(frozen=True)
class Suite:
    name: str
    description: str
    reference: str
    entries: tuple[Entry, ...] = field(default_factory=tuple)

    @property
    def tasks(self) -> set[Task]:
        return {entry.task for entry in self.entries}

    def for_task(self, task: Task | None) -> list[Entry]:
        return [e for e in self.entries if task is None or e.task is task]


C = Task.CLASSIFICATION
R = Task.REGRESSION

# Curated subsets rather than the full published suites. The full CC18 is 72
# datasets and several gigabytes, which is a benchmarking session, not a
# smoke test. These cover the shapes that break things: mixed types, high
# cardinality, missing values, wide data, and a range of sizes.
SUITES: dict[str, Suite] = {
    "smoke": Suite(
        name="smoke",
        description="Five small datasets. Seconds to fetch, minutes to run.",
        reference="Assorted OpenML classics.",
        entries=(
            Entry("iris", 61, C),
            Entry("wine", 187, C),
            Entry("breast-w", 15, C),
            Entry("autoPrice", 207, R),
            Entry("cholesterol", 204, R),
        ),
    ),
    "cc18-lite": Suite(
        name="cc18-lite",
        description="Twelve classification tasks sampled from OpenML-CC18.",
        reference="Bischl et al., OpenML Benchmarking Suites (NeurIPS 2021). "
                  "Full suite: https://www.openml.org/s/99",
        entries=(
            Entry("credit-g", 31, C),          # mixed categorical/numeric
            Entry("diabetes", 37, C),
            Entry("tic-tac-toe", 50, C),       # all categorical
            Entry("vehicle", 54, C),
            Entry("kr-vs-kp", 3, C),           # all categorical, larger
            Entry("sick", 38, C),              # missing values
            Entry("spambase", 44, C),
            Entry("phoneme", 1489, C),
            Entry("banknote-authentication", 1462, C),
            Entry("blood-transfusion-service-center", 1464, C),
            Entry("climate-model-simulation-crashes", 1467, C),
            Entry("ilpd", 1480, C),
        ),
    ),
    "ctr23-lite": Suite(
        name="ctr23-lite",
        description="Ten regression tasks sampled from OpenML-CTR23.",
        reference="Fischer et al., OpenML-CTR23 (AutoML 2023). "
                  "Full suite: https://www.openml.org/s/353",
        entries=(
            Entry("abalone", 44956, R),
            Entry("concrete_compressive_strength", 44959, R),
            Entry("energy_efficiency", 44960, R),
            Entry("forest_fires", 44962, R),
            Entry("solar_flare", 44966, R),
            Entry("student_performance_por", 44967, R),
            Entry("red_wine", 44972, R),
            Entry("california_housing", 44977, R),
            Entry("cpu_activity", 44978, R),
            Entry("QSAR_fish_toxicity", 44970, R),
        ),
    ),
}


def cache_dir() -> Path:
    """Where fetched datasets live.

    Honours ``SALTML_CACHE``, then ``XDG_CACHE_HOME``, then ``~/.cache``.
    Deliberately outside the repository so nothing fetched can be committed
    by accident.
    """
    override = os.environ.get("SALTML_CACHE")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base).expanduser() / "saltml" / "benchmarks"


def resolve_suite(name: str) -> Suite:
    try:
        return SUITES[name]
    except KeyError:
        raise KeyError(
            f"Unknown suite {name!r}. Available: {', '.join(sorted(SUITES))}"
        ) from None


def _cache_path(entry: Entry) -> Path:
    return cache_dir() / f"{entry.openml_id}_{entry.name}.parquet"


def fetch(entry: Entry, *, refresh: bool = False) -> Path:
    """Download one dataset to the cache, returning its path.

    Stored as Parquet so column types survive; a CSV round trip would turn
    every categorical column back into a guess.
    """
    destination = _cache_path(entry)
    if destination.exists() and not refresh:
        return destination

    from sklearn.datasets import fetch_openml

    destination.parent.mkdir(parents=True, exist_ok=True)
    log.info("Fetching %s (OpenML id %d)...", entry.name, entry.openml_id)
    started = time.perf_counter()
    bunch = fetch_openml(data_id=entry.openml_id, as_frame=True, parser="auto")

    frame = bunch.data.copy()
    target = bunch.target
    # Land the target in the last column, which is where load() looks by default.
    frame["__target__"] = target.values if hasattr(target, "values") else target
    frame.to_parquet(destination, index=False)
    log.info("  %s: %s in %.1fs", entry.name, frame.shape, time.perf_counter() - started)
    return destination


def load_entry(entry: Entry, *, refresh: bool = False) -> Dataset:
    """Fetch if needed, then load as a Dataset."""
    path = fetch(entry, refresh=refresh)
    return load(path, "__target__", task=entry.task, name=entry.name,
                warn_suspicious=False)


def iter_suite(
    suite: Suite | str,
    *,
    task: Task | None = None,
    refresh: bool = False,
) -> Iterator[Dataset]:
    """Yield every dataset in a suite, fetching as needed."""
    resolved = resolve_suite(suite) if isinstance(suite, str) else suite
    for entry in resolved.for_task(task):
        try:
            yield load_entry(entry, refresh=refresh)
        except Exception as exc:  # one unavailable dataset must not stop a run
            log.warning("Skipping %s: %s: %s", entry.name, type(exc).__name__, exc)
