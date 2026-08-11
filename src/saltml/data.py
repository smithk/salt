"""Loading datasets and working out what kind of problem they pose.

Accepts CSV, ARFF, and in-memory pandas objects. ARFF stays supported because
the benchmark corpus under ``data/`` is entirely ARFF.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .task import Task

__all__ = [
    "Dataset",
    "DatasetError",
    "load",
    "detect_task",
    "as_categorical",
    "suspect_categorical",
]

log = logging.getLogger("saltml")

# A numeric target is treated as class labels rather than a regression target
# when it takes few enough distinct values, both absolutely and relative to the
# sample count. Integer-coded classes are common and must not be regressed on.
_MAX_DISCRETE_LABELS = 20
# Distinct values only look like class labels if enough rows share each one.
# Twenty distinct integers across twenty-five rows is an identifier, not a
# target; the same twenty across a thousand rows is a classification problem.
_MIN_ROWS_PER_LABEL = 3

# A numeric feature with at most this many distinct integer values is worth
# querying: it may be a code rather than a measurement.
_SUSPICIOUS_LEVELS = 12

# An integer target running over consecutive levels — ratings 1-5, grades
# 0-20 — is a rating scale, and a rating scale is genuinely both tasks. Below
# this many levels the ordering carries too little to regress on and it is
# just class labels, so the query would only be noise.
_ORDINAL_MIN_LEVELS = 5


@dataclass
class Dataset:
    """Features, target, and the task they imply."""

    X: pd.DataFrame
    y: pd.Series
    task: Task
    name: str = "dataset"
    #: Columns the caller forced to be treated as labels.
    forced_categorical: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def categorical_columns(self) -> list[str]:
        return [c for c in self.X.columns if not is_numeric(self.X[c])]

    @property
    def numeric_columns(self) -> list[str]:
        return [c for c in self.X.columns if is_numeric(self.X[c])]

    @property
    def class_names(self) -> list[str] | None:
        if self.task is not Task.CLASSIFICATION:
            return None
        return [str(v) for v in sorted(self.y.unique())]

    def describe(self) -> str:
        bits = [
            f"{self.name}: {self.n_samples} samples x {self.n_features} features",
            f"task={self.task}",
        ]
        if self.task is Task.CLASSIFICATION:
            bits.append(f"{self.y.nunique()} classes")
        n_cat = len(self.categorical_columns)
        if n_cat:
            bits.append(f"{n_cat} categorical features")
        return ", ".join(bits)


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def detect_task(y: pd.Series) -> Task:
    """Infer whether ``y`` holds class labels or continuous values.

    Non-numeric targets are always classification. Numeric targets are
    classification only when they look like a small set of discrete codes.
    """
    # These checks come before the dtype branch: a target of all-None or a
    # single repeated label is just as unusable when it is text as when it is
    # numeric, and returning CLASSIFICATION for it only defers the failure to
    # somewhere less legible.
    values = y.dropna()
    if values.empty:
        raise ValueError("Target column is entirely missing.")

    n_unique = values.nunique()
    if n_unique <= 1:
        raise ValueError(f"Target column has only {n_unique} distinct value(s); nothing to learn.")

    if not is_numeric(y):
        return Task.CLASSIFICATION

    looks_integral = bool(np.allclose(values, np.round(values)))
    few_enough = n_unique <= _MAX_DISCRETE_LABELS
    well_populated = n_unique * _MIN_ROWS_PER_LABEL <= len(values)

    if looks_integral and few_enough and well_populated:
        return Task.CLASSIFICATION
    return Task.REGRESSION


def looks_ordinal(y: pd.Series) -> bool:
    """Whether a target is a rating scale rather than clearly one task.

    The signature is an integer column whose values run consecutively over
    their range: 1-5 stars, 0-20 grades, a 1-10 severity score. Those are
    legitimately either task — classification keeps the levels distinct and
    scores every mistake alike, regression uses the ordering and treats being
    one level out as a smaller error than being five — and which is better is
    an empirical question, not something a heuristic can settle.

    Arbitrary class codes usually fail the consecutive test, and anything with
    fewer than ``_ORDINAL_MIN_LEVELS`` levels is treated as plain labels.
    """
    if not is_numeric(y):
        return False
    values = y.dropna()
    if values.empty:
        return False
    if not bool(np.allclose(values, np.round(values))):
        return False
    n_unique = values.nunique()
    if n_unique < _ORDINAL_MIN_LEVELS:
        return False
    span = int(values.max()) - int(values.min()) + 1
    return span == n_unique


class DatasetError(ValueError):
    """A dataset file could not be read or made sense of.

    Subclasses ValueError so callers that already handle bad input — the CLI
    among them — keep working without knowing about this type.
    """


def _unreadable(path: Path, detail: str) -> DatasetError:
    return DatasetError(
        f"Could not read {path} as a {path.suffix.lstrip('.') or 'data'} file: {detail}"
    )


def _read_arff(path: Path) -> pd.DataFrame:
    from scipy.io import arff

    raw, _meta = arff.loadarff(str(path))
    frame = pd.DataFrame(raw)
    # scipy returns nominal attributes as bytes, and ARFF's missing marker
    # survives as a literal '?'. Both need normalising before use.
    for column in frame.columns:
        if frame[column].dtype == object:
            decoded = frame[column].apply(
                lambda v: v.decode("utf-8", "replace") if isinstance(v, bytes) else v
            )
            frame[column] = decoded.replace("?", np.nan)
    return frame


#: Extensions we know how to read, and how.
READABLE_SUFFIXES = (".csv", ".tsv", ".tab", ".arff", ".parquet", ".pq")


def _read_table(path: Path) -> pd.DataFrame:
    """Read a file into a DataFrame, or fail with a message naming the file.

    Every underlying reader has its own idea of how to complain — scipy's ARFF
    parser raises a bare ``StopIteration`` with no message on an empty file,
    which tells a user nothing at all. They are normalised here so the CLI can
    print one clear line instead of a traceback.
    """
    if path.stat().st_size == 0:
        raise _unreadable(path, "the file is empty")

    suffix = path.suffix.lower()
    try:
        if suffix == ".arff":
            return _read_arff(path)
        if suffix in {".tsv", ".tab"}:
            return pd.read_csv(path, sep="\t")
        if suffix in {".parquet", ".pq"}:
            try:
                return pd.read_parquet(path)
            except ImportError as exc:  # pragma: no cover - depends on install
                raise ImportError(
                    "Reading Parquet needs pyarrow, which should have been installed "
                    "with saltml. Install it with: pip install pyarrow"
                ) from exc
        if suffix not in READABLE_SUFFIXES:
            log.warning(
                "Unrecognised extension %r; reading %s as CSV. Known formats: %s.",
                suffix or "(none)", path.name, ", ".join(READABLE_SUFFIXES),
            )
        return pd.read_csv(path)
    except (DatasetError, FileNotFoundError, PermissionError, ImportError):
        raise
    except UnicodeDecodeError as exc:
        raise _unreadable(path, "it is not text (binary content)") from exc
    except StopIteration as exc:
        # scipy's ARFF reader runs off the end of a file with no @data section
        # and raises StopIteration carrying no message at all.
        raise _unreadable(path, "it ended unexpectedly; the header may be missing "
                                "or incomplete") from exc
    except pd.errors.EmptyDataError as exc:
        raise _unreadable(path, "it contains no columns") from exc
    except Exception as exc:
        # Covers scipy's ParseArffError (an OSError, not a ValueError),
        # pandas' ParserError, and anything else a reader invents.
        raise _unreadable(path, str(exc) or type(exc).__name__) from exc


def as_categorical(series: pd.Series) -> pd.Series:
    """Recast a column so it is treated as labels rather than quantities.

    Values become plain strings and missing entries stay missing, which is the
    form the encoding pipeline expects.
    """
    converted = series.astype(object)
    present = series.notna()
    converted[present] = converted[present].map(
        lambda v: str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v)
    )
    converted[~present] = np.nan
    return converted


def suspect_categorical(X: pd.DataFrame) -> list[str]:
    """Numeric columns that look like they are really codes, not quantities.

    Integer-coded categories (site 1, site 2, site 3) are indistinguishable
    from measurements once a CSV has been written, and treating them as
    numbers lets a model read an ordering into them that is not there. Binary
    columns are excluded: 0/1 is already the encoding a category would get.
    """
    suspects: list[str] = []
    for column in X.columns:
        series = X[column]
        if not is_numeric(series):
            continue
        values = series.dropna()
        if values.empty or not bool(np.allclose(values, np.round(values))):
            continue
        if 2 < values.nunique() <= _SUSPICIOUS_LEVELS:
            suspects.append(str(column))
    return suspects


def load(
    source: str | Path | pd.DataFrame,
    target: str | int | None = None,
    *,
    task: Task | str | None = None,
    categorical: Sequence[str] | None = None,
    name: str | None = None,
    warn_suspicious: bool = True,
) -> Dataset:
    """Load a dataset and split off its target column.

    :param source: path to a CSV/TSV/ARFF/Parquet file, or a DataFrame.
    :param target: target column name or position. Defaults to the last
        column, which is the ARFF convention and the usual CSV one.
    :param task: force ``classification`` or ``regression`` instead of
        inferring it.
    :param categorical: feature columns to treat as labels regardless of their
        stored type. Use this for integer-coded categories, which nothing in a
        CSV distinguishes from measurements.
    :param name: label used in reports; defaults to the file stem.
    :param warn_suspicious: log a warning about numeric columns that look like
        codes and were not declared categorical.
    """
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
        default_name = name or "dataframe"
    else:
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"No such dataset: {path}")
        frame = _read_table(path)
        default_name = name or path.stem

    if frame.empty:
        raise DatasetError(f"{default_name} has no rows.")

    if target is None:
        target_name = frame.columns[-1]
    elif isinstance(target, int):
        target_name = frame.columns[target]
    else:
        if target not in frame.columns:
            raise KeyError(
                f"No column named {target!r}. Available columns: {list(frame.columns)}"
            )
        target_name = target

    y = frame[target_name]
    X = frame.drop(columns=[target_name])
    if X.shape[1] == 0:
        raise ValueError("Dataset has no feature columns once the target is removed.")

    # Rows with no target teach nothing; missing features are imputed later.
    keep = y.notna()
    if not keep.all():
        X, y = X[keep], y[keep]

    # Validated here rather than only in detect_task, so that forcing the task
    # with task= does not skip the check.
    if len(y) == 0:
        raise DatasetError(
            f"Every row of {default_name} is missing column {target_name!r}; "
            "nothing is left to learn from."
        )
    if y.nunique() <= 1:
        raise DatasetError(
            f"Column {target_name!r} has only {y.nunique()} distinct value(s) in "
            f"{default_name}; a model needs something to tell apart."
        )

    if categorical:
        unknown = [c for c in categorical if c not in X.columns]
        if unknown:
            raise KeyError(
                f"Cannot treat unknown column(s) as categorical: {', '.join(unknown)}. "
                f"Feature columns are: {list(X.columns)}"
            )
        X = X.copy()
        for column in categorical:
            X[column] = as_categorical(X[column])

    if warn_suspicious:
        overlooked = [c for c in suspect_categorical(X) if c not in set(categorical or ())]
        if overlooked:
            log.warning(
                "These columns hold few distinct whole numbers and are being treated as "
                "quantities: %s. If they are codes rather than measurements, pass "
                "categorical=%r (CLI: --categorical %s) so they are encoded as labels.",
                ", ".join(overlooked), overlooked, ",".join(overlooked),
            )

    resolved = Task(task) if task is not None else detect_task(y)
    # Only worth saying when the task was inferred: if it was passed in, the
    # caller has already answered the question.
    if warn_suspicious and task is None and looks_ordinal(y):
        instead = (
            Task.REGRESSION if resolved is Task.CLASSIFICATION else Task.CLASSIFICATION
        )
        log.warning(
            "Column %r holds %d consecutive whole numbers, which reads as a rating "
            "scale. It is being treated as %s; %s is equally defensible and can score "
            "quite differently, since only one of them uses the ordering. Pass "
            "task=%r (CLI: --task %s) to choose.",
            target_name, y.nunique(), resolved, instead, str(instead), instead,
        )
    if resolved is Task.CLASSIFICATION:
        y = y.astype("category")
    else:
        y = pd.to_numeric(y)

    return Dataset(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        task=resolved,
        name=default_name,
        forced_categorical=list(categorical or ()),
    )
