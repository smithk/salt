"""Integer-coded categories.

Nothing in a CSV distinguishes a site ID from a measurement, so a column of
1/2/3 is treated as a quantity by default and a linear model will read an
ordering into it. These tests cover the escape hatch and the warning that
points at it.
"""

import logging

import numpy as np
import pandas as pd
import pytest

import saltml
from saltml.data import as_categorical, load, suspect_categorical


def _coded_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "site": np.tile([1, 2, 3], n // 3),          # codes, not quantities
            "temp": rng.normal(size=n),                   # genuine measurement
            "flag": np.tile([0, 1], n // 2),              # binary: already encoded
            "y": np.tile(["p", "q"], n // 2),
        }
    )


def test_coded_column_is_numeric_without_help():
    dataset = load(_coded_frame(), warn_suspicious=False)
    assert "site" in dataset.numeric_columns


def test_declaring_a_column_categorical_moves_it():
    dataset = load(_coded_frame(), categorical=["site"], warn_suspicious=False)
    assert "site" in dataset.categorical_columns
    assert "site" not in dataset.numeric_columns
    assert dataset.forced_categorical == ["site"]


def test_declared_column_is_one_hot_encoded():
    dataset = load(_coded_frame(), categorical=["site"], warn_suspicious=False)
    from saltml.preprocess import make_preprocessor

    encoded = make_preprocessor(dataset, scale=False).fit_transform(dataset.X)
    # site contributes three indicator columns instead of one numeric column.
    assert encoded.shape[1] == 2 + 3


def test_unknown_categorical_column_is_rejected():
    with pytest.raises(KeyError, match="banana"):
        load(_coded_frame(), categorical=["banana"])


def test_suspect_flags_codes_but_not_measurements_or_binary():
    frame = _coded_frame()
    suspects = suspect_categorical(frame.drop(columns=["y"]))
    assert "site" in suspects
    assert "temp" not in suspects   # continuous
    assert "flag" not in suspects   # binary is already its own encoding


def test_warning_names_the_column_and_the_fix(caplog):
    with caplog.at_level(logging.WARNING, logger="saltml"):
        load(_coded_frame())
    assert "site" in caplog.text
    assert "--categorical" in caplog.text


def test_no_warning_once_declared(caplog):
    with caplog.at_level(logging.WARNING, logger="saltml"):
        load(_coded_frame(), categorical=["site"])
    assert "site" not in caplog.text


def test_as_categorical_preserves_missing_values():
    converted = as_categorical(pd.Series([1.0, np.nan, 3.0]))
    assert converted.isna().tolist() == [False, True, False]
    assert converted.dropna().tolist() == ["1", "3"]


def test_holdout_split_keeps_the_declaration():
    result = saltml.fit(
        _coded_frame(120), categorical=["site"], learners=["decision_tree"],
        n_trials=2, folds=3, seed=0,
    )
    assert result.dataset.forced_categorical == ["site"]


def test_declaring_categoricals_changes_what_a_linear_model_learns():
    """The point of the flag: codes must not be read as an ordering.

    ``site`` determines the target non-monotonically, so a linear model can
    only fit it once the column is one-hot encoded.
    """
    rng = np.random.default_rng(0)
    site = rng.choice([1, 2, 3], 300)
    # Site 2 is the odd one out, which no ordering of 1 < 2 < 3 can express.
    target = np.where(site == 2, 10.0, 0.0) + rng.normal(scale=0.1, size=300)
    frame = pd.DataFrame({"site": site, "noise": rng.normal(size=300), "y": target})

    numeric = saltml.fit(frame, learners=["ridge"], n_trials=4, folds=3,
                       holdout=0.0, seed=0)
    encoded = saltml.fit(frame, categorical=["site"], learners=["ridge"],
                       n_trials=4, folds=3, holdout=0.0, seed=0)

    assert numeric.cv_score < 0.2       # ordering cannot capture it
    assert encoded.cv_score > 0.9       # one-hot can
