import numpy as np
import pandas as pd
import pytest

from salt.data import Dataset, detect_task, load
from salt.task import Task

ARFF = "data/standard_ml_sets/classification/datasets/standard/iris.arff"


def test_detects_string_target_as_classification():
    assert detect_task(pd.Series(["a", "b", "a"])) is Task.CLASSIFICATION


def test_detects_integer_codes_as_classification():
    # Integer-coded classes must not be regressed on.
    assert detect_task(pd.Series([0, 1, 2] * 40)) is Task.CLASSIFICATION


def test_detects_continuous_target_as_regression():
    rng = np.random.default_rng(0)
    assert detect_task(pd.Series(rng.normal(size=500))) is Task.REGRESSION


def test_many_distinct_integers_are_regression():
    # Integral but too many distinct values to be labels.
    assert detect_task(pd.Series(range(500))) is Task.REGRESSION


def test_constant_target_is_rejected():
    with pytest.raises(ValueError, match="distinct value"):
        detect_task(pd.Series([1, 1, 1]))


def test_loads_arff():
    dataset = load(ARFF)
    assert dataset.task is Task.CLASSIFICATION
    assert dataset.n_samples == 150
    assert dataset.n_features == 4
    assert len(dataset.class_names) == 3


def test_target_defaults_to_last_column():
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "label": ["x", "y", "x", "y"]})
    assert load(frame).y.tolist() == ["x", "y", "x", "y"]


def test_target_by_name_and_position():
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": ["x", "y", "x", "y"]})
    assert load(frame, "a").y.tolist() == [1, 2, 3, 4]
    assert load(frame, 0).y.tolist() == [1, 2, 3, 4]


def test_unknown_target_names_available_columns():
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(KeyError, match="nope"):
        load(frame, "nope")


def test_rows_without_a_target_are_dropped():
    frame = pd.DataFrame({"a": [1, 2, 3], "y": ["x", None, "z"]})
    assert load(frame).n_samples == 2


def test_task_can_be_forced():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
    assert load(frame, task="regression").task is Task.REGRESSION


def test_categorical_and_numeric_columns_are_separated():
    frame = pd.DataFrame({"n": [1.0, 2.0], "c": ["a", "b"], "y": ["x", "y"]})
    dataset = load(frame)
    assert dataset.numeric_columns == ["n"]
    assert dataset.categorical_columns == ["c"]


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load("/nonexistent/nope.csv")
