"""Feature preparation shared by every learner.

The 2014 tool only accepted fully numeric ARFF. Handling mixed CSV data is
what makes the modern version usable on real files, so imputation and
categorical encoding are part of every candidate pipeline rather than
something the user is expected to arrange beforehand.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import Dataset

__all__ = ["make_preprocessor"]


def _one_hot() -> OneHotEncoder:
    # sklearn renamed this argument in 1.2; support both so the package works
    # across the versions a user is likely to already have installed.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(
    dataset: Dataset, *, scale: bool, encode_categorical: bool = True
) -> ColumnTransformer:
    """Build the impute/encode/scale step for one candidate pipeline.

    :param scale: standardise numeric features. Distance- and margin-based
        learners need it; tree ensembles are invariant to it and pay only the
        cost, so each learner declares its own requirement.
    :param encode_categorical: one-hot encode categorical columns. Set False
        for a learner that encodes them itself; the columns are still imputed
        and are passed through as strings.
    """
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="most_frequent")),
    ]
    if encode_categorical:
        categorical_steps.append(("encode", _one_hot()))

    transformer = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), dataset.numeric_columns),
            ("categorical", Pipeline(categorical_steps), dataset.categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if not encode_categorical:
        # A learner doing its own encoding needs column names to say which
        # columns are categorical, so keep the output a DataFrame.
        transformer.set_output(transform="pandas")
    return transformer
