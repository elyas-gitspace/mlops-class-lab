"""Construction of UNFITTED preprocessing + classifier pipelines.

Builders never read data and never call fit. Every call returns fresh,
independent objects, so candidates never share fitted state.

Preprocessing policy:
- numeric (tenure, MonthlyCharges, TotalCharges): missing -> training median,
  then StandardScaler (mean/std learned on training data).
- categorical (Contract, InternetService, PaymentMethod): missing -> most
  frequent training category, then one-hot encoding. A category never seen in
  training is encoded as all zeros for that feature (handle_unknown="ignore").
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from churnlab.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

ESTIMATORS = {
    "DummyClassifier": DummyClassifier,
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
}


def build_preprocessor(
    numeric_features: list[str] = NUMERIC_FEATURES,
    categorical_features: list[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ],
        remainder="drop",
    )


def build_pipeline(classifier) -> Pipeline:
    """Fresh preprocessor + the given (unfitted) classifier."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def make_estimator(estimator: str, params: dict[str, Any], seed: int):
    """Instantiate a classifier from its config; inject the seed if supported."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator {estimator!r}. Known: {sorted(ESTIMATORS)}")
    instance = ESTIMATORS[estimator](**params)
    if "random_state" in instance.get_params():
        instance.set_params(random_state=seed)
    return instance
