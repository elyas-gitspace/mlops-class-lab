import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from churnlab.pipeline import build_pipeline, make_estimator
from churnlab.schema import ID_COLUMN, TARGET_COLUMN
from churnlab.split import allocation_table, split_customers


def toy_features(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n).astype(float),
            "MonthlyCharges": rng.uniform(20, 110, n),
            "TotalCharges": rng.uniform(20, 8000, n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], n),
        }
    ), pd.Series(np.tile([0, 1], n // 2))


def test_builder_returns_unfitted_and_independent_pipelines():
    first, second = build_pipeline(LogisticRegression()), build_pipeline(LogisticRegression())
    with pytest.raises(NotFittedError):
        check_is_fitted(first)
    assert first.named_steps["preprocessor"] is not second.named_steps["preprocessor"]


def test_imputer_learns_from_fit_data_only_and_predict_does_not_refit():
    X, y = toy_features()
    X.loc[0, "TotalCharges"] = np.nan
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    imputer = pipe.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    expected = X[["tenure", "MonthlyCharges", "TotalCharges"]].median().to_numpy()
    assert np.allclose(imputer.statistics_, expected)
    other, _ = toy_features(seed=1)
    pipe.predict(other)
    assert np.allclose(imputer.statistics_, expected)


def test_unknown_category_does_not_crash():
    X, y = toy_features()
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    new = X.iloc[[0]].copy()
    new["Contract"] = "Three year"
    assert pipe.predict_proba(new).shape == (1, 2)


def test_make_estimator_injects_seed():
    tree = make_estimator("DecisionTreeClassifier", {"max_depth": 3}, seed=7)
    assert tree.random_state == 7


def test_split_is_disjoint_stratified_and_reproducible():
    n = 1000
    df = pd.DataFrame(
        {ID_COLUMN: [f"C{i:04d}" for i in range(n)], TARGET_COLUMN: ["Yes"] * 250 + ["No"] * 750}
    )
    first = allocation_table(split_customers(df, seed=42))
    again = allocation_table(split_customers(df, seed=42))
    assert first.equals(again)
    assert first["split"].value_counts().to_dict() == {"train": 600, "valid": 200, "test": 200}
    splits = split_customers(df, seed=42)
    for frame in splits.values():
        assert (frame[TARGET_COLUMN] == "Yes").mean() == pytest.approx(0.25, abs=0.01)
