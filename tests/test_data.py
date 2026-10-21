import numpy as np
import pandas as pd
import pytest

from churnlab.data import parse_features, parse_target
from churnlab.schema import FEATURES


def record(**overrides):
    base = {
        "customerID": "0001-TEST",
        "tenure": "12",
        "MonthlyCharges": "70.5",
        "TotalCharges": "846.0",
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_parse_features_returns_documented_columns_and_types():
    parsed = parse_features(record())
    assert list(parsed.columns) == FEATURES
    assert parsed["TotalCharges"].dtype == "float64"
    assert parsed["Contract"].dtype == object


@pytest.mark.parametrize("blank", [" ", "", "   ", None, np.nan])
def test_blank_total_charges_becomes_missing(blank):
    assert np.isnan(parse_features(record(TotalCharges=blank))["TotalCharges"].iloc[0])


def test_malformed_numeric_text_is_rejected():
    with pytest.raises(ValueError, match="non-numeric"):
        parse_features(record(TotalCharges="12,5abc"))


def test_missing_required_column_is_rejected():
    with pytest.raises(ValueError, match="Missing required feature columns"):
        parse_features(record().drop(columns=["PaymentMethod"]))


def test_extra_columns_are_not_model_inputs():
    parsed = parse_features(record(Churn="Yes"))
    assert "Churn" not in parsed.columns and "customerID" not in parsed.columns


def test_parse_target_encodes_positive_class_and_rejects_unknown():
    assert parse_target(pd.Series(["No", "Yes"])).tolist() == [0, 1]
    with pytest.raises(ValueError):
        parse_target(pd.Series(["Yes", "maybe"]))
