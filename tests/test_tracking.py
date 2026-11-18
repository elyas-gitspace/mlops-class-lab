"""TP03 tracking helpers: no server needed (nothing is sent to MLflow here)."""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from churnlab.schema import FEATURES, TARGET_COLUMN
from churnlab.tracking import (
    confusion_matrix_frame,
    dataset_frame,
    resolve_tracking_uri,
    run_params,
    validation_metrics,
)


def fake_metrics():
    return {
        "average_precision": 0.6, "precision": 0.5, "recall": 0.4, "f1": 0.44,
        "roc_auc": 0.8, "accuracy": 0.75, "predicted_positive_rate": 0.2,
        "confusion_matrix": {"matrix": [[10, 2], [3, 5]]},
    }


def test_validation_metrics_are_numeric_and_prefixed():
    metrics = validation_metrics(fake_metrics())
    for key in ["val_average_precision", "val_precision", "val_recall", "val_f1"]:
        assert key in metrics
    assert all(key.startswith("val_") and isinstance(v, float) for key, v in metrics.items())


def test_confusion_matrix_frame_orders_actual_rows_and_predicted_columns():
    frame = confusion_matrix_frame(fake_metrics())
    assert list(frame.index) == ["actual_0_no_churn", "actual_1_churn"]
    assert list(frame.columns) == ["predicted_0_no_churn", "predicted_1_churn"]
    assert frame.loc["actual_1_churn", "predicted_0_no_churn"] == 3


def test_run_params_read_actual_classifier_settings():
    spec = {"name": "lr", "estimator": "LogisticRegression", "complexity_rank": 1}
    params = run_params(spec, LogisticRegression(C=0.3, random_state=42), 42, 0.5)
    assert params["model_type"] == "LogisticRegression"
    assert params["clf__C"] == 0.3
    assert params["threshold"] == 0.5 and params["random_state"] == 42


def test_dataset_frame_has_six_parsed_features_and_target_only():
    raw = pd.DataFrame({
        "customerID": ["A", "B"], "tenure": ["1", "2"], "MonthlyCharges": ["10.5", "20"],
        "TotalCharges": [" ", "40"], "Contract": ["Month-to-month", "One year"],
        "InternetService": ["DSL", "No"], "PaymentMethod": ["Mailed check", "Electronic check"],
        "Churn": ["Yes", "No"],
    })
    frame = dataset_frame(raw)
    assert list(frame.columns) == [*FEATURES, TARGET_COLUMN]
    assert frame["TotalCharges"].isna().iloc[0]


def test_tracking_uri_priority(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri(None, "http://cfg") == "http://cfg"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env")
    assert resolve_tracking_uri(None, "http://cfg") == "http://env"
    assert resolve_tracking_uri("http://cli", "http://cfg") == "http://cli"
