"""Validation metrics. Positive class = Churn 'Yes' = 1."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_class_scores(model, X) -> np.ndarray:
    """Probability of class 1, taken from the predict_proba column matching classes_."""
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Positive class 1 not found in classes_ {classes}.")
    return model.predict_proba(X)[:, classes.index(1)]


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Predict churn (1) when score >= threshold."""
    return (np.asarray(scores) >= threshold).astype(int)


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    scores = positive_class_scores(model, X)
    y_pred = apply_threshold(scores, threshold)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "predicted_positive_rate": float(y_pred.mean()),
        "threshold": threshold,
        "confusion_matrix": {
            "labels": [0, 1],
            "label_meaning": {"0": "No churn", "1": "Churn"},
            "rows": "actual",
            "columns": "predicted",
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
