"""MLflow helpers for TP03: explicit tracking of one candidate per run.

Nothing here trains a model. These functions only turn what the TP02 workflow
already produces (config, splits, fitted pipeline, validation metrics) into
MLflow parameters, metrics, tags, artifacts, dataset inputs and logged models.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import mlflow
import pandas as pd

from churnlab.data import parse_features
from churnlab.schema import (
    CATEGORICAL_FEATURES,
    FEATURES,
    ID_COLUMN,
    NEGATIVE_LABEL,
    NUMERIC_FEATURES,
    POSITIVE_LABEL,
    TARGET_COLUMN,
)

# Validation metrics sent to MLflow: (MLflow key, key in churnlab.evaluate output)
VALIDATION_METRICS = [
    ("val_average_precision", "average_precision"),
    ("val_precision", "precision"),
    ("val_recall", "recall"),
    ("val_f1", "f1"),
    ("val_roc_auc", "roc_auc"),
    ("val_accuracy", "accuracy"),
    ("val_predicted_positive_rate", "predicted_positive_rate"),
]


def resolve_tracking_uri(cli_value: str | None, config_value: str) -> str:
    """Priority: --tracking-uri, then MLFLOW_TRACKING_URI, then the config file."""
    return cli_value or os.environ.get("MLFLOW_TRACKING_URI") or config_value


def setup_tracking(tracking_uri: str, experiment: str):
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.set_experiment(experiment)


def validation_metrics(metrics: dict) -> dict[str, float]:
    """Numeric, consistently named validation metrics for mlflow.log_metrics."""
    return {key: float(metrics[source]) for key, source in VALIDATION_METRICS}


def run_params(spec: dict, classifier, seed: int, threshold: float) -> dict:
    """Actual settings of the fitted classifier (read back from the object)."""
    params = {
        "model_name": spec["name"],
        "model_type": spec["estimator"],
        "random_state": seed,
        "split_seed": seed,
        "threshold": threshold,
        "complexity_rank": spec["complexity_rank"],
        "n_features": len(FEATURES),
    }
    for key, value in classifier.get_params(deep=False).items():
        params[f"clf__{key}"] = value
    return params


def dataset_frame(split_frame: pd.DataFrame) -> pd.DataFrame:
    """Six parsed features + raw target, used only to describe the dataset in MLflow."""
    frame = parse_features(split_frame).copy()
    frame[TARGET_COLUMN] = split_frame[TARGET_COLUMN].to_numpy()
    return frame.reset_index(drop=True)


def log_split_datasets(splits: dict[str, pd.DataFrame], data_path: str) -> dict[str, str]:
    """Log training and validation dataset records with distinct contexts."""
    digests = {}
    for split_name, context, name in [
        ("train", "training", "telco_train"),
        ("valid", "validation", "telco_validation"),
    ]:
        with warnings.catch_warnings():
            # MLflow registers the local-file source class twice and warns that a
            # plain path is "ambiguous" between two identical classes: harmless.
            warnings.filterwarnings("ignore", message="The specified dataset source can be interpreted")
            dataset = mlflow.data.from_pandas(
                dataset_frame(splits[split_name]),
                source=data_path,
                name=name,
                targets=TARGET_COLUMN,
            )
        mlflow.log_input(dataset, context=context)
        digests[context] = dataset.digest
    return digests


def confusion_matrix_frame(metrics: dict) -> pd.DataFrame:
    """Rows = actual class, columns = predicted class, order [0, 1]."""
    cm = metrics["confusion_matrix"]
    return pd.DataFrame(
        cm["matrix"],
        index=["actual_0_no_churn", "actual_1_churn"],
        columns=["predicted_0_no_churn", "predicted_1_churn"],
    )


def save_confusion_matrix_png(metrics: dict, title: str, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion_matrix_frame(metrics).to_numpy()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["0 = No churn", "1 = Churn"])
    ax.set_yticks([0, 1], labels=["0 = No churn", "1 = Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color=color)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def provenance_record(
    *,
    config: dict,
    config_path: str,
    spec: dict,
    code: dict,
    data_sha256: str,
    source_metadata: dict,
    allocation_file: str,
    allocation_sha256: str,
    split_summary: dict,
    dataset_digests: dict,
    environment: dict,
) -> dict:
    return {
        "candidate": spec["name"],
        "estimator": spec["estimator"],
        "configured_params": spec["params"],
        "code": code,
        "config_file": config_path,
        "data": {
            "path": config["data"]["path"],
            "sha256": data_sha256,
            "source_url": source_metadata.get("url"),
            "source_commit": source_metadata.get("commit"),
            "note": "SHA-256 of the original CSV bytes. It identifies the file; it does not store it.",
        },
        "features": {
            "inputs": FEATURES,
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "excluded": [ID_COLUMN, TARGET_COLUMN],
            "parsing": "churnlab.data.parse_features (stateless, applied before the pipeline)",
        },
        "class_mapping": {POSITIVE_LABEL: 1, NEGATIVE_LABEL: 0, "positive_class": 1},
        "split": {
            **{k: v for k, v in config["split"].items() if k != "reference_allocation"},
            "allocation_file": allocation_file,
            "allocation_sha256": allocation_sha256,
            "summary": split_summary,
            "test_set": "reserved: not used for training, comparison or selection",
        },
        "mlflow_dataset_digests": dataset_digests,
        "evaluation": {
            "evaluated_on": "valid",
            "primary_metric": "val_average_precision (continuous predict_proba score of class 1)",
            "threshold": config["threshold"],
            "threshold_rule": "predict churn (1) when score >= threshold",
            "thresholded_metrics": ["val_precision", "val_recall", "val_f1"],
            "confusion_matrix": "rows = actual, columns = predicted, label order [0, 1]",
            "selection_rule": config["selection"]["rule"],
            "selection_tolerance": config["selection"]["tolerance"],
        },
        "environment": environment,
    }
