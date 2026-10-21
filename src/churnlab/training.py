"""Training workflow building blocks: data, candidates, selection, records."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from churnlab.data import file_sha256, load_raw, parse_features, parse_target
from churnlab.evaluate import evaluate
from churnlab.pipeline import build_pipeline, make_estimator
from churnlab.schema import FEATURES, FIRST_MODEL_FEATURES, TARGET_COLUMN
from churnlab.split import split_customers

METRIC_COLUMNS = [
    "average_precision",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "predicted_positive_rate",
]


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_splits(config: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    data_cfg, split_cfg = config["data"], config["split"]
    total = split_cfg["train_fraction"] + split_cfg["valid_fraction"] + split_cfg["test_fraction"]
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1, got {total}.")
    raw = load_raw(data_cfg["path"], data_cfg["sha256"])
    splits = split_customers(
        raw,
        seed=split_cfg["seed"],
        valid_fraction=split_cfg["valid_fraction"],
        test_fraction=split_cfg["test_fraction"],
    )
    return raw, splits


def xy(frame: pd.DataFrame, features: list[str] = FEATURES):
    return parse_features(frame)[features], parse_target(frame[TARGET_COLUMN])


def run_first_model(splits: dict[str, pd.DataFrame], threshold: float) -> dict:
    """Milestone 1: two numeric features, plain logistic regression vs majority baseline."""
    X_train, y_train = xy(splits["train"], FIRST_MODEL_FEATURES)
    X_valid, y_valid = xy(splits["valid"], FIRST_MODEL_FEATURES)
    models = {
        "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression_2_features": LogisticRegression(max_iter=1000),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)  # training data only
        results[name] = evaluate(model, X_valid, y_valid, threshold)
    return {
        "features": FIRST_MODEL_FEATURES,
        "positive_class": "Churn = Yes, encoded 1",
        "evaluated_on": "valid",
        "results": results,
    }


def json_safe_params(estimator) -> dict:
    safe = {}
    for key, value in estimator.get_params(deep=False).items():
        safe[key] = value if isinstance(value, (bool, int, float, str, type(None))) else repr(value)
    return safe


def fit_candidate(spec: dict, seed: int, X_train, y_train):
    """Fresh pipeline for this candidate, fitted once on training data."""
    pipeline = build_pipeline(make_estimator(spec["estimator"], spec["params"], seed))
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    return pipeline, time.perf_counter() - start


def compare_candidates(config: dict, splits: dict[str, pd.DataFrame]):
    """Baseline + learned candidates, same features, same split, same metrics."""
    seed, threshold = config["split"]["seed"], config["threshold"]
    X_train, y_train = xy(splits["train"])
    X_valid, y_valid = xy(splits["valid"])
    specs = [{**config["baseline"], "complexity_rank": 0, "learned": False}] + [
        {**spec, "learned": True} for spec in config["candidates"]
    ]
    rows, fitted = [], {}
    for spec in specs:
        pipeline, seconds = fit_candidate(spec, seed, X_train, y_train)
        metrics = evaluate(pipeline, X_valid, y_valid, threshold)
        fitted[spec["name"]] = pipeline
        rows.append(
            {
                "model": spec["name"],
                "estimator": spec["estimator"],
                "learned": spec["learned"],
                "complexity_rank": spec["complexity_rank"],
                "params": json_safe_params(pipeline.named_steps["classifier"]),
                "fit_seconds": round(seconds, 4),
                "metrics": metrics,
            }
        )
    return rows, fitted


def comparison_frame(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        cm = row["metrics"]["confusion_matrix"]
        records.append(
            {
                "model": row["model"],
                "estimator": row["estimator"],
                "learned": row["learned"],
                **{name: round(row["metrics"][name], 4) for name in METRIC_COLUMNS},
                "tn": cm["tn"],
                "fp": cm["fp"],
                "fn": cm["fn"],
                "tp": cm["tp"],
                "fit_seconds": row["fit_seconds"],
                "params": json.dumps(row["params"], sort_keys=True),
            }
        )
    return pd.DataFrame(records)


def select_candidate(rows: list[dict], selection: dict) -> dict:
    """Apply the selection rule fixed in the config before training."""
    metric, tolerance = selection["metric"], selection["tolerance"]
    learned = [row for row in rows if row["learned"]]
    best = max(learned, key=lambda row: row["metrics"][metric])
    best_value = best["metrics"][metric]
    close = [row for row in learned if best_value - row["metrics"][metric] <= tolerance]
    chosen = min(close, key=lambda row: (row["complexity_rank"], -row["metrics"][metric]))
    return {
        "selected_model": chosen["model"],
        "metric": metric,
        "evaluated_on": selection["evaluated_on"],
        "tolerance": tolerance,
        "rule": selection["rule"],
        "best_model_by_metric": best["model"],
        "best_value": round(best_value, 4),
        "selected_value": round(chosen["metrics"][metric], 4),
        "candidates_within_tolerance": [row["model"] for row in close],
        "ranking": [
            {"model": row["model"], metric: round(row["metrics"][metric], 4)}
            for row in sorted(rows, key=lambda row: -row["metrics"][metric])
        ],
    }


def code_identity(config_path: str | Path) -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    tracked_code = ["src", "scripts", "configs", "pyproject.toml", "uv.lock"]
    status = git("status", "--porcelain", "--untracked-files=no", "--", *tracked_code)
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "code_dirty": status not in ("", "unknown"),
        "entry_point": "uv run python scripts/train.py --config " + Path(config_path).as_posix(),
        "config_file": Path(config_path).as_posix(),
        "config_sha256": file_sha256(config_path),
    }


def environment_identity() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in ["churnlab", "scikit-learn", "pandas", "numpy", "scipy", "joblib"]
        },
    }


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
