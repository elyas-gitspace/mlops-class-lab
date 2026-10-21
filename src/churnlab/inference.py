"""Inference only: load a trusted fitted pipeline and score records. No fit here."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from churnlab.data import file_sha256, parse_features
from churnlab.evaluate import apply_threshold, positive_class_scores
from churnlab.schema import ID_COLUMN


def load_metadata(artifact_dir: str | Path) -> dict:
    return json.loads((Path(artifact_dir) / "metadata.json").read_text(encoding="utf-8"))


def load_pipeline(artifact_dir: str | Path):
    """Load pipeline.joblib after checking its SHA-256 against metadata.json.

    joblib/pickle can execute code when loading: only load artifacts you produced
    or trust, with compatible library versions (recorded in metadata.json).
    """
    artifact_dir = Path(artifact_dir)
    metadata = load_metadata(artifact_dir)
    path = artifact_dir / metadata["artifact"]["file"]
    actual = file_sha256(path)
    expected = metadata["artifact"]["sha256"]
    if actual != expected:
        raise ValueError(f"Artifact hash mismatch for {path}: {actual} != {expected}")
    return joblib.load(path), metadata


def predict_frame(pipeline, frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Score raw records (the six documented features, as text or numbers)."""
    features = parse_features(frame)
    scores = positive_class_scores(pipeline, features)
    result = pd.DataFrame(index=frame.index)
    if ID_COLUMN in frame.columns:
        result[ID_COLUMN] = frame[ID_COLUMN]
    result["churn_score"] = scores
    result["churn_pred"] = apply_threshold(scores, threshold)
    return result
