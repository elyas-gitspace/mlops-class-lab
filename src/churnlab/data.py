"""Loading, hash verification and explicit, stateless parsing of Telco records.

Parsing rules (identical at training and inference time):
- numeric features: surrounding whitespace is stripped; empty / whitespace-only
  text, None and NaN become missing (NaN). Any other text that is not a number
  raises a ValueError: it is never silently turned into a missing value.
- categorical features: surrounding whitespace is stripped; empty text, None
  and NaN become missing (NaN). Other values are kept as strings.
- a missing required feature column raises a ValueError.
- extra columns (customerID, Churn, ...) are ignored by parse_features.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from churnlab.schema import (
    CATEGORICAL_FEATURES,
    FEATURES,
    ID_COLUMN,
    NEGATIVE_LABEL,
    NUMERIC_FEATURES,
    POSITIVE_LABEL,
    TARGET_COLUMN,
)

RAW_DATA_PATH = Path("data/raw/telco.csv")
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw(
    path: str | Path = RAW_DATA_PATH,
    expected_sha256: str | None = EXPECTED_SHA256,
) -> pd.DataFrame:
    """Read the raw CSV as text after checking its SHA-256. The file is never modified."""
    path = Path(path)
    actual = file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(
            f"Source hash mismatch for {path}: got {actual}, expected {expected_sha256}."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = [ID_COLUMN, TARGET_COLUMN, *FEATURES]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError(f"Duplicate {ID_COLUMN} values in {path}.")
    return df


def _normalize_missing(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        return np.nan if value == "" else value
    return value


def parse_numeric(series: pd.Series, name: str) -> pd.Series:
    normalized = series.map(_normalize_missing)
    parsed = pd.to_numeric(normalized, errors="coerce")
    malformed = parsed.isna() & normalized.notna()
    if malformed.any():
        examples = list(pd.unique(normalized[malformed]))[:5]
        raise ValueError(f"Column {name!r} contains non-numeric values: {examples!r}")
    return parsed.astype("float64")


def parse_categorical(series: pd.Series) -> pd.Series:
    def normalize(value):
        value = _normalize_missing(value)
        return value if pd.isna(value) else str(value)

    return series.map(normalize).astype(object)


def parse_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the six model inputs with documented types (float64 / object)."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("parse_features expects a pandas DataFrame.")
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    parsed = pd.DataFrame(index=frame.index)
    for column in NUMERIC_FEATURES:
        parsed[column] = parse_numeric(frame[column], column)
    for column in CATEGORICAL_FEATURES:
        parsed[column] = parse_categorical(frame[column])
    return parsed[FEATURES]


def parse_target(series: pd.Series) -> pd.Series:
    """Encode Churn: 'Yes' -> 1 (positive class), 'No' -> 0. Anything else fails."""
    mapping = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}
    stripped = series.astype(str).str.strip()
    unexpected = sorted(set(stripped) - set(mapping))
    if unexpected:
        raise ValueError(f"Unexpected {TARGET_COLUMN} values: {unexpected}")
    return stripped.map(mapping).astype(int)
