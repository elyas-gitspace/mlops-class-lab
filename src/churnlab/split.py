"""Reproducible, stratified train/validation/test split by customer."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from churnlab.schema import ID_COLUMN, TARGET_COLUMN

SPLIT_NAMES = ("train", "valid", "test")


def split_customers(
    df: pd.DataFrame,
    seed: int,
    valid_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> dict[str, pd.DataFrame]:
    """Split ~60/20/20, stratified on Churn, with a single seed."""
    holdout_fraction = valid_fraction + test_fraction
    train, holdout = train_test_split(
        df, test_size=holdout_fraction, stratify=df[TARGET_COLUMN], random_state=seed
    )
    valid, test = train_test_split(
        holdout,
        test_size=test_fraction / holdout_fraction,
        stratify=holdout[TARGET_COLUMN],
        random_state=seed,
    )
    splits = {"train": train, "valid": valid, "test": test}
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    ids = {name: set(frame[ID_COLUMN]) for name, frame in splits.items()}
    for name, frame in splits.items():
        if frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"Duplicate customer IDs inside split {name!r}.")
    names = list(ids)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = ids[left] & ids[right]
            if overlap:
                raise ValueError(
                    f"{len(overlap)} customer IDs appear in both {left!r} and {right!r}."
                )


def allocation_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per customer: customerID -> split name."""
    parts = [
        pd.DataFrame({ID_COLUMN: frame[ID_COLUMN].to_numpy(), "split": name})
        for name, frame in splits.items()
    ]
    return pd.concat(parts).sort_values(ID_COLUMN).reset_index(drop=True)


def splits_from_allocation(
    df: pd.DataFrame, allocation: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Rebuild the exact same splits from a saved allocation table."""
    split_of = dict(zip(allocation[ID_COLUMN], allocation["split"]))
    if set(split_of) != set(df[ID_COLUMN]):
        raise ValueError("Allocation table does not cover exactly the dataset customers.")
    labels = df[ID_COLUMN].map(split_of)
    return {name: df[labels == name] for name in SPLIT_NAMES}
