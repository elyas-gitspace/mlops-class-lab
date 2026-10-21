"""Focused reliability checks on the saved pipeline (TP02 part 5).

    uv run python scripts/check_pipeline.py

Writes reports/tp02/reliability_checks.json and exits non-zero if a check fails.
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from churnlab.data import load_raw, parse_features
from churnlab.evaluate import positive_class_scores
from churnlab.inference import load_pipeline
from churnlab.schema import CATEGORICAL_FEATURES, ID_COLUMN, NUMERIC_FEATURES
from churnlab.split import SPLIT_NAMES, splits_from_allocation


def record(checks, name, passed, **evidence):
    checks.append({"check": name, "passed": bool(passed), **evidence})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")


def expect_error(func):
    try:
        func()
    except Exception as error:  # noqa: BLE001 - we want to show which error surfaces
        return {"raised": True, "error_type": type(error).__name__, "message": str(error)[:300]}
    return {"raised": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-dir", default="artifacts/tp02")
    parser.add_argument("--output", default="reports/tp02/reliability_checks.json")
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir)

    pipeline, metadata = load_pipeline(artifact_dir)
    raw = load_raw(metadata["data"]["path"], metadata["data"]["sha256"])
    allocation = pd.read_csv(artifact_dir / metadata["split"]["allocation_file"])
    splits = splits_from_allocation(raw, allocation)
    X_train = parse_features(splits["train"])
    X_valid = parse_features(splits["valid"])
    X_full = parse_features(raw)
    checks = []

    # 1. Disjoint customer IDs
    ids = {name: set(splits[name][ID_COLUMN]) for name in SPLIT_NAMES}
    overlaps = {
        f"{a}&{b}": len(ids[a] & ids[b])
        for i, a in enumerate(SPLIT_NAMES) for b in SPLIT_NAMES[i + 1:]
    }
    covered = sum(len(v) for v in ids.values())
    record(checks, "customer IDs do not overlap across train/valid/test",
           all(v == 0 for v in overlaps.values()) and covered == len(raw),
           sizes={k: len(v) for k, v in ids.items()}, overlaps=overlaps,
           total_customers=len(raw))

    # 2. Learned statistics come from training data only
    num = pipeline.named_steps["preprocessor"].named_transformers_["num"]
    learned_medians = num.named_steps["imputer"].statistics_
    train_medians = X_train[NUMERIC_FEATURES].median().to_numpy()
    full_medians = X_full[NUMERIC_FEATURES].median().to_numpy()
    record(checks, "imputer medians equal training-only medians",
           np.allclose(learned_medians, train_medians),
           features=NUMERIC_FEATURES,
           learned=learned_medians.tolist(),
           train_only=train_medians.tolist(),
           full_dataset=full_medians.tolist(),
           differs_from_full_dataset=[bool(not np.isclose(a, b)) for a, b in zip(learned_medians, full_medians)])
    train_imputed = X_train[NUMERIC_FEATURES].fillna(dict(zip(NUMERIC_FEATURES, train_medians)))
    learned_means = num.named_steps["scaler"].mean_
    record(checks, "scaler means equal training-only means (after median imputation)",
           np.allclose(learned_means, train_imputed.mean().to_numpy()),
           learned=learned_means.tolist(),
           train_only=train_imputed.mean().tolist(),
           full_dataset=X_full[NUMERIC_FEATURES].mean().tolist())

    # 3. Predicting does not refit or change fitted state
    before_stats = learned_medians.copy()
    before_bytes = pickle.dumps(pipeline)
    positive_class_scores(pipeline, X_valid)
    pipeline.predict(X_valid)
    record(checks, "prediction does not change fitted state",
           np.array_equal(before_stats, num.named_steps["imputer"].statistics_)
           and before_bytes == pickle.dumps(pipeline),
           statistics_before=before_stats.tolist(),
           statistics_after=num.named_steps["imputer"].statistics_.tolist(),
           pickled_pipeline_identical=before_bytes == pickle.dumps(pipeline))

    # 4a. Permitted missing numeric input -> training median
    row = X_valid.iloc[[0]].copy()
    for missing_value in (None, " ", np.nan):
        raw_row = splits["valid"].iloc[[0]].copy()
        raw_row["TotalCharges"] = missing_value
        with_missing = parse_features(raw_row)
        filled = row.copy()
        filled["TotalCharges"] = learned_medians[NUMERIC_FEATURES.index("TotalCharges")]
        score_missing = positive_class_scores(pipeline, with_missing)[0]
        score_filled = positive_class_scores(pipeline, filled)[0]
        record(checks, f"missing TotalCharges ({missing_value!r}) is imputed with the training median",
               np.isclose(score_missing, score_filled),
               score_with_missing=float(score_missing), score_with_train_median=float(score_filled))

    # 4b. Missing categorical -> most frequent training category
    cat = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    mode_contract = cat.named_steps["imputer"].statistics_[CATEGORICAL_FEATURES.index("Contract")]
    missing_cat, mode_cat = row.copy(), row.copy()
    missing_cat["Contract"] = np.nan
    mode_cat["Contract"] = mode_contract
    record(checks, "missing Contract is imputed with the most frequent training category",
           np.isclose(positive_class_scores(pipeline, missing_cat)[0], positive_class_scores(pipeline, mode_cat)[0]),
           most_frequent_training_category=str(mode_contract))

    # 4c. Unseen category -> ignored (all-zero one-hot), no crash
    unseen = row.copy()
    unseen["Contract"] = "Three year"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        encoded = pipeline.named_steps["preprocessor"].transform(unseen)
        score_unseen = positive_class_scores(pipeline, unseen)[0]
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    contract_cols = [i for i, n in enumerate(names) if n.startswith("cat__Contract_")]
    record(checks, "unseen category 'Three year' is encoded as all zeros (handle_unknown='ignore')",
           float(np.abs(encoded[0, contract_cols]).sum()) == 0.0,
           contract_columns=[names[i] for i in contract_cols],
           encoded_contract_values=encoded[0, contract_cols].tolist(),
           score=float(score_unseen),
           warnings=[str(w.message) for w in caught],
           note="Handled without error, but the score is not reliable for a category never seen in training.")

    # 5. Missing required feature and malformed input fail visibly
    missing_column = splits["valid"].iloc[[0]].drop(columns=["Contract"])
    parse_error = expect_error(lambda: parse_features(missing_column))
    pipeline_error = expect_error(lambda: pipeline.predict(row.drop(columns=["Contract"])))
    record(checks, "missing required column 'Contract' raises an error (parser and pipeline)",
           parse_error["raised"] and pipeline_error["raised"],
           parse_features=parse_error, pipeline_predict=pipeline_error)
    malformed = splits["valid"].iloc[[0]].copy()
    malformed["TotalCharges"] = "abc"
    malformed_error = expect_error(lambda: parse_features(malformed))
    record(checks, "malformed numeric text 'abc' raises instead of becoming missing",
           malformed_error["raised"], parse_features=malformed_error)

    # 6. Reload is checked in a separate process: scripts/check_reload.py
    result = {
        "model": metadata["model"]["name"],
        "artifact_sha256": metadata["artifact"]["sha256"],
        "all_passed": all(c["passed"] for c in checks),
        "checks": checks,
        "reload_check": "see reload_check.json (scripts/check_reload.py, separate process)",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    if not result["all_passed"]:
        raise SystemExit("Some reliability checks FAILED.")
    print("All reliability checks passed.")


if __name__ == "__main__":
    main()
