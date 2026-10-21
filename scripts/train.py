"""TP02 training entry point.

    uv run python scripts/train.py --config configs/tp02_train.json

1. verify and load the raw CSV, split customers 60/20/20 (stratified, one seed)
2. milestone 1: two-feature logistic regression vs majority baseline
3. compare baseline + three learned pipelines on validation (six features)
4. select with the rule fixed in the config (validation only, test untouched)
5. save the fitted pipeline (preprocessing + classifier) and its metadata
"""

import argparse
import json
from pathlib import Path

import joblib

from churnlab.data import file_sha256
from churnlab.inference import predict_frame
from churnlab.schema import (
    CATEGORICAL_FEATURES,
    FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from churnlab.split import allocation_table
from churnlab.training import (
    code_identity,
    compare_candidates,
    comparison_frame,
    environment_identity,
    json_safe_params,
    load_config,
    now_utc,
    prepare_splits,
    run_first_model,
    select_candidate,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/tp02_train.json")
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_dir = Path(config["output"]["artifact_dir"])
    report_dir = Path(config["output"]["report_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    threshold = config["threshold"]

    # 1. Data and split
    raw, splits = prepare_splits(config)
    allocation = allocation_table(splits)
    allocation_path = artifact_dir / "split_allocation.csv"
    allocation.to_csv(allocation_path, index=False)
    split_summary = {
        name: {
            "rows": int(len(frame)),
            "churn_yes": int((frame[TARGET_COLUMN] == "Yes").sum()),
            "churn_rate": round(float((frame[TARGET_COLUMN] == "Yes").mean()), 4),
        }
        for name, frame in splits.items()
    }
    print("Split:", json.dumps(split_summary))

    # 2. Milestone 1
    first = run_first_model(splits, threshold)
    write_json(report_dir / "first_model.json", first)
    for name, metrics in first["results"].items():
        print(f"[first model] {name}: accuracy={metrics['accuracy']:.4f} "
              f"cm={metrics['confusion_matrix']['matrix']}")

    # 3. Fair comparison
    rows, fitted = compare_candidates(config, splits)
    table = comparison_frame(rows)
    table.to_csv(report_dir / "comparison.csv", index=False)
    write_json(report_dir / "comparison.json", rows)
    print(table.drop(columns=["params"]).to_string(index=False))

    # 4. Selection (validation only)
    selection = select_candidate(rows, config["selection"])
    write_json(report_dir / "selection.json", selection)
    selected_name = selection["selected_model"]
    selected = fitted[selected_name]
    selected_row = next(row for row in rows if row["model"] == selected_name)
    print("Selected:", selected_name, f"({selection['metric']}={selection['selected_value']})")

    # 5. Artifact + reference predictions for the reload check
    artifact_path = artifact_dir / "pipeline.joblib"
    joblib.dump(selected, artifact_path)

    reference_batch = splits["valid"][[ID_COLUMN, *FEATURES]].head(
        config["output"]["reference_batch_size"]
    )
    reference_batch.to_csv(artifact_dir / "reference_batch.csv", index=False)
    predict_frame(selected, reference_batch, threshold).to_csv(
        artifact_dir / "reference_predictions.csv", index=False, float_format="%.17g"
    )

    preprocessor = selected.named_steps["preprocessor"]
    source_metadata_path = Path(config["data"]["source_metadata"])
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.exists()
        else {}
    )
    metadata = {
        "created_at": now_utc(),
        "artifact": {
            "file": artifact_path.name,
            "sha256": file_sha256(artifact_path),
            "format": "joblib (pickle) - load trusted files only, with the package versions below",
            "content": "fitted sklearn Pipeline: preprocessor (ColumnTransformer) + classifier",
        },
        "task": {
            "target": TARGET_COLUMN,
            "positive_class": "Churn = 'Yes', encoded 1",
            "negative_class": "Churn = 'No', encoded 0",
            "score": "predict_proba column of class 1",
            "decision_threshold": threshold,
            "decision_rule": "predict churn when score >= threshold",
        },
        "data": {
            "path": config["data"]["path"],
            "sha256": file_sha256(config["data"]["path"]),
            "source_url": source_metadata.get("url"),
            "source_commit": source_metadata.get("commit"),
            "rows": int(len(raw)),
        },
        "split": {
            **config["split"],
            "allocation_file": allocation_path.name,
            "allocation_sha256": file_sha256(allocation_path),
            "summary": split_summary,
            "test_set": "reserved: not used for training, comparison or selection",
        },
        "features": {
            "identifier_not_input": ID_COLUMN,
            "target_not_input": TARGET_COLUMN,
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "input_types": {
                **{name: "number or numeric text; blank/None/NaN = missing" for name in NUMERIC_FEATURES},
                **{name: "text; blank/None/NaN = missing" for name in CATEGORICAL_FEATURES},
            },
            "parsing": "churnlab.data.parse_features (stateless, applied before the pipeline)",
        },
        "preprocessing": {
            "numeric": "SimpleImputer(median) -> StandardScaler, learned on training data",
            "categorical": "SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')",
            "missing_numeric": "replaced by the training median",
            "missing_categorical": "replaced by the most frequent training category",
            "unknown_category": "encoded as all zeros for that feature; no error, no reliability guarantee",
            "malformed_numeric_text": "rejected with ValueError during parsing",
            "missing_feature_column": "rejected with ValueError during parsing",
            "learned_numeric_medians": dict(zip(
                NUMERIC_FEATURES,
                preprocessor.named_transformers_["num"].named_steps["imputer"].statistics_.tolist(),
            )),
            "learned_categories": {
                name: list(categories)
                for name, categories in zip(
                    CATEGORICAL_FEATURES,
                    preprocessor.named_transformers_["cat"].named_steps["encoder"].categories_,
                )
            },
        },
        "model": {
            "name": selected_name,
            "estimator": selected_row["estimator"],
            "params": json_safe_params(selected.named_steps["classifier"]),
            "fit_seconds": selected_row["fit_seconds"],
            "trained_on": "train split only",
        },
        "validation_metrics": selected_row["metrics"],
        "selection": selection,
        "code": code_identity(args.config),
        "environment": environment_identity(),
    }
    write_json(artifact_dir / "metadata.json", metadata)
    print(f"Saved {artifact_path} and {artifact_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
