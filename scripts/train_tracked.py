"""TP03 tracked training: one MLflow run per candidate, then register the selected pipeline.

    uv run python scripts/train_tracked.py --config configs/tp03_train.json

Requires a running tracking server (see README, "TP03"):

    uv run mlflow server --host 127.0.0.1 --port 5000 \
        --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts

Same data, features, customer allocation, threshold and selection rule as TP02.
For each candidate (baseline + 3 learned pipelines), one run records:
parameters, validation metrics, Git/data tags, training/validation dataset
inputs, a confusion matrix, a provenance JSON and the fitted pipeline
(mlflow.sklearn.log_model). The selected pipeline is then registered as
`telco-churn` and the alias `candidate` points to the new version.
The test split is never used.
"""

import argparse
import json
import tempfile
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from churnlab.data import file_sha256
from churnlab.evaluate import evaluate
from churnlab.inference import predict_frame
from churnlab.schema import FEATURES, ID_COLUMN, TARGET_COLUMN
from churnlab.split import allocation_table
from churnlab.tracking import (
    confusion_matrix_frame,
    log_split_datasets,
    provenance_record,
    resolve_tracking_uri,
    run_params,
    save_confusion_matrix_png,
    setup_tracking,
    validation_metrics,
)
from churnlab.training import (
    code_identity,
    environment_identity,
    fit_candidate,
    load_config,
    prepare_splits,
    select_candidate,
    write_json,
    xy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/tp03_train.json")
    parser.add_argument("--tracking-uri", default=None, help="overrides MLFLOW_TRACKING_URI and the config")
    parser.add_argument("--no-register", action="store_true", help="log runs only, skip the Model Registry")
    args = parser.parse_args()

    config = load_config(args.config)
    ml_cfg = config["mlflow"]
    seed, threshold = config["split"]["seed"], config["threshold"]
    artifact_dir = Path(config["output"]["artifact_dir"])
    report_dir = Path(config["output"]["report_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = resolve_tracking_uri(args.tracking_uri, ml_cfg["tracking_uri"])
    experiment = setup_tracking(tracking_uri, ml_cfg["experiment"])
    print(f"Tracking URI: {tracking_uri} | experiment: {experiment.name} (id {experiment.experiment_id})")

    # 1. Same data and customer allocation as TP02
    raw, splits = prepare_splits(config)
    data_sha256 = file_sha256(config["data"]["path"])
    allocation_path = artifact_dir / "split_allocation.csv"
    allocation_table(splits).to_csv(allocation_path, index=False)
    allocation_sha256 = file_sha256(allocation_path)
    reference_allocation = Path(config["split"]["reference_allocation"])
    allocation_matches_tp02 = (
        reference_allocation.exists() and file_sha256(reference_allocation) == allocation_sha256
    )
    if reference_allocation.exists() and not allocation_matches_tp02:
        raise SystemExit(
            f"Customer allocation differs from {reference_allocation}: TP03 must reuse the TP02 split."
        )
    split_summary = {
        name: {
            "rows": int(len(frame)),
            "churn_yes": int((frame[TARGET_COLUMN] == "Yes").sum()),
            "churn_rate": round(float((frame[TARGET_COLUMN] == "Yes").mean()), 4),
        }
        for name, frame in splits.items()
    }
    print("Split:", json.dumps(split_summary), "| matches TP02 allocation:", allocation_matches_tp02)

    source_metadata_path = Path(config["data"]["source_metadata"])
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8")) if source_metadata_path.exists() else {}
    )
    code = code_identity(args.config)
    code["entry_point"] = "uv run python scripts/train_tracked.py --config " + Path(args.config).as_posix()
    environment = {**environment_identity(), "mlflow": version("mlflow")}
    if code["code_dirty"]:
        print("WARNING: uncommitted changes in src/, scripts/, configs/ or the lock files. "
              "The git_commit tag alone does not describe the code that ran (tag code_dirty=true).")

    X_train, y_train = xy(splits["train"])
    X_valid, y_valid = xy(splits["valid"])
    reference_batch = splits["valid"][[ID_COLUMN, *FEATURES]].head(config["output"]["reference_batch_size"])

    specs = [{**config["baseline"], "complexity_rank": 0, "learned": False}] + [
        {**spec, "learned": True} for spec in config["candidates"]
    ]

    # 2. One run per candidate
    rows = []
    for spec in specs:
        with mlflow.start_run(run_name=spec["name"]) as run:
            run_id = run.info.run_id
            pipeline, seconds = fit_candidate(spec, seed, X_train, y_train)  # training split only
            classifier = pipeline.named_steps["classifier"]
            metrics = evaluate(pipeline, X_valid, y_valid, threshold)        # validation split only

            mlflow.log_params(run_params(spec, classifier, seed, threshold))
            mlflow.set_tags({
                "git_commit": code["git_commit"],
                "git_branch": code["git_branch"],
                "code_dirty": str(code["code_dirty"]).lower(),
                "data_sha256": data_sha256,
                "split_allocation_sha256": allocation_sha256,
                "allocation_matches_tp02": str(allocation_matches_tp02).lower(),
                "candidate": spec["name"],
                "learned": str(spec["learned"]).lower(),
                "tp": "tp03",
                "mlflow.note.content": (
                    f"{spec['name']} ({spec['estimator']}) fitted on the TP02 training split, "
                    f"evaluated on validation. Primary metric: val_average_precision. Threshold {threshold}."
                ),
            })
            digests = log_split_datasets(splits, config["data"]["path"])

            mlflow.log_metrics({**validation_metrics(metrics), "fit_seconds": round(seconds, 4)})

            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                cm_png = save_confusion_matrix_png(
                    metrics, f"{spec['name']} - validation (threshold {threshold})",
                    tmp / "validation_confusion_matrix.png",
                )
                cm_csv = tmp / "validation_confusion_matrix.csv"
                confusion_matrix_frame(metrics).to_csv(cm_csv)
                mlflow.log_artifact(str(cm_png), artifact_path="evaluation")
                mlflow.log_artifact(str(cm_csv), artifact_path="evaluation")

                ref_batch_path = tmp / "reference_batch.csv"
                ref_pred_path = tmp / "reference_predictions.csv"
                reference_batch.to_csv(ref_batch_path, index=False)
                predict_frame(pipeline, reference_batch, threshold).to_csv(
                    ref_pred_path, index=False, float_format="%.17g"
                )
                mlflow.log_artifact(str(ref_batch_path), artifact_path="reference")
                mlflow.log_artifact(str(ref_pred_path), artifact_path="reference")

            mlflow.log_artifact(str(allocation_path), artifact_path="split")
            mlflow.log_dict(metrics, "evaluation/validation_metrics.json")
            mlflow.log_dict(
                provenance_record(
                    config=config, config_path=Path(args.config).as_posix(), spec=spec, code=code,
                    data_sha256=data_sha256, source_metadata=source_metadata,
                    allocation_file=f"split/{allocation_path.name}", allocation_sha256=allocation_sha256,
                    split_summary=split_summary, dataset_digests=digests, environment=environment,
                ),
                "provenance.json",
            )

            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name=ml_cfg["model_name"],
                input_example=X_train.head(3),
                serialization_format="cloudpickle",
            )

        rows.append({
            "model": spec["name"],
            "estimator": spec["estimator"],
            "learned": spec["learned"],
            "complexity_rank": spec["complexity_rank"],
            "metrics": metrics,
            "fit_seconds": round(seconds, 4),
            "run_id": run_id,
            "model_uri": model_info.model_uri,
        })
        print(f"[{spec['name']}] run_id={run_id} val_AP={metrics['average_precision']:.4f} "
              f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} model_uri={model_info.model_uri}")

    # 3. Selection with the rule fixed before training (validation only)
    selection = select_candidate(rows, config["selection"])
    selected = next(row for row in rows if row["model"] == selection["selected_model"])
    client = MlflowClient()
    for row in rows:
        client.set_tag(row["run_id"], "selected", str(row is selected).lower())

    comparison = [
        {
            "run_id": row["run_id"],
            "candidate": row["model"],
            "estimator": row["estimator"],
            "val_average_precision": round(row["metrics"]["average_precision"], 4),
            "val_precision": round(row["metrics"]["precision"], 4),
            "val_recall": round(row["metrics"]["recall"], 4),
            "val_f1": round(row["metrics"]["f1"], 4),
            "fit_seconds": row["fit_seconds"],
            "decision": (
                "selected" if row is selected
                else "baseline (never selected)" if not row["learned"]
                else "rejected"
            ),
        }
        for row in rows
    ]
    pd.DataFrame(comparison).to_csv(report_dir / "comparison.csv", index=False)
    write_json(report_dir / "selection.json", {**selection, "selected_run_id": selected["run_id"]})
    print(pd.DataFrame(comparison).drop(columns=["run_id"]).to_string(index=False))
    print(f"Selected: {selected['model']} (val_average_precision={selection['selected_value']})")

    # 4. Registry: register the selected logged pipeline and point the alias at it
    handover = {
        "tracking_uri": tracking_uri,
        "experiment": experiment.name,
        "experiment_id": experiment.experiment_id,
        "selected_candidate": selected["model"],
        "source_run_id": selected["run_id"],
        "logged_model_uri": selected["model_uri"],
        "git_commit": code["git_commit"],
        "code_dirty": code["code_dirty"],
        "data_sha256": data_sha256,
        "split_allocation_sha256": allocation_sha256,
        "threshold": threshold,
    }
    if not args.no_register:
        name, alias = ml_cfg["registered_model"], ml_cfg["alias"]
        model_version = mlflow.register_model(selected["model_uri"], name)
        client.set_registered_model_alias(name, alias, model_version.version)
        client.set_model_version_tag(name, model_version.version, "source_run_id", selected["run_id"])
        client.set_model_version_tag(name, model_version.version, "git_commit", code["git_commit"])
        client.set_model_version_tag(name, model_version.version, "data_sha256", data_sha256)
        client.update_model_version(
            name, model_version.version,
            description=(
                f"{selected['model']} pipeline (preprocessing + classifier) selected on validation "
                f"average precision ({selection['selected_value']}). Source run {selected['run_id']}."
            ),
        )
        handover.update({
            "registered_model": name,
            "registered_version": str(model_version.version),
            "alias": alias,
            "alias_uri": f"models:/{name}@{alias}",
            "version_uri": f"models:/{name}/{model_version.version}",
        })
        print(f"Registered {name} version {model_version.version}; alias '{alias}' -> version {model_version.version}")

    write_json(report_dir / "handover.json", handover)
    print(json.dumps(handover, indent=2))


if __name__ == "__main__":
    main()
