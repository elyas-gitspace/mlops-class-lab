"""TP03 inference only: load the registered pipeline through its alias and score customers.

    uv run python scripts/predict_registry.py

No fit and no training import: the pipeline comes from the Model Registry
(models:/telco-churn@candidate by default). The same stateless parsing as TP02
(churnlab.data.parse_features) is applied before the pipeline.

It also re-scores the reference validation batch saved in the source run and
compares with the predictions recorded at training time (agreement check).
"""

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.artifacts
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from churnlab.inference import predict_frame

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-uri", default="models:/telco-churn@candidate")
    parser.add_argument("--tracking-uri", default=None, help="default: MLFLOW_TRACKING_URI or " + DEFAULT_TRACKING_URI)
    parser.add_argument("--input", default="examples/new_customers.csv")
    parser.add_argument("--output", default="reports/tp03/new_customers_predictions.csv")
    parser.add_argument("--check-output", default="reports/tp03/reload_check.json")
    parser.add_argument("--tolerance", type=float, default=1e-9, help="max |score difference| allowed")
    args = parser.parse_args()

    tracking_uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Resolve which version / run the URI points to (alias is a mutable reference)
    match = re.fullmatch(r"models:/([^/@]+)(?:@(.+)|/(\d+))", args.model_uri)
    if not match:
        raise SystemExit("Expected models:/<name>@<alias> or models:/<name>/<version>.")
    name, alias, number = match.groups()
    model_version = (
        client.get_model_version_by_alias(name, alias) if alias else client.get_model_version(name, number)
    )
    run_id = model_version.run_id
    threshold = float(client.get_run(run_id).data.params["threshold"])
    print(f"Tracking URI: {tracking_uri}")
    print(f"Model URI: {args.model_uri} -> {name} version {model_version.version} "
          f"(source run {run_id}, source {model_version.source})")

    pipeline = mlflow.sklearn.load_model(args.model_uri)
    print("Loaded:", type(pipeline).__name__, "steps:", list(pipeline.named_steps))

    # Agreement check against the predictions recorded in the source run
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="reference", dst_path=tmp))
        reference_batch = pd.read_csv(local / "reference_batch.csv", dtype=str, keep_default_na=False)
        recorded = pd.read_csv(local / "reference_predictions.csv", dtype={"customerID": str})
    reloaded = predict_frame(pipeline, reference_batch, threshold)
    max_diff = float(np.max(np.abs(reloaded["churn_score"].to_numpy() - recorded["churn_score"].to_numpy())))
    classes_agree = bool((reloaded["churn_pred"].to_numpy() == recorded["churn_pred"].to_numpy()).all())
    ids_agree = bool((reloaded["customerID"].to_numpy() == recorded["customerID"].to_numpy()).all())
    check = {
        "model_uri": args.model_uri,
        "registered_model": name,
        "registered_version": str(model_version.version),
        "alias": alias,
        "source_run_id": run_id,
        "logged_model_source": model_version.source,
        "threshold": threshold,
        "records_compared": int(len(reloaded)),
        "same_customer_ids": ids_agree,
        "class_predictions_agree": classes_agree,
        "max_abs_score_difference": max_diff,
        "score_tolerance": args.tolerance,
        "scores_within_tolerance": max_diff <= args.tolerance,
        "fit_called": False,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(args.check_output, check)
    print(json.dumps(check, indent=2))
    if not (ids_agree and classes_agree and max_diff <= args.tolerance):
        raise SystemExit("Reloaded pipeline does not agree with the recorded predictions.")

    # Score new customers (same input rules as TP02)
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    predictions = predict_frame(pipeline, frame, threshold)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(predictions.to_string(index=False))
    print(f"Scored {len(predictions)} records -> {args.output}")


if __name__ == "__main__":
    main()
