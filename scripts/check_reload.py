"""Fresh-process reload check: load the artifact, predict a saved batch, compare.

    uv run python scripts/check_reload.py

Runs in its own Python process. It never imports the training code, never reads
labels and never calls fit.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from churnlab.inference import load_pipeline, predict_frame

TOLERANCE = 1e-12  # absolute tolerance on probability scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-dir", default="artifacts/tp02")
    parser.add_argument("--output", default="reports/tp02/reload_check.json")
    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir)

    pipeline, metadata = load_pipeline(artifact_dir)  # also verifies the artifact hash
    batch = pd.read_csv(artifact_dir / "reference_batch.csv", dtype=str, keep_default_na=False)
    reference = pd.read_csv(artifact_dir / "reference_predictions.csv")
    if "Churn" in batch.columns:
        raise ValueError("The inference batch must not contain the Churn label.")

    reloaded = predict_frame(pipeline, batch, metadata["task"]["decision_threshold"])
    max_abs_diff = float(np.max(np.abs(reloaded["churn_score"].to_numpy() - reference["churn_score"].to_numpy())))
    same_ids = bool((reloaded["customerID"].to_numpy() == reference["customerID"].to_numpy()).all())
    same_classes = bool((reloaded["churn_pred"].to_numpy() == reference["churn_pred"].to_numpy()).all())
    result = {
        "artifact_sha256_verified": True,
        "model": metadata["model"]["name"],
        "records": int(len(batch)),
        "input_columns": list(batch.columns),
        "label_column_present": False,
        "fit_called": False,
        "score_tolerance": TOLERANCE,
        "max_abs_score_difference": max_abs_diff,
        "same_customer_order": same_ids,
        "same_class_predictions": same_classes,
        "passed": same_ids and same_classes and max_abs_diff <= TOLERANCE,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Reload check FAILED.")


if __name__ == "__main__":
    main()
