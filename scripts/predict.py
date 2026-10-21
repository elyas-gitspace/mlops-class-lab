"""Score new customers with the saved pipeline. Inference only: no fit, no labels.

    uv run python scripts/predict.py --input new_customers.csv --output predictions.csv

The input CSV needs the six feature columns (customerID optional, Churn not needed).
"""

import argparse

import pandas as pd

from churnlab.inference import load_pipeline, predict_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-dir", default="artifacts/tp02")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pipeline, metadata = load_pipeline(args.artifact_dir)
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    predictions = predict_frame(pipeline, frame, metadata["task"]["decision_threshold"])
    predictions.to_csv(args.output, index=False)
    print(f"Scored {len(predictions)} records with {metadata['model']['name']} -> {args.output}")


if __name__ == "__main__":
    main()
