"""Build reports/tp03_summary.md from the recorded TP03 evidence and the live tracking server.

    uv run python scripts/make_tp03_summary.py

Run it last (after the restart check): it reads reports/tp03/*.json|csv,
queries the server to confirm runs, version and alias still exist, copies the
small evidence files to evidence/tp03/ and writes reports/tp03_summary.md.
"""

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

REPORT_DIR = Path("reports/tp03")
EVIDENCE_DIR = Path("evidence/tp03")
SCREENSHOTS = {
    "comparison": "evidence/tp03/screenshots/01_compare_runs.png",
    "run": "evidence/tp03/screenshots/02_selected_run.png",
    "registry": "evidence/tp03/screenshots/03_registry_alias.png",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--output", default="reports/tp03_summary.md")
    args = parser.parse_args()

    handover = json.loads((REPORT_DIR / "handover.json").read_text())
    selection = json.loads((REPORT_DIR / "selection.json").read_text())
    check = json.loads((REPORT_DIR / "reload_check.json").read_text())
    comparison = pd.read_csv(REPORT_DIR / "comparison.csv")

    tracking_uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or handover["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    runs = client.search_runs([handover["experiment_id"]], max_results=1000)
    alias_version = client.get_model_version_by_alias(handover["registered_model"], handover["alias"])
    run_ids_present = {run.info.run_id for run in runs}
    all_candidate_runs_present = set(comparison["run_id"]) <= run_ids_present

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / "screenshots").mkdir(exist_ok=True)
    for name in ["handover.json", "selection.json", "comparison.csv", "reload_check.json",
                 "new_customers_predictions.csv"]:
        if (REPORT_DIR / name).exists():
            shutil.copy2(REPORT_DIR / name, EVIDENCE_DIR / name)

    table = ["| Run ID | Candidate / changed setting | Validation AP | Validation recall | Validation F1 | Decision |",
             "|---|---|---:|---:|---:|---|"]
    for row in comparison.itertuples(index=False):
        table.append(f"| `{row.run_id}` | {row.candidate} ({row.estimator}) | {row.val_average_precision:.4f} "
                     f"| {row.val_recall:.4f} | {row.val_f1:.4f} | {row.decision} |")
    ranked = comparison[comparison["decision"] != "baseline (never selected)"].sort_values(
        "val_average_precision", ascending=False)
    best, second = ranked.iloc[0], ranked.iloc[1]
    gap = best.val_average_precision - second.val_average_precision
    best_recall = comparison.loc[comparison["val_recall"].idxmax()]

    v = handover["registered_version"]
    md = f"""# TP03 summary - MLflow tracking, registry and recovery

Team: Elyas MEZIANI, Michael LI, Amin RACHID. Generated on {datetime.now(timezone.utc).isoformat(timespec="seconds")}.

## 1. Candidate comparison and selection table

![Comparison of the candidate runs](../{SCREENSHOTS['comparison']})

Experiment `{handover['experiment']}` (id {handover['experiment_id']}). Same six features, same TP02 customer allocation
(allocation SHA-256 `{handover['split_allocation_sha256']}`), same threshold ({handover['threshold']}), validation split only.

{chr(10).join(table)}

**Selection.** The primary metric was fixed in the TP02 config before training: validation average precision (AP),
computed from the continuous class-1 probability. The rule picks the best learned candidate, unless a simpler one is within
{selection['tolerance']} AP. `{best.candidate}` has the best AP ({best.val_average_precision:.4f}), {gap:.4f} above
`{second.candidate}` ({second.val_average_precision:.4f}); the gap is larger than the tolerance, so the simpler model is not
preferred and `{selection['selected_model']}` is selected. `{best_recall.candidate}` has the highest recall at threshold
{handover['threshold']} ({best_recall.val_recall:.4f}), but recall and F1 depend on the threshold, which was not tuned, while AP
summarises the ranking quality over all thresholds. The baseline only confirms the learned models beat the 26.5 % churn
prevalence. This is one split, so it is not proof of a statistically significant improvement; the test set is still unused.

## 2. Selected run

![Selected run: parameters, metrics and dataset inputs](../{SCREENSHOTS['run']})

## 3. Registry

![Registry: telco-churn, selected version and candidate alias](../{SCREENSHOTS['registry']})

## 4. Identifiers

| Identifier | Value | Question it answers |
|---|---|---|
| Source run ID | `{handover['source_run_id']}` | Which execution produced the model? |
| Logged model URI | `{handover['logged_model_uri']}` | Which recorded model artifact to load? |
| Registered name + version | `{handover['registered_model']}` version `{v}` (`models:/{handover['registered_model']}/{v}`) | Which version of the managed model? |
| Alias | `{handover['alias']}` -> version `{alias_version.version}` (`models:/{handover['registered_model']}@{handover['alias']}`) | Which version currently holds this role? |
| Git commit | `{handover['git_commit']}` (uncommitted code changes: {str(handover['code_dirty']).lower()}) | Which code? |
| Original data SHA-256 | `{handover['data_sha256']}` | Which CSV bytes? |

## 5. Fresh-process inference and agreement

```sh
uv run python scripts/predict_registry.py
```

Separate Python process, no `fit`, no training import. It loads `{check['model_uri']}` (version {check['registered_version']},
source run `{check['source_run_id']}`), re-scores the {check['records_compared']} validation reference records stored in the source
run, and compares them with the predictions recorded at training time:

- same customer IDs: {check['same_customer_ids']}
- class predictions agree: {check['class_predictions_agree']}
- max absolute score difference: {check['max_abs_score_difference']:.3g} (tolerance {check['score_tolerance']})

It then scores `examples/new_customers.csv` with the TP02 parsing rules (`evidence/tp03/new_customers_predictions.csv`).

## 6. Restart observation and storage

The server was stopped (Ctrl+C) and restarted from the project root with the same command. After the restart, the
server still lists {len(runs)} runs in `{handover['experiment']}` (all {len(comparison)} candidate runs present: {all_candidate_runs_present}),
`{handover['registered_model']}` version {v}, and the alias `{handover['alias']}` still resolves to version {alias_version.version}.
The inference command above was run again after the restart (check timestamp {check['checked_at']}) and agreed again.
Stopping the server stops the API and the UI only; the data lives in files.

**What do the database and the artifact store each preserve?**
`mlflow.db` (SQLite backend store) keeps the metadata: experiments, runs, parameters, metrics, tags, dataset input records
(name, digest, schema, context), logged-model records and the Model Registry (names, versions, aliases, tags). `mlartifacts/`
(artifact store) keeps the files: the pickled pipeline with its `MLmodel`, signature, input example and requirements, plus the
confusion matrices, `provenance.json`, the split allocation and the reference predictions. If only the database survived, every
record and ID would still be listed but loading the model (and opening any artifact) would fail, because the files the records point
to are gone. If only the model files survived, the pickle could perhaps be loaded by path, but nothing would say which run, parameters,
metrics, data hash, commit, version or alias it belongs to: no traceability and `models:/telco-churn@candidate` would not resolve.

**What must a colleague on another machine obtain separately?**
A run ID or a `models:/` URI only means something on this tracking server, which listens on 127.0.0.1. A colleague needs access to the
same server (a shared team server) or a copy of both `mlflow.db` and `mlartifacts/`; the repository at commit
`{handover['git_commit'][:7]}` with `uv.lock` to rebuild a compatible environment (cloudpickle needs compatible Python, scikit-learn
and churnlab versions); and the original CSV itself (downloaded with `scripts/download_data.py` and checked against its SHA-256), since
neither the hash nor the MLflow dataset record stores the data. Here none of this is shared yet: this limitation is documented, not solved.
"""
    Path(args.output).write_text(md, encoding="utf-8")
    missing = [path for path in SCREENSHOTS.values() if not Path(path).exists()]
    print(f"Wrote {args.output} and copied evidence to {EVIDENCE_DIR}/")
    if missing:
        print("Screenshots still missing (add them, then re-run this script):", *missing, sep="\n  ")


if __name__ == "__main__":
    main()
