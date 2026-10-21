# churnlab - Telco customer churn

Course project for ESILV *Machine Learning Operations*. Over the course, this repository grows into a churn prediction system that is trained, tracked, packaged, deployed and monitored. TP01 only lays the reproducible foundation: environment, data provenance, an inspection script and a reviewed Git workflow. No model is trained yet.

## Business question

> Which customers of a telecommunications company are likely to leave its service?

- **Task:** binary classification.
- **Target column:** `Churn` (`Yes` / `No`). It is the label to predict and must never be used as a model input.
- **Identifier:** `customerID` identifies a customer and is not a model feature.
- **Baseline inputs (later TPs):** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`.

## Environment setup

Requirements: Git, [uv](https://docs.astral.sh/uv/) (lock produced with uv 0.12.x), and **Python 3.11** (3.12 is also accepted by `pyproject.toml`). uv can download Python 3.11 itself if it is not installed.

```sh
git clone git@github.com:elyas-gitspace/mlops-class-lab.git
cd mlops-class-lab
uv sync --locked
uv run python -c "import churnlab; print(churnlab.__version__)"   # -> 0.1.0
uv run pytest
```

`uv sync --locked` fails if `uv.lock` no longer matches `pyproject.toml`: investigate the mismatch instead of re-locking silently. Do not install the `tensorflow` or `kubeflow` extras for TP01.

| File | Role |
|---|---|
| `pyproject.toml` | What the project needs: Python constraint, direct dependencies and their allowed ranges |
| `uv.lock` | What was resolved: exact versions and hashes of every direct and transitive package |
| `.venv/` | The installed environment on one machine; generated, never committed |

## Data source and verification

| | |
|---|---|
| Dataset | IBM Telco Customer Churn (7,043 rows, 21 columns) |
| Source | [IBM/telco-customer-churn-on-icp4d @ `1fd6fd7`](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv) |
| Local path | `data/raw/telco.csv` (not committed) |
| SHA-256 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |

```sh
uv run python scripts/download_data.py   # downloads the pinned CSV if absent, checks its hash, writes data/raw/telco.source.json
uv run python scripts/inspect_telco.py   # re-checks the hash and writes reports/tp01_data_overview.json
```

Expected inspection results: 7,043 rows, 21 columns, `Churn` = 5,174 `No` / 1,869 `Yes` (about 26.54 % churn), 0 duplicate `customerID`, 11 whitespace-only `TotalCharges`.

The raw CSV is never edited. Opening it in a spreadsheet tool and saving it changes its bytes, so its hash no longer matches and `inspect_telco.py` refuses it. A matching hash proves *which bytes* were used, not that the labels are correct.

## What this dataset is (and is not)

This is a **single customer snapshot**: one row per customer, at an unspecified point in time. It is **not** an event-time log (no dated history of what happened to each customer and when), and **not** an intervention dataset (no randomized retention offer or treatment assignment). It therefore supports predicting churn from observed attributes. It does not support causal claims such as "offering a discount reduces churn". Do not invent an event history or a treatment and present it as IBM data.

## Repository layout

| Path | Purpose |
|---|---|
| `src/churnlab/` | Importable project code (data, training, serving) |
| `scripts/` | Executable workflow commands (download, inspect, record environment, train, predict, checks) |
| `configs/` | Training configuration (split seed, candidates, selection rule) |
| `examples/` | Small fictional inference input without labels |
| `tests/` | Small behavior checks run with `uv run pytest` |
| `data/raw/` | Original input snapshot (ignored) and its committed source metadata `telco.source.json` |
| `data/processed/` | Generated splits (later TPs, ignored) |
| `artifacts/` | Generated model files and their metadata (ignored) |
| `reports/` | Generated reports (ignored); small reports to submit are copied to `evidence/` |
| `evidence/` | Non-sensitive proof of each TP (environment records, data overview, answers) |
| `k8s/`, `pipelines/` | Deployment and orchestration (later TPs) |

## TP02 - Training pipeline

### Train

```sh
uv sync --locked
uv run python scripts/download_data.py          # if data/raw/telco.csv is absent
uv run python scripts/train.py --config configs/tp02_train.json
```

Everything the run depends on is in `configs/tp02_train.json`: data hash, split seed (42) and fractions (60/20/20, stratified on `Churn`), decision threshold (0.5), the three candidates and their parameters, and the selection rule. Training takes a few seconds.

What it does:

1. checks the CSV hash and splits customers into train / validation / test (the test set is never used in TP02);
2. milestone 1: logistic regression on `tenure` + `MonthlyCharges` vs a majority-class baseline;
3. fits a fresh `Pipeline` (preprocessing + classifier) per candidate on the training split, with the six features, and evaluates on validation;
4. picks a model with the rule fixed in the config (validation average precision, simpler model if within 0.01);
5. saves the fitted pipeline and its record.

Outputs (both folders are git-ignored):

| File | Content |
|---|---|
| `artifacts/tp02/pipeline.joblib` | Fitted preprocessing + classifier, one object |
| `artifacts/tp02/metadata.json` | Data hash, split seed, feature roles, preprocessing policy, model params, metrics, threshold, git commit, package versions, artifact SHA-256 |
| `artifacts/tp02/split_allocation.csv` | `customerID` -> `train` / `valid` / `test` |
| `reports/tp02/comparison.csv` | Baseline + 3 learned models, same split, same metrics |
| `reports/tp02/selection.json` | Selection rule and decision |

A copy of the small files (everything except `pipeline.joblib`) is committed in `evidence/tp02/`.

Submitted artifact: `evidence/tp02/macos/pipeline.joblib` (5.3 MB, SHA-256 `e42ab3f7…`, listed in the `metadata.json` next to it) is the selected random forest trained on macOS arm64 from the same code, config and split. It can be used without retraining:

```sh
uv run python scripts/predict.py --artifact-dir evidence/tp02/macos --input examples/new_customers.csv --output predictions.csv
```

`predict.py` refuses the file if its SHA-256 does not match its `metadata.json`. The Linux run's artifact (`289cd06d…`, described by `evidence/tp02/metadata.json`) is attached to the GitHub release [`tp02-model-v1`](https://github.com/elyas-gitspace/mlops-class-lab/releases/tag/tp02-model-v1). To use it, download it into `artifacts/tp02/` next to a copy of `evidence/tp02/metadata.json`, then run `predict.py` (default `--artifact-dir artifacts/tp02`).

### Model inputs

| Feature | Type | Missing value | Other rules |
|---|---|---|---|
| `tenure`, `MonthlyCharges`, `TotalCharges` | number (or numeric text) | blank / None / NaN -> training median | non-numeric text is rejected |
| `Contract`, `InternetService`, `PaymentMethod` | text | blank / None / NaN -> most frequent training category | unseen category -> ignored (all-zero one-hot) |

`customerID` and `Churn` are never inputs. A missing feature column raises an error. Parsing lives in `churnlab.data.parse_features` and is the same at training and inference.

### Predict and check

```sh
uv run python scripts/predict.py --input examples/new_customers.csv --output predictions.csv
uv run python scripts/check_reload.py     # fresh process: reload, predict a saved batch, compare
uv run python scripts/check_pipeline.py   # split, leakage, missing values, unseen category, schema checks
uv run pytest
```

`pipeline.joblib` is a pickle: only load a file you trust, with the package versions listed in `metadata.json`. `predict.py` verifies the artifact SHA-256 before loading it.

## What is not reproduced yet

The lock file does not pin the OS, CPU architecture or hardware. A hash does not establish label quality or the absence of leakage. Training runs are recorded in a JSON file, not in an experiment tracker (next TP).

Re-running `train.py` on the same platform gives identical metrics. Across platforms, tree ensembles can differ slightly because of floating-point differences, even with the same `uv.lock`, seed, config and split. On macOS arm64 the random forest reached a validation average precision of 0.6665 instead of 0.6662 on Linux aarch64, with one validation customer predicted differently; the split, the other models and the selection were identical (`evidence/tp02/macos/`). The fitted `pipeline.joblib` therefore has a different SHA-256 on each platform: always check it against the `metadata.json` that ships with it. The test set has not been evaluated yet, and the predicted probabilities are not calibrated.
