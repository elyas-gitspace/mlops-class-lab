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
| `scripts/` | Executable workflow commands (download, inspect, record environment) |
| `tests/` | Small behavior checks run with `uv run pytest` |
| `data/raw/` | Original input snapshot (ignored) and its committed source metadata `telco.source.json` |
| `data/processed/` | Generated splits (later TPs, ignored) |
| `artifacts/` | Generated model files (later TPs, ignored) |
| `reports/` | Generated reports (ignored); small reports to submit are copied to `evidence/` |
| `evidence/` | Non-sensitive proof of each TP (environment records, data overview, answers) |
| `k8s/`, `pipelines/` | Deployment and orchestration (later TPs) |

## What is not reproduced yet

The lock file does not pin the OS, CPU architecture or hardware. A hash does not establish label quality or the absence of leakage. No model, training configuration or random seed exists yet: those come with later TPs.
