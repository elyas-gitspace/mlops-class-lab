# TP02 · From a working model to a reusable training pipeline

> *First make a model work. Then make the whole prediction process reusable by another engineer.*

|  |  |
|---|---|
| **Student** | Elyas MEZIANI |
| **Worked with** | Michael LI, Amin RACHID |
| **Repo / branch** | [elyas-gitspace/mlops-class-lab](https://github.com/elyas-gitspace/mlops-class-lab) · `feat/tp02-training-pipeline` |
| **Code used for the run** | commit `1779715` (recorded in `metadata.json`, `code_dirty: false`) |
| **Entry point** | `uv run python scripts/train.py --config configs/tp02_train.json` |
| **Selected model** | `random_forest` · artifact SHA-256 `289cd06d2d6a…` |

---

## How the code is organised

I split the work into small modules, each with one job:

| Module | Job |
|---|---|
| `churnlab/schema.py` | Feature names, target, positive class |
| `churnlab/data.py` | Hash check, loading, explicit parsing of features and target |
| `churnlab/split.py` | Stratified 60/20/20 split, disjointness check, allocation table |
| `churnlab/pipeline.py` | Builds an **unfitted** preprocessing + classifier pipeline. It never reads data and never calls `fit` |
| `churnlab/evaluate.py` | Positive-class scores and metrics |
| `churnlab/training.py` | First model, candidate comparison, selection rule, run records |
| `churnlab/inference.py` | Loads the artifact (after a hash check) and predicts. No training code |
| `scripts/train.py` | The terminal entry point that chains everything |

All the choices that drive a run (seed, fractions, threshold, candidates, selection rule) are in `configs/tp02_train.json`, and its SHA-256 is written into the metadata.

---

## 1 · A first model that works

**Task:** binary classification, `Churn = Yes` is the positive class (encoded `1`). `customerID` is only used to identify customers and check the split, never as an input.

**Split:** stratified on `Churn`, seed `42`, done once and saved in `split_allocation.csv`.

| Split | Rows | Churners | Churn rate |
|---|---:|---:|---:|
| train | 4,225 | 1,121 | 26.53 % |
| valid | 1,409 | 374 | 26.54 % |
| test | 1,409 | 374 | 26.54 % (reserved, never used) |

**Results on validation**, features `tenure` + `MonthlyCharges` only (threshold 0.5).

Confusion matrices are written `[[TN, FP], [FN, TP]]`: rows = actual (0 = No, 1 = Yes), columns = predicted.

| Model | Accuracy | Precision | Recall | Confusion matrix |
|---|---:|---:|---:|---|
| Majority baseline (`DummyClassifier`, fitted on train) | 0.7346 | 0.000 | 0.000 | `[[1035, 0], [374, 0]]` |
| Logistic regression | 0.7828 | 0.643 | 0.409 | `[[950, 85], [221, 153]]` |

**What would always predicting "No churn" give?** 73.5 % accuracy, and not a single churner found. That's exactly what the baseline shows.

**Can high accuracy and zero churners go together?** Yes, the baseline is the proof. With 26.5 % positives, accuracy alone hides the cases we care about.

**Which data can influence model selection?** Only the training split (to fit) and the validation split (to compare). The test split stays untouched until the very end.

---

## 2 · Preprocessing + model in one Pipeline

**Features:** `tenure`, `MonthlyCharges`, `TotalCharges` (numeric) and `Contract`, `InternetService`, `PaymentMethod` (categorical).

**Parsing, before the pipeline (stateless, same at training and inference):**
- the CSV is read as text and stays unchanged on disk;
- whitespace-only `TotalCharges` becomes `NaN`. I checked the data: the 11 blank values are all customers with `tenure = 0`, i.e. new customers who haven't been billed yet;
- any other non-numeric text raises a `ValueError`. I compared `pd.to_numeric(errors="raise")` and `errors="coerce"`: `coerce` would silently turn `"abc"` into `NaN`, which the handout forbids. So I blank out the empty values first, coerce, and then fail if anything that wasn't blank became `NaN`.

**Inside the pipeline:**

| Branch | Steps | Learned on train |
|---|---|---|
| numeric | `SimpleImputer(median)` → `StandardScaler` | medians, means, standard deviations |
| categorical | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")` | most frequent category, list of categories |

**Unseen category at inference:** it's encoded as all zeros for that feature, so prediction still works. That doesn't make the prediction reliable for that customer. It just avoids a crash.

**Checkpoint:** one `pipeline.fit(X_train, y_train)` trains the preprocessing and the classifier together. `predict` takes the six original columns. I never refit an encoder or scaler on validation.

**Think about it**

- **`fit` / `transform` / `predict`:** `fit` learns from data (medians, categories, model weights). `transform` applies what was learned. `predict` runs the transforms and then the model, without learning anything.
- **Scaler fitted on validation:** validation information leaks into the model, so the scores are too optimistic, and the validation data would be scaled differently from the training data.
- **Does a Pipeline prevent leakage on its own?** No. If I call `fit` on train + validation, the pipeline will happily learn from both. It only guarantees that the steps are applied in the right order to whatever data I give it.
- **Pipeline vs ColumnTransformer:** `Pipeline` orders the steps (preprocess, then classify). `ColumnTransformer` sends each column to the right branch (numeric or categorical).

---

## 3 · Comparing models fairly

Same six features, same split, same metrics, fresh pipeline for each candidate (`build_pipeline` is called again every time).

**Selection metric, chosen before running the comparison:** validation **average precision** (AP), computed from the class-1 probability. It looks at how well the model ranks churners above non-churners, without depending on a threshold, and unlike accuracy it isn't fooled by the 73/27 imbalance.

**Selection rule (in the config before the run):** take the best AP. If a simpler model is within 0.01 of the best, take the simpler one (logistic regression < decision tree < random forest).

| Model | AP | ROC AUC | Precision | Recall | F1 | Accuracy | TN | FP | FN | TP | Fit (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (most frequent) | 0.2654 | 0.5000 | 0.000 | 0.000 | 0.000 | 0.7346 | 1035 | 0 | 374 | 0 | 0.005 |
| logistic regression (`max_iter=1000`) | 0.6475 | 0.8492 | 0.664 | 0.529 | 0.589 | **0.8041** | 935 | 100 | 176 | 198 | 0.24 |
| decision tree (`max_depth=5`, `min_samples_leaf=20`) | 0.6173 | 0.8377 | 0.636 | **0.602** | **0.618** | 0.8027 | 906 | 129 | 149 | 225 | 0.01 |
| random forest (300 trees, `max_depth=8`, `min_samples_leaf=5`) | **0.6662** | **0.8529** | 0.648 | 0.487 | 0.556 | 0.7935 | 936 | 99 | 192 | 182 | 0.39 |

Precision / recall / F1 / accuracy use a threshold of 0.5 (`score >= 0.5` → churn). Fit times are on my machine, just for information. Full table: `comparison.csv` / `comparison.json`.

**Decision:** `random_forest`. Its AP (0.6662) is 0.0187 above logistic regression, which is more than the 0.01 tolerance, so the simpler model doesn't win.

**Things I want to be honest about:**
- The random forest has the best ranking (AP, ROC AUC) but **not** the best accuracy (logistic regression) or the best recall at 0.5 (decision tree). At threshold 0.5, it misses 192 of the 374 churners. If the team cares most about catching churners, the threshold should be tuned on validation (optional extension 2, not done here).
- I can't say "the random forest beats the first model because it's a better algorithm": the feature set also went from 2 to 6.
- The gap between the candidates is small, and I measured it on a single validation split.

**Think about it**

- **A different split per model?** No: the differences would mix "better model" with "easier split". Same split = differences come from the model.
- **Highest accuracy = what the team needs?** No. Here logistic regression has the best accuracy, but the team wants to find churners, so ranking (AP) and recall matter more.
- **Two close candidates?** The simpler one wins: faster, easier to explain, less risk of overfitting. That's why the rule has a tolerance.
- **Is 0.8 a real 80 % risk?** Not automatically. Random forest probabilities are averaged tree votes and aren't calibrated. Checking that would need a calibration curve.

---

## 4 · Saving the whole fitted pipeline

`artifacts/tp02/pipeline.joblib` contains the **fitted preprocessing and classifier together**. `metadata.json` next to it records:

- data: path, SHA-256, IBM source URL and commit;
- split: seed, fractions, allocation file + its SHA-256, row counts per split, "test reserved";
- features: roles, input types, missing-value and unknown-category policy, learned medians and categories;
- model: name, all parameters, fit time, "trained on train only";
- task: positive class, threshold, decision rule;
- validation metrics and the selection decision;
- code: git commit, `code_dirty: false`, config path + SHA-256, entry point;
- environment: Python 3.11.16, scikit-learn 1.5.2, pandas 2.2.3, numpy 1.26.4, scipy 1.17.1, joblib 1.6.0, platform;
- the artifact's own SHA-256, checked before every load.

**Reload in a fresh process** (`scripts/check_reload.py`, `reload_check.json`): it loads the artifact, scores 25 saved validation customers (no `Churn` column), and compares the result with the scores computed right after training.

> 25 records · max score difference `9.0e-17` (tolerance `1e-12`) · same 25 class predictions · no `fit`, no labels, no training code imported → **passed**

`scripts/predict.py` does the same thing on any CSV with the six columns (e.g. `examples/new_customers.csv`, 4 fictional customers).

**Think about it**

- **Why not save only the classifier?** It expects scaled numbers and one-hot columns. Without the fitted preprocessing, someone would have to rebuild it by hand and could get different medians or a different column order.
- **What's inside a fitted imputer or encoder?** The training medians, the most frequent categories, and the list of categories (which fixes the order of the one-hot columns).
- **Can two `model.joblib` files be different models?** Yes, same as with the CSV in TP01. That's why I store the SHA-256 of the artifact and check it before loading.
- **sklearn Pipeline vs Kubeflow Pipelines:** a sklearn Pipeline is one Python object that chains transformations and a model in memory. A Kubeflow pipeline is a workflow (DAG) of separate steps, each running in its own container on Kubernetes and passing files between them (e.g. "download → train → evaluate → deploy").

---

## 5 · Reliability checks

`scripts/check_pipeline.py` → `reliability_checks.json`. All checks pass.

| Check | Evidence |
|---|---|
| Customer IDs don't overlap across splits | 4,225 / 1,409 / 1,409 IDs, 0 overlap, 7,043 total |
| Imputer medians come from train only | learned `[29.0, 70.55, 1373.625]` = train medians. Full dataset would give `[29.0, 70.35, 1397.475]` |
| Scaler means come from train only | learned = train means after imputation (e.g. `TotalCharges` 2297.77 vs 2283.30 on the full data) |
| Predicting doesn't change fitted state | medians identical before/after, pickled pipeline byte-identical |
| Missing numeric (`None`, `" "`, `NaN`) | same score as with the training median filled in |
| Missing category | same score as with the most frequent training category (`Month-to-month`) |
| Unseen category `"Three year"` | one-hot `Contract` columns all zero, no error. The prediction is not trustworthy, it's just handled |
| Missing column `Contract` | `ValueError` from my parser, and also from the pipeline itself |
| Malformed numeric `"abc"` | `ValueError`, not silently turned into missing |
| Save / reload | see section 4 (separate process) |

There are also 16 unit tests (`uv run pytest`) on synthetic data: parsing rules, unfitted builder, train-only statistics, unseen category, seed injection, reproducible and stratified split.

Running `train.py` twice gives the same allocation file, the same metrics and the same reference predictions; only the fit times change.

---

## Peer feedback

> Reviewer: `<name>` · Issue found: `<what blocked them when following the README>` · Resolution: `<what I changed>`

---

## Limits and what's left

- **Test set** not evaluated yet. It's kept for a final, one-off evaluation after selection.
- **Threshold** fixed at 0.5, not tuned. The selected model misses about half of the churners at that threshold.
- **Probabilities** not calibrated.
- **One validation split**, no cross-validation, small hyperparameter choices (no search).
- **Artifact** is a pickle: safe only if it's trusted and loaded with the same library versions. It's 5.3 MB and git-ignored, so it's shared separately (see submission).
- **Environment**: the run was done on Linux aarch64. scikit-learn 1.5.2 with scipy 1.17 shows a harmless `DeprecationWarning` from the logistic regression solver.
- **No experiment tracking yet**: one JSON record per run. That's the next TP.
