# TP02 - Answers (Michael LI)

Worked with Elyas MEZIANI and Amin RACHID. Repo: https://github.com/elyas-gitspace/mlops-class-lab

The numbers below come from the team's run, saved in `evidence/tp02/` (`first_model.json`, `comparison.csv`, `selection.json`, `reliability_checks.json`, `reload_check.json`, `metadata.json`). Entry point: `uv run python scripts/train.py --config configs/tp02_train.json`.

---

## 1. A first model that works

**Setup.** The task is binary classification and `Churn = Yes` is the positive class (1). `customerID` is only used to identify customers and check the split, never as a feature. The split is stratified on `Churn`, 60/20/20, seed 42, done once and saved in `split_allocation.csv`.

| Split | Rows | Churners | Churn rate |
|---|---:|---:|---:|
| train | 4,225 | 1,121 | 26.53 % |
| valid | 1,409 | 374 | 26.54 % |
| test | 1,409 | 374 | 26.54 % (reserved) |

**Checkpoint (validation, only `tenure` + `MonthlyCharges`, threshold 0.5).** The matrix is `[[TN, FP], [FN, TP]]`, with rows = actual and columns = predicted, labels 0 = No and 1 = Yes.

| Model | Accuracy | Confusion matrix |
|---|---:|---|
| Majority baseline (`DummyClassifier`, fitted on train) | 0.7346 | `[[1035, 0], [374, 0]]` |
| Logistic regression | 0.7828 | `[[950, 85], [221, 153]]` |

**Which dataset should the baseline be fitted on?** The training set, like every other model. Fitting it on validation or on everything would be a form of leakage.

**Think about it**

- **What does a system that always says "No churn" get?** 73.5 % accuracy (1035 / 1409), and it finds 0 churners.
- **Can high accuracy and zero churners go together?** Yes, the baseline shows it. About 3 customers in 4 don't churn, so accuracy looks good even when the model is useless for the business.
- **Which data may influence model selection?** Train to fit and validation to compare. The test set must not be used until the very end.

---

## 2. Preprocessing and model in a Pipeline

**Features.** Numeric: `tenure`, `MonthlyCharges`, `TotalCharges`. Categorical: `Contract`, `InternetService`, `PaymentMethod`.

**Parsing before the pipeline (documented, same at training and inference).**

- The CSV on disk is never changed.
- A whitespace-only `TotalCharges` becomes `NaN` (missing). These are the 11 rows we found in TP01.
- Any other non-numeric text raises an error. `pd.to_numeric(errors="coerce")` would turn `"abc"` into `NaN` without telling anyone, and the handout says not to do that. `errors="raise"` fails loudly, so we empty out the blanks first and then fail on anything else.

**Inside the pipeline.**

| Branch | Steps | What is learned on train |
|---|---|---|
| numeric | `SimpleImputer(median)` then `StandardScaler` | medians, means, standard deviations |
| categorical | `SimpleImputer(most_frequent)` then `OneHotEncoder(handle_unknown="ignore")` | most frequent category, list of categories |

**Unseen category at inference.** `handle_unknown="ignore"` encodes it as all zeros, so prediction doesn't crash. That doesn't make the prediction reliable. It only avoids an error.

**Checkpoint.** One `pipeline.fit(X_train, y_train)` trains everything. `predict` takes the six original, unencoded columns. Nothing is refitted on validation.

**Think about it**

- **`fit`, `transform`, `predict`.** `fit` learns something from the training data (medians, categories, model weights). `transform` applies what was learned without learning anything new. `predict` runs the transforms, then the model's prediction.
- **A scaler fitted on validation.** The validation data leaks into the preprocessing, so the validation score is too optimistic. Validation and train would also be scaled differently.
- **Does a Pipeline stop leakage by itself?** No. If I give it the wrong training data (for example train + validation), it will learn from that data. It only keeps the steps in the right order.
- **`Pipeline` vs `ColumnTransformer`.** `Pipeline` orders the steps (preprocess, then classifier). `ColumnTransformer` decides which columns go to which branch (numeric or categorical).

---

## 3. Comparing models fairly

Same six features, same split, same metrics, and a fresh pipeline built for each model.

**Primary metric, chosen before looking at results: average precision (AP)** from the class-1 probability. It measures how well the model ranks churners above non-churners, it doesn't depend on a threshold, and it isn't fooled by the 73/27 imbalance. Precision, recall, F1 and the confusion matrix at threshold 0.5 are supporting evidence.

**Rule, set in the config before the run:** take the best AP. If a simpler model is within 0.01, take the simpler one (logistic regression < decision tree < random forest).

| Model | AP | Precision | Recall | F1 | Accuracy | Fit (s) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (most frequent) | 0.2654 | 0.000 | 0.000 | 0.000 | 0.7346 | 0.005 |
| Logistic regression | 0.6475 | 0.664 | 0.529 | 0.589 | 0.8041 | 0.24 |
| Decision tree (depth 5, min leaf 20) | 0.6173 | 0.636 | 0.602 | 0.618 | 0.8027 | 0.01 |
| Random forest (300 trees, depth 8, min leaf 5) | 0.6662 | 0.648 | 0.487 | 0.556 | 0.7935 | 0.39 |

**Decision: random forest.** Its AP is 0.0187 higher than logistic regression, which is more than the 0.01 tolerance, so the simpler model doesn't win under our rule. The choice is based on validation only.

Caveats:

- The random forest has the best AP, but logistic regression has the best accuracy and the decision tree has the best recall and F1 at 0.5.
- At 0.5 the forest misses 192 of 374 churners. Tuning the threshold on validation would be the next step (not done).
- We can't say the forest beats the first model because the algorithm is better, because the feature set also went from 2 to 6.
- The gaps between models are small and come from a single validation split.

**Think about it**

- **Change the split for every model?** No. Then a difference could come from an easier split and not from the model, so the comparison wouldn't be fair.
- **Does the highest accuracy meet the team's needs?** Not necessarily. Logistic regression has the best accuracy here, but the team wants to find churners, so ranking (AP) and recall matter more.
- **Two close candidates?** I would pick the simpler one. It is faster, easier to explain and less likely to overfit. That's why our rule has a tolerance.
- **Is a probability of 0.8 a real 80 % risk?** Not automatically. Random forest scores are averaged tree votes and are not calibrated. We would need a calibration curve to check.

---

## 4. Saving the fitted pipeline

`artifacts/tp02/pipeline.joblib` holds the fitted preprocessing and the classifier together. `metadata.json` records:

- the data source, URL and SHA-256;
- the split seed, fractions and the allocation file with its hash;
- the feature roles, the missing-value and unknown-category policy, and the learned statistics;
- the model name and parameters;
- the positive class and the threshold (0.5);
- the validation metrics and the selection decision;
- the git commit (`code_dirty: false`) and the config hash;
- the library versions (Python 3.11.16, scikit-learn 1.5.2, pandas 2.2.3, numpy 1.26.4);
- the artifact's own SHA-256.

**Reload check, in a separate process** (`scripts/check_reload.py`). It scored 25 saved validation customers, with no `Churn` column and no `fit` call. The maximum score difference from the original is 9.0e-17 (tolerance 1e-12) and the 25 class predictions are identical. Result: passed.

**Think about it**

- **Why isn't saving only the classifier enough?** It expects scaled numbers and one-hot columns. Without the fitted preprocessing, someone would have to rebuild it by hand and could get different medians or a different column order.
- **What is inside a fitted imputer or encoder?** The training medians, the most frequent categories, and the list of categories (which fixes the order of the one-hot columns).
- **Can two `model.joblib` files be different models?** Yes, the filename proves nothing, like `telco.csv` in TP01. That's why we store and check the artifact's SHA-256 before loading.
- **sklearn Pipeline vs Kubeflow Pipelines.** A sklearn Pipeline is one Python object that chains preprocessing and a model in memory. A Kubeflow pipeline is a workflow (DAG) of separate steps, each running in its own container on Kubernetes and passing files between them.

---

## 5. Reliability checks

From `reliability_checks.json` (all passed):

| Check | Result |
|---|---|
| Customer IDs disjoint across splits | 4,225 / 1,409 / 1,409 IDs, 0 overlap, 7,043 total |
| Imputer statistics come from train only | learned medians `[29.0, 70.55, 1373.625]` equal the train medians. The full dataset would give `[29.0, 70.35, 1397.475]` |
| Predicting doesn't change the fitted state | medians identical before and after, pickled pipeline identical |
| Missing numeric (`None`, `" "`, `NaN`) | same score as with the training median filled in |
| Unseen category `"Three year"` | `Contract` columns all zeros, no error, but not a reliable prediction |
| Missing column `Contract` | `ValueError` (from the parser and from the pipeline) |
| Malformed numeric `"abc"` | `ValueError`, not silently turned into missing |
| Save and reload | passed (section 4) |

There are also unit tests (`uv run pytest`) on synthetic data.

---

## Peer review

Amin reproduced the workflow on macOS using only the README. He got an AP of 0.6665 instead of 0.6662 and 193 false negatives instead of 192. The README now says that runs are identical on the same platform, but tree ensembles can differ slightly across platforms. The selected model didn't change. His results are in `evidence/tp02/macos/`.

---

## Limits and what is left

- The test set hasn't been evaluated. It is kept for one final evaluation after selection.
- The threshold is fixed at 0.5 and not tuned, and the probabilities are not calibrated.
- There is one validation split, no cross-validation and no hyperparameter search.
- The artifact is a pickle, so it must be trusted and loaded with the same library versions. It is git-ignored and shared separately.
- There is no experiment tracking yet, only one JSON record per run. That comes in the next TP.
