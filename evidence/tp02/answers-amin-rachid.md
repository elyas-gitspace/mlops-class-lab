# TP02 - Answers (Amin Rachid)

Team: Elyas Meziani, Michael Li, Amin Rachid. Repository: https://github.com/elyas-gitspace/mlops-class-lab

Elyas wrote the TP02 code (the `churnlab` modules, `scripts/train.py` and the check scripts). On my side, I reviewed that code, ran it again on my Mac following only the README, and compared my results with his. That review is what surfaced the only real issue of the TP (Part 6). The answers below are my own. The evidence from my run is in `evidence/tp02/macos/`.

## 1. A first model that works

The task is binary classification. The positive class is `Churn = Yes` (encoded `1`). `customerID` is only used to identify customers and to check the split. It is never given to the model, and neither is `Churn`.

The split is done once, stratified on `Churn`, with seed `42`: 4,225 customers in train, 1,409 in validation and 1,409 in test, each with about 26.5 % churners. The allocation of every customer is saved in `split_allocation.csv`. On my Mac, this file is identical to Elyas's for all 7,043 customers. The test set was not used at all in this TP.

With only `tenure` and `MonthlyCharges`, logistic regression reaches 0.783 accuracy on validation, against 0.735 for the baseline that always predicts "No". Confusion matrices read `[[TN, FP], [FN, TP]]` (rows = actual, columns = predicted, 0 = stays, 1 = leaves).

| Model | Accuracy | Confusion matrix |
|---|---:|---|
| Baseline `DummyClassifier` (fitted on train) | 0.7346 | `[[1035, 0], [374, 0]]` |
| Logistic regression (2 features) | 0.7828 | `[[950, 85], [221, 153]]` |

**What would a system that always predicts "no churn" achieve?** Exactly the baseline: 73.5 % accuracy and not a single churner detected. It is the clearest example of what we saw in TP01 with the class imbalance.

**Can high accuracy coexist with finding zero churners?** Yes, and the baseline proves it. With 26.5 % positives, accuracy mostly rewards classifying the customers who stay correctly, whereas the team cares about the ones who leave.

**Which data are allowed to influence model selection?** Train to learn, validation to compare. The test set must stay untouched until the final evaluation. Otherwise, we end up picking the model that got lucky on that specific set.

## 2. Preprocessing and model in a single Pipeline

The six features are `tenure`, `MonthlyCharges` and `TotalCharges` (numeric), then `Contract`, `InternetService` and `PaymentMethod` (categorical).

Before the pipeline, a parsing step (`parse_features`) converts the columns. It learns nothing and is applied the same way at training and at inference. The 11 blank `TotalCharges` become missing values. I checked that they all belong to customers with `tenure = 0`, meaning new customers who have not been billed yet. On the other hand, a text like `"abc"` raises an error. Rather than using `pd.to_numeric(errors="coerce")` directly, which would silently turn `"abc"` into `NaN`, the code first marks the truly blank cells, converts, and then fails if a non-blank value could not be read. That is exactly the distinction the handout asks for between an allowed missing value and malformed data.

Inside the pipeline, a `ColumnTransformer` sends each column to its branch:

- **Numeric**: `SimpleImputer(strategy="median")` then `StandardScaler`. Medians, means and standard deviations are learned on train.
- **Categorical**: `SimpleImputer(strategy="most_frequent")` then `OneHotEncoder(handle_unknown="ignore")`. The most frequent category and the list of categories are learned on train.

A category never seen during training (for example `"Three year"`) is encoded as zeros on every `Contract` column. The prediction does not crash, but that does not mean it is reliable for that customer.

**What do `fit`, `transform` and `predict` do?** `fit` learns from the data (medians, categories, model weights). `transform` applies what was learned without changing anything. `predict` chains the transformations and then the model, still without learning anything.

**What goes wrong if the scaler is fitted on validation data?** Information from validation leaks into the model, so its scores become too optimistic. Validation would also no longer be transformed the same way as train, which distorts the comparison.

**Does a Pipeline prevent leakage on its own?** No. If we call `fit` on train and validation together, it will learn from both without complaining. It guarantees the order of the steps, not the choice of data we give it.

**What orders what?** The `Pipeline` orders the steps (preprocess, then classify). The `ColumnTransformer` routes the columns (numeric on one side, categorical on the other).

## 3. Comparing models fairly

The three models and the baseline use the same six features, the same split and the same metrics. Each candidate gets a fresh pipeline (`build_pipeline` is called again every time), so none of them reuses the state learned by another.

The primary metric was set in `configs/tp02_train.json` before running the comparison: validation average precision, computed from the class-1 probability. It measures how well the model ranks customers who leave above those who stay, without depending on a threshold, and unlike accuracy it is not fooled by the 73/27 imbalance. The selection rule also includes a 0.01 tolerance: if a simpler model is within 0.01 of the best one, that is the one we keep.

| Model | AP (Linux) | AP (my Mac) | Accuracy | Recall at 0.5 |
|---|---:|---:|---:|---:|
| Baseline | 0.2654 | 0.2654 | 0.7346 | 0.000 |
| Logistic regression | 0.6475 | 0.6475 | 0.8041 | 0.529 |
| Decision tree | 0.6173 | 0.6173 | 0.8027 | 0.602 |
| Random forest | 0.6662 | 0.6665 | 0.7935 (0.7928 on mine) | 0.487 (0.484 on mine) |

The random forest is selected in both cases. Its AP is about 0.019 above logistic regression, which is beyond the tolerance. That said, it is not the model with the best accuracy (logistic regression) nor the best recall at the 0.5 threshold (the decision tree). At that threshold, it misses about half of the customers who leave. If the team mostly wants to catch them, the threshold should be chosen on validation rather than kept at 0.5. Finally, we cannot say that the random forest "beats" the first model because of the algorithm, since the number of features also went from 2 to 6.

**Should the split change for each model?** No. The differences would then mix the effect of the model with the effect of an easier or harder validation set. With the same split, the only thing that changes is the model.

**Is the most accurate model the one the team needs?** Not necessarily, and that is the case here. Logistic regression has the best accuracy, but the team mainly wants to spot churners, so ranking (AP) and recall matter more.

**What if two candidates are close?** We keep the simpler one. It is faster, easier to explain and less exposed to overfitting. That is exactly the role of the tolerance in the rule.

**Is a predicted probability of 0.8 a real 80 % risk?** Not automatically. Random forest probabilities are averages of tree votes, not calibrated probabilities. Checking that would require a calibration curve.

## 4. Saving the whole fitted pipeline

The `pipeline.joblib` file contains the fitted preprocessing and classifier in a single object. Next to it, `metadata.json` records the data hash, the seed and the customer allocation, the feature roles, the preprocessing policy, the model parameters, the metrics, the positive class, the threshold, the Git commit and the package versions. It also contains the artifact's own SHA-256, which is checked before every load.

On my Mac, `scripts/check_reload.py` reloads the artifact in a separate process and predicts 25 validation customers without a `Churn` column. The maximum difference with the scores computed right after training is `1.1e-16` (tolerance `1e-12`), and the 25 class predictions are identical. No `fit` is called in that process.

**Why is saving only the classifier not enough?** The classifier expects scaled numbers and one-hot columns. Without the fitted preprocessing, it would have to be rebuilt by hand, with the risk of getting other medians or another column order.

**What lives inside a fitted imputer or encoder?** The training medians, the most frequent category of each column and the list of categories, which fixes the order of the one-hot columns.

**Could two `model.joblib` files contain different models?** Yes, and we had a concrete example. My `pipeline.joblib` (SHA-256 `e42ab3f7…`) is not the same as Elyas's (`289cd06d…`), even though the code, the config and the data are identical. It is the same reasoning as in TP01 with the CSV: the file name proves nothing, the hash does.

**sklearn Pipeline or Kubeflow pipeline?** A sklearn `Pipeline` is a single Python object that chains transformations and a model in memory. A Kubeflow pipeline is a sequence of separate steps (a graph), each run in its own container, passing files to each other (for example "download, train, evaluate, deploy").

## 5. Reliability checks

I ran `scripts/check_pipeline.py` again on my Mac. All 11 checks pass:

- **Split**: no `customerID` shared between train, validation and test.
- **Statistics learned on train only**: the imputer medians and the scaler means are those of train, not those of the full dataset.
- **`predict` changes nothing**: the learned statistics are identical before and after a prediction.
- **Allowed missing values**: a blank, `None` or `NaN` `TotalCharges` is replaced by the training median, and a missing `Contract` by the most frequent category.
- **Unknown category**: `"Three year"` is encoded as zeros, without an error.
- **Missing column**: a missing `Contract` raises a visible error, rather than a silently wrong prediction.
- **Malformed text**: `"abc"` in a numeric column raises an error, instead of becoming a missing value.

The 16 unit tests also pass (`uv run pytest`).

## 6. Review and issue found

The handout asks to exchange the project with a peer and record one issue. I followed only Elyas's README on another platform (macOS arm64, against Linux aarch64 for him), with the same `uv.lock`, the same config and the same seed. The split, the baseline, logistic regression and the decision tree give exactly his numbers.

The random forest, however, does not give quite the same result. AP goes from 0.6662 to 0.6665, and one validation customer is predicted differently (193 false negatives instead of 192). The package versions are nevertheless identical. The difference comes from floating-point computations, which are not strictly the same from one platform to another, and a model made of 300 trees is more sensitive to them than a logistic regression. The problem was not so much the gap itself, which does not change the selection, as the way it was presented. Elyas's answers said that re-running `train.py` gives the same metrics, and the README said nothing about the platform. That is true on the same machine, not from one machine to another.

The resolution was to state this in the README, in the section about what is not reproduced yet, and to add my macOS results in `evidence/tp02/macos/` so the gap can be checked. What I take from this is that pinning package versions is not enough to pin results: the platform is part of the environment too.

## What is left

The test set has not been evaluated yet. It is reserved for a single evaluation once selection is finished. The threshold stayed at 0.5 and the probabilities are not calibrated. The comparison relies on a single validation split, without cross-validation. Finally, each training run is recorded in a plain JSON file. Experiment tracking and model versioning come in the next TP.
