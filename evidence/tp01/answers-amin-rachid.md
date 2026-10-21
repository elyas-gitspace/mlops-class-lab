# TP01 - Answers (Amin Rachid)

Team: Elyas Meziani, Michael Li, Amin Rachid. Repository: https://github.com/elyas-gitspace/mlops-class-lab

Elyas set up the starter project and the README pull request. On my side, I rebuilt the project on my Mac to check that it could be reproduced on another machine (Part D), and I wrote the answers below.

## Part A - The project and the data

**A1. Why do the raw CSV, a fitted model and the source code play different roles in the repository?**

These three things do not come from the same place, and that is what decides where they should live. The code is what we write ourselves. It is small, it gets reviewed in a PR, so it belongs in Git. The raw CSV is what we receive. We must never edit it, and rather than committing it, we identify it by its source URL and its SHA-256 hash (recorded in `data/raw/telco.source.json`). Finally, the model is what we produce from the code, the data and a configuration. It can be regenerated, and it will go into an artifact store in later TPs, not into Git. Mixing the three means we no longer know what was written, what was received and what was generated.

**A2. Why would including `Churn` among the model inputs invalidate the prediction task?**

`Churn` is exactly the column we are trying to predict. If we give it to the model, there is nothing left to learn: it just copies the answer. That would give an almost perfect score in evaluation, while in production the value of `Churn` does not exist yet at the moment we make the prediction. This is called target leakage, and it is the kind of mistake that does not show in the metrics but makes the model useless.

## Part B - The Python environment

1. **Python constraint**: in `pyproject.toml`, `requires-python = ">=3.11,<3.13"`. **Direct dependency**: `numpy~=1.26.0`.
2. **The same dependency in `uv.lock`**: numpy is pinned to version `1.26.4`, with the URL and hash of each downloadable file (wheel).
3. **Transitive dependency**: `uv tree` shows `scipy 1.17.1` under `scikit-learn`. We never declared it ourselves. It is scikit-learn that needs it. The same goes for `python-dateutil` and `tzdata`, pulled in by `pandas`.
4. **`pyproject.toml`, `uv.lock` and `.venv/`**: `pyproject.toml` says what the project needs, with allowed version ranges. `uv.lock` says what was actually chosen within those ranges, for every dependency (direct and transitive). `.venv/` is the concrete installation on one given machine. It is rebuilt from the lock and never committed.

**B3.** `numpy ~= 1.26.0` accepts any `1.26.x` version, whereas `numpy==1.26.4` accepts only one. In practice, the range goes in `pyproject.toml` and the exact version that was picked is written in `uv.lock`. A range leaves room for bug-fix releases. The lock guarantees that everyone installs the same thing.

## Part C - The Telco data

1. **Why classification?** The target `Churn` only takes two values (`Yes` or `No`). We predict a category, not a quantity.
2. **What does the class balance say about accuracy?** Only 26.54 % of customers leave (1,869 out of 7,043). A model that always answered "No" would already reach 73.5 % accuracy without detecting a single churner. Accuracy alone will therefore not be enough. We will have to look at precision, recall, F1 or AUC.
3. **Why is `TotalCharges` read as text?** Eleven rows contain a space (`" "`) instead of a number. A single non-numeric value is enough for pandas to read the whole column as `object`, even though it represents an amount.
4. **Why is `isna()` not enough?** For pandas, `" "` is not a missing value. It is a non-empty string. It has to be found explicitly with `.str.strip().eq("")`, or the column has to be converted with `pd.to_numeric(..., errors="coerce")` so that these cells become `NaN`.
5. **What evidence shows we inspected the same file?** The SHA-256 (`16320c9c…5e91`), together with the source URL pinned to a specific IBM commit. The file name proves nothing, and we saw it concretely. The `datasets.csv` file we had received contained exactly the same 7,043 rows, but it had been re-saved (a BOM at the start, Windows line endings, every line wrapped in quotes). Its hash was therefore `62ca518b…` and the inspection script rejected it. What I take from this is that the same content "to the eye" does not mean the same bytes.

## Part D - Recreating the environment on macOS

Elyas had already produced the evidence on his Linux machine. Rather than doing the same thing in the same place, I redid Part D on my Mac to check that the project really reproduced elsewhere. The files are in `evidence/tp01/macos/`.

Concretely, I recorded the environment in `environment-before.json`, renamed `.venv/` to `.venv-backup/`, then ran `uv sync --locked` and both scripts again. The recreated environment is identical to the one before (Python 3.11.15, pandas 2.2.3, numpy 1.26.4, scikit-learn 1.5.2, churnlab 0.1.0). `uv.lock` and `pyproject.toml` did not change, and `uv run pytest` passes. The data report is identical to Elyas's, character for character (same hash, 7,043 rows, 5,174 "No" and 1,869 "Yes", no duplicates, 11 blank `TotalCharges`).

The comparison with Linux is more interesting. The package versions are the same, but not the operating system (macOS arm64 versus Linux aarch64), nor the exact Python version (3.11.15 versus 3.11.16), nor the uv version (0.11.14 versus 0.12.13). This shows clearly that the lock file pins the packages, not the interpreter or the operating system.

| Item | Evidence produced in TP01 | Remaining limitation |
|---|---|---|
| Code | Git commit | Uncommitted edits are not covered by that commit |
| Package selections | `pyproject.toml`, `uv.lock` | Platform and interpreter can change the installation (3.11.15 on my side, 3.11.16 on Elyas's) |
| Installed environment | Before/after records (Linux and macOS) | They do not capture every OS or hardware detail |
| Source data | Source URL and SHA-256 | This guarantees neither label quality nor the absence of leakage |
| Trained model | Not produced yet | Training, configuration and artifact evidence will be needed in later TPs |

## Part E - Working through a pull request

- **PR**: https://github.com/elyas-gitspace/mlops-class-lab/pull/1 (`docs/telco-onboarding` into `main`).
- **Commits**: `b60315c` (initial project), `bba201c` (README change), `7c7cdfd` (merge into `main`).
- **Reviewer question** (@PraiseTheLemon): "Was the onboarding process not reproducible before?" Not completely. The packages were already pinned by `pyproject.toml` and `uv.lock`, but the README said neither where the data came from, nor which bytes to expect, nor how to check them. The PR adds the pinned URL, the SHA-256, the inspection commands with the expected results, the target column and the role of each folder.
- **Check on another machine**: my branch `evidence/macos-reproduction` adds the Part D evidence.

## Part F - Restoring versus unstaging

The difference lies in what is lost. `git restore README.md` (F1) puts the file back in the state of the last commit: the unstaged change is lost, and Git cannot bring it back since it was never recorded. `git restore --staged README.md` (F2) does not touch the file. It only removes the change from the index (the staging area), so it stays in the file but will not be part of the next commit. The first one therefore erases work, while the second only changes what will be committed.

## What TP01 does not reproduce yet

For now there is no model, no training configuration, no random seed and no metrics. The operating system, the exact interpreter version and the hardware are not pinned, as the comparison between my Mac and Elyas's machine showed. Finally, the hash proves which bytes we used, not that the labels are correct. This foundation is necessary, but it does not yet guarantee that a training run will give the same result twice.
