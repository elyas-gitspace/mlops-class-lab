# TP01 - Answers (Michael LI)

Worked with Elyas MEZIANI. Repo: https://github.com/elyas-gitspace/mlops-class-lab

Driving question: can another engineer get the same data and run the project without guessing? For the setup and the data check, yes. For training, not yet, because there is no model at this point.

---

## Part A

**A1. Why do the original CSV, a fitted model and the training code have different roles?**

They are not the same kind of thing, so I wouldn't treat them the same way.

- The training code is what we write ourselves. It is small and readable, and teammates can review it, so it goes in Git.
- The raw CSV is what we receive. We should never edit it. We identify it with its source URL and SHA-256 hash, and it stays out of Git because the data can be big and isn't ours.
- The fitted model is what we produce. We can rebuild it from the code, the data and the config, so it doesn't belong in Git either. It will go in an artifact store later.

If we mixed them up, we couldn't tell which file was an input, which was written by us and which was generated.

**A2. Why would using `Churn` as an input invalidate the task?**

`Churn` is the answer we want to predict. If it is also an input, the model just copies it. The score on the test set would look amazing, but in real life we don't know if a customer will leave yet, so we wouldn't have that column. That is target leakage, and it makes the model useless in production.

---

## Part B

**B2.**

1. In `pyproject.toml`, the Python constraint is `requires-python = ">=3.11,<3.13"`. One direct dependency is `numpy~=1.26.0`.
2. In `uv.lock`, numpy appears as `name = "numpy"`, `version = "1.26.4"`, with source PyPI and the file hashes.
3. A transitive dependency is `scipy` (1.17.1 in the lock). We never asked for it. It comes in because scikit-learn needs it. `joblib` and `threadpoolctl` are other examples.
4. Difference between the three:
   - `pyproject.toml` says what the project needs, as ranges, and we write it ourselves. It is committed.
   - `uv.lock` says exactly what uv picked: the version and hash of every package, including transitive ones. It is committed.
   - `.venv/` is what is actually installed on one machine. It is generated and not committed, and we can delete it and rebuild it from the lock.

**B3. `numpy ~= 1.26.0` vs `numpy==1.26.4`**

`~=1.26.0` means `>=1.26.0, <1.27.0`, so several versions are allowed (1.26.0, 1.26.1, ..., 1.26.4). `==1.26.4` allows only one. The range is the rule in `pyproject.toml`, and the lock file remembers which version was really chosen. Also, a version number alone doesn't guarantee that nothing will break, so the lock is what makes it reproducible.

---

## Part C

Results of `inspect_telco.py` (they match what the handout expects):

| Check | Value |
|---|---:|
| Rows x columns | 7,043 x 21 |
| `Churn = No` / `Yes` | 5,174 / 1,869 |
| Churn rate | about 26.54 % |
| Duplicate `customerID` | 0 |
| Whitespace-only `TotalCharges` | 11 |
| SHA-256 | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |

**C3.**

1. **Why classification?** `Churn` is `Yes` or `No`, so we predict a category and not a number.
2. **Class balance and accuracy:** only about 1 customer in 4 leaves. A model that always says "No" would already get about 73.5 % accuracy while finding zero churners. So accuracy alone is misleading, and we should also look at recall, precision, F1 or AUC.
3. **Why is `TotalCharges` a string (object)?** In 11 rows the value is just a space instead of a number. pandas can't convert that to a float, so it keeps the whole column as text.
4. **Why doesn't `isna()` catch it?** A space is not a missing value. It is a real string, so `isna()` says it is fine. We have to strip it and compare to `""`, as the script does, or use `pd.to_numeric(..., errors="coerce")`.
5. **Which evidence shows a teammate used the same file?** The SHA-256 hash (plus the commit-pinned URL and the same counts). The filename `telco.csv` proves nothing, since two files with that name can have different bytes.

---

## Part D

Plain description of what I have on my side:

- `environment-before.json` was recorded on Elyas's Linux machine (Python 3.11.16, uv 0.12.13).
- `environment.json` was recorded on my Windows machine (Python 3.11.9, uv 0.13.4).
- The platform, Python patch version and uv version are different, which is normal. The package versions are the same on both: churnlab 0.1.0, pandas 2.2.3, numpy 1.26.4, scikit-learn 1.5.2. That is what the lock file is for.
- The dataset hash and the inspection numbers are the same.

**What did we reproduce, and what is still missing?**

| Item | Evidence in TP01 | Remaining limitation |
|---|---|---|
| Code | Git commit | Uncommitted edits are not covered by the commit |
| Package selections | `pyproject.toml`, `uv.lock` | The platform and interpreter can change what gets installed |
| Installed environment | Before/after environment records | Doesn't capture every OS or hardware detail |
| Source data | Source URL and SHA-256 | Doesn't say anything about label quality or leakage |
| Trained model | Nothing yet | Needs training config, seed and artifacts (later TPs) |

---

## Part E

- `.gitignore` keeps `.venv/`, `.venv-backup*/`, `.env`, the raw CSV, generated reports, artifacts and model files out of Git. Source, tests, `pyproject.toml`, `uv.lock`, the README and small non-sensitive evidence go in.
- `reports/` is ignored, so the small report is copied to `evidence/tp01/data-overview.json` instead of un-ignoring the whole folder.
- The PR `docs/telco-onboarding` -> `main` (Elyas's repo, PR #1) updates the README with the business question and target, the Python version and setup command, the data source, hash and inspection command, what some folders are for, and the fact that this is a snapshot and not an event log or an experiment.

**My role (reviewer).** I reviewed the README and asked: *"Was the onboarding process not reproducible before?"* The answer was that it was only partly reproducible. The packages were already pinned, but the README did not say where the dataset came from, which bytes to expect or how to check them. The PR adds that.

Commits: `b60315c` initial project, `bba201c` README change, `7c7cdfd` merge into `main`.
PR: https://github.com/elyas-gitspace/mlops-class-lab/pull/1

---

## Part F

- **F1, `git restore README.md`:** it discards the unstaged edit. The file goes back to the last commit and the sentence is lost for good, because it was never staged or committed.
- **F2, `git restore --staged README.md`:** it only unstages the file. The edit is still in my working file, it just isn't in the next commit anymore (`git status --short` shows ` M`).

In short: F1 changes the file, F2 only changes what is staged. Before using F1, I should check that I don't need the changes.

---

## What TP01 does not reproduce yet

- **Training:** no split, seed, hyperparameters, metrics or model are recorded.
- **The machine:** OS, CPU and system libraries are not pinned. Docker should help later.
- **The data:** a hash tells us which bytes we used. It doesn't say the labels are correct, that there is no leakage, or that the data won't drift later.
