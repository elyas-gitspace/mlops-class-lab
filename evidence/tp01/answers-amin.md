# TP01 - Answers (Amin Rachid)

Team: Elyas Meziani, Michael Li, Amin Rachid. Repository: https://github.com/elyas-gitspace/mlops-class-lab

## Part A

**A1. Why do the raw CSV, a fitted model and the training code play different roles?**
The source code is written by us and reviewed: it belongs in Git. The raw CSV is an *input* we received and must never edit; it is identified by its source URL and SHA-256 (committed in `data/raw/telco.source.json`) rather than committed itself. A fitted model is an *output* generated from code + data + configuration; it can be regenerated and will be versioned with dedicated artifact tooling in later TPs, not in Git.

**A2. Why would including `Churn` among the inputs invalidate the task?**
`Churn` is the label we want to predict. Giving it to the model is target leakage: the model would simply copy the answer, look perfect offline, and be useless in production where the label is unknown at prediction time.

## Part B

1. **Python constraint:** `requires-python = ">=3.11,<3.13"`. **Direct dependency:** `numpy~=1.26.0`.
2. **Same dependency in `uv.lock`:** numpy resolved to exactly `1.26.4`, with the URL and hash of each wheel.
3. **Transitive dependency (`uv tree`):** `scipy` is not declared by us; it is pulled in by `scikit-learn`. Same for `python-dateutil` and `tzdata` via `pandas`.
4. **`pyproject.toml` vs `uv.lock` vs `.venv/`:** `pyproject.toml` states what the project *needs* (allowed ranges). `uv.lock` records what was *resolved* (exact versions and hashes of all direct and transitive packages). `.venv/` is the environment *installed* on one machine from the lock; it is generated and never committed.

**B3.** `numpy ~= 1.26.0` accepts any `1.26.x` release; `numpy==1.26.4` accepts exactly one. The range is in `pyproject.toml`; the single choice is in `uv.lock`.

## Part C

1. **Classification:** the target `Churn` takes two discrete values (`Yes`/`No`), so we predict a class, not a quantity.
2. **Class balance:** only 26.54 % of customers churn. A model that always answers "No" would already score 73.5 % accuracy while detecting no churner. We will need precision/recall, F1 or ROC-AUC.
3. **`TotalCharges` read as a string:** 11 rows contain a blank `" "` instead of a number. One non-numeric value is enough for pandas to read the whole column as `object`.
4. **`isna()` misses whitespace:** `" "` is a non-empty string, not a missing value (`NaN`). It must be detected with `.str.strip().eq("")` or converted with `pd.to_numeric(..., errors="coerce")`.
5. **Evidence of the same source file:** the SHA-256 `16320c9c…5e91`, together with the commit-pinned source URL. The filename alone proves nothing: the course `datasets.csv` contains the same rows but was re-saved (BOM, CRLF, quoted lines), so its hash is `62ca518b…` and the inspection script rejects it.

## Part D - Environment recreation on macOS

Evidence in `evidence/tp01/macos/`. I saved `environment-before.json`, renamed `.venv/` to `.venv-backup/`, ran `uv sync --locked`, and re-ran both scripts.

- Before/after on my Mac: identical (Python 3.11.15, pandas 2.2.3, numpy 1.26.4, scikit-learn 1.5.2, churnlab 0.1.0).
- `uv.lock` and `pyproject.toml` unchanged; `uv run pytest` passes.
- Data overview: byte-for-byte identical to Elyas's (same hash, 7,043 rows, 5,174 No / 1,869 Yes, 0 duplicate IDs, 11 blank `TotalCharges`).
- Compared with Elyas's Linux record: same package versions, but a different OS (macOS arm64 vs Linux aarch64), Python patch (3.11.15 vs 3.11.16) and uv version (0.11.14 vs 0.12.13). The lock fixes packages, not the interpreter build or OS.

| Item | Evidence produced in TP01 | Remaining limitation |
|---|---|---|
| Code | Git commit | Uncommitted edits are not identified by that commit |
| Package selections | `pyproject.toml`, `uv.lock` | Platform/interpreter selection can affect installations (e.g. Python 3.11.15 vs 3.11.16 between us) |
| Installed environment | Before/after environment records (Linux and macOS) | This does not capture every OS or hardware detail |
| Source data | Source URL and SHA-256 | This does not establish label quality or absence of leakage |
| Trained model | Not produced in TP01 | Requires later training/configuration/artifact evidence |

## Part E - Pull request

- PR URL: https://github.com/elyas-gitspace/mlops-class-lab/pull/1 (`docs/telco-onboarding` → `main`)
- Commits: `b60315c` initial project · `bba201c` README change · `7c7cdfd` merge into `main`
- Reviewer question (@PraiseTheLemon): *Was the onboarding process not reproducible before?* Only partly: packages were already pinned by `pyproject.toml` and `uv.lock`, but the README did not say where the data came from, which bytes to expect or how to check them. The PR adds the pinned URL, the SHA-256, the inspection commands with expected counts, the target column and the folder roles.
- Cross-machine check: branch `evidence/macos-reproduction` adds my macOS evidence (Part D above).

## Part F - Restore vs unstage

`git restore README.md` (F1) overwrites the working file with the staged/committed version: the unstaged edit is **lost**. `git restore --staged README.md` (F2) only removes the change from the index: the edit **stays** in the working file, it is just no longer part of the next commit.

## What TP01 does not reproduce yet

No model, training configuration, random seed or metrics exist yet. The OS, interpreter build and hardware are not pinned. The hash proves which bytes we used, not that the labels are correct.
