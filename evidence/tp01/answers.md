# 📦 TP01 · A reproducible churn project with Git

> *Can another engineer get the same data and run our project without guessing?*
> That's the question for this TP. Short answer: yes, for the setup and the data check. Not yet for training (no model so far).

|  |  |
|---|---|
| 👥 **Team** | Elyas MEZIANI · Michael LI · Amin RACHID |
| 🗂️ **Repo** | [elyas-gitspace/mlops-class-lab](https://github.com/elyas-gitspace/mlops-class-lab) |
| 🔀 **Pull request** | [#1 · Document reproducible Telco onboarding](https://github.com/elyas-gitspace/mlops-class-lab/pull/1) |
| 🔖 **Commits** | `b60315c` initial project · `bba201c` README change · `7c7cdfd` merge into `main` |

---

## A · The project and the data

**Why don't the raw CSV, the model and the code play the same role in the repo?**

Think of it as three kinds of things:

- the **code** is something *we write*. It's small, we review it, it goes in Git;
- the **raw CSV** is something *we receive*. We never touch it. We identify it by its URL and its SHA-256, and we keep it out of Git;
- the **model** is something *we produce*. We can rebuild it from code + data + config, so it'll go into an artifact store later, not Git.

If you mix them up, you can't tell anymore what was written, what was received and what was generated.

**Why can't `Churn` be an input?**

Because it's the answer we're trying to predict. Give it to the model and it just copies it: great score in evaluation, useless in production, where nobody knows the label yet. That's target leakage.

---

## B · The Python environment

- **Python constraint** in `pyproject.toml`: `requires-python = ">=3.11,<3.13"`
- **A direct dependency**: `numpy~=1.26.0`. In `uv.lock` it shows up as `numpy 1.26.4`, with its source (PyPI) and file hashes.
- **A transitive dependency** (from `uv tree`): `scipy 1.17.1`, which comes with `scikit-learn`. We never asked for it ourselves.

**The three files, in one line each:**

| File | What it is | In Git? |
|---|---|:---:|
| `pyproject.toml` | What we *need* (allowed ranges, written by hand) | ✅ |
| `uv.lock` | What uv *resolved* (exact version + hash of every package) | ✅ |
| `.venv/` | What's actually *installed* on one machine | ❌ |

**`~=` vs `==`:** `numpy ~= 1.26.0` accepts anything from `1.26.0` up to (not including) `1.27.0`. `numpy == 1.26.4` accepts only that one. The range goes in `pyproject.toml`; the lock file remembers which version was actually picked.

---

## C · Inspecting the Telco dataset

Everything matches what the handout expects ✅

| Check | Value |
|---|---:|
| Rows × columns | 7,043 × 21 |
| `Churn = No` / `Yes` | 5,174 / 1,869 |
| Churn rate | ~26.54 % |
| Duplicate `customerID` | 0 |
| Blank `TotalCharges` | 11 |
| SHA-256 | `16320c9c…3055e91` |

Full report: `evidence/tp01/data-overview.json`.

1. **Why classification?** `Churn` is `Yes` or `No`. We're predicting a category, not a number.
2. **What about accuracy?** Only about 1 customer in 4 churns. A model that always answers "No" already gets ~73.5 % accuracy and catches zero churners. We'll have to look at recall, precision, F1 or AUC instead.
3. **Why is `TotalCharges` a string column?** 11 rows contain a space instead of a number. pandas can't turn `" "` into a float, so it keeps the whole column as text (`object`).
4. **Why doesn't `isna()` catch it?** `" "` isn't missing, it's a real string with a space in it. You have to strip it and compare to `""` (like the script does), or use `pd.to_numeric(..., errors="coerce")`.
5. **How do I know my teammate checked the same file?** Same SHA-256, same commit-pinned URL, same counts. The filename proves nothing.

> 💡 **We hit the "source mismatch" case for real.** The first CSV we downloaded (`datasets.csv`, from another link) had a different hash (`62ca518b…a20c`). It looked the same, but it had a BOM at the start and every line was wrapped in quotes. The hash check rejected it, so we switched to the commit-pinned IBM file. Good lesson: same name ≠ same data.

---

## D · Recreating the environment

We saved the environment, renamed `.venv/` to `.venv-backup/`, then rebuilt it with `uv sync --locked`. Before and after are **identical**:

> Python 3.11.16 · churnlab 0.1.0 · pandas 2.2.3 · numpy 1.26.4 · scikit-learn 1.5.2 · uv 0.12.13

`pyproject.toml`, `uv.lock` and `telco.csv` kept the same hashes, and the data report came out exactly the same.

⚠️ These records come from a Linux (aarch64) machine, as you can see in the `platform` field. On another OS the platform and the installed wheels change, but the resolved versions stay the same.

**What we can and can't reproduce so far:**

| Item | What proves it in TP01 | What's still missing |
|---|---|---|
| Code | Git commit | Uncommitted edits aren't covered by the commit |
| Package versions | `pyproject.toml` + `uv.lock` | Platform/interpreter can change what gets installed |
| Installed environment | Before/after environment records | Doesn't capture every OS or hardware detail |
| Source data | URL + SHA-256 | Says nothing about label quality or leakage |
| Trained model | Nothing yet | Needs training config + artifacts (later TPs) |

---

## E · Working through a pull request

- `.gitignore` keeps out `.venv/`, `.venv-backup*/`, `.env`, the raw CSV, `data/processed/`, `artifacts/`, `reports/` and model files.
- We copied the data report to `evidence/tp01/data-overview.json` instead of un-ignoring `reports/`.
- The PR (`docs/telco-onboarding` → `main`) adds the business question + target, the Python version and setup command, the data source + hash + inspection commands, the folder roles, and why this is a snapshot (not an event log or an experiment).

**💬 Review**

> **@PraiseTheLemon:** *Was the onboarding process not reproducible before?*
>
> **Us:** Only partly. `pyproject.toml` and `uv.lock` already pinned the packages, but the README only had two setup commands. You had to guess where the dataset came from, which bytes to expect and how to check them. The PR adds the pinned URL, the SHA-256, the download/inspection commands with expected counts, the target column and the folder roles, so you can redo the setup and the data check without guessing.

---

## F · Recovering from a local mistake

| | Command | What happens to the edit |
|---|---|---|
| **F1** | `git restore README.md` | **Gone.** The file goes back to the last commit. Since the edit was never staged or committed, Git can't bring it back. |
| **F2** | `git restore --staged README.md` | **Still there.** It's just removed from the staging area, so it won't be in the next commit (`git status` shows ` M`). |

Basically: F1 touches the file, F2 only touches what's staged.

---

## 🚧 What TP01 doesn't cover yet

- **Training runs:** no split, seed, hyperparameters, metrics or model are recorded yet.
- **The machine itself:** OS, hardware and system libraries aren't pinned (a Docker image should help later).
- **The data itself:** a hash says *which* bytes we used. It doesn't say the labels are right, that there's no leakage, or that the data won't drift.
