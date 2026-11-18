# TP03 answers - Elyas MEZIANI

## 1. Start tracking locally

**If a colleague receives only `mlflow.db`, what else would they need to load the recorded pipeline?**
The artifact store (`mlartifacts/`). The database only holds metadata and the location of each artifact
(`mlflow-artifacts:/...`); the pickled pipeline, its `MLmodel` file and requirements live in `mlartifacts/`. They would
also need a tracking server started with both stores, and a compatible environment (`uv.lock`, the `churnlab` package)
to unpickle the pipeline.

**Experiment vs run. Is the browser page executing the training code?**
An experiment (`telco-churn-tp03`) is a named folder that groups related runs. A run is one execution of training code,
here one candidate: its parameters, metrics, tags, dataset inputs, artifacts and logged model. The browser page does not
execute anything: the training client (`scripts/train_tracked.py`, in another terminal) runs the code and sends records to
the tracking server; the UI only reads what the server stored.

## 2. Record what produced each result

**Metric vs artifact.** A metric is a numeric value stored in the database under a key (`val_average_precision = 0.6662`),
so runs can be sorted and compared on it. An artifact is a file stored in the artifact store (confusion matrix PNG/CSV,
`provenance.json`, the model). Metrics are queryable values; artifacts are opened or downloaded.

**What evidence distinguishes two different train/validation allocations made from the same CSV?**
The original-file SHA-256 is identical in both cases, so it cannot distinguish them. What does: the split seed and fractions
(parameters), the customer allocation table (`split/split_allocation.csv`) and its SHA-256 (tag `split_allocation_sha256`),
and the MLflow dataset digests of `telco_train` / `telco_validation`, which change when the rows change.

**Does recording a random seed guarantee identical results on every machine?**
No. It makes the random choices repeatable for the same code and library versions, but results can still change with
another scikit-learn/numpy version, another BLAS or CPU (floating-point differences), or parallelism. TP02 already saw one
validation prediction differ between macOS and Linux with the same seed.

**Can someone recover the training data from its hash alone?**
No. A SHA-256 is a one-way fingerprint: it can only check that a file someone already has is exactly the right one.
The CSV must be obtained separately (pinned URL + `download_data.py`).

**What happens to traceability if you log a commit while training uncommitted edits?**
The commit points to code that is not the code that ran, so the run looks reproducible when it is not. The script records
`code_dirty=true` and prints a warning in that case; the final runs were made from a clean commit.

## 3. Compare candidate runs

**Why are results from different validation populations harder to compare?**
A different population has a different difficulty and churn rate. Average precision in particular depends on the positive
rate (a random model scores about the prevalence). A score difference could then come from the data, not the model. All
candidates here share the same 1,409 validation customers.

**Why avoid selecting repeatedly on the test set?**
Each selection on the test set leaks information from it into the choice. After several rounds the test score is optimistic
and no longer an unbiased estimate of performance on new customers. Selection is done on validation; the test set stays
reserved for one final evaluation.

## 4. Log, register and recover

**Stored input schema.** The signature, inferred from the input example (3 parsed training rows), lists the six input
columns with their types: `tenure`, `MonthlyCharges`, `TotalCharges` as double, `Contract`, `InternetService`,
`PaymentMethod` as string; the output is a long (class 0/1). It describes the expected columns and types after the TP02
parsing, not the raw CSV text, and not `customerID` or `Churn`. It is inferred from 3 rows, so the "required" flags only
reflect that example; `mlflow.sklearn.load_model` does not enforce it (the pyfunc flavor would).

**Why is loading in a new process stronger evidence than predicting again in the training notebook?**
In the notebook, the fitted object is still in memory: predicting again proves nothing about what was saved. A new process
starts with nothing, so a correct prediction proves the saved artifact alone contains the full fitted pipeline
(preprocessing + classifier), that the registry name/alias resolves to it, and that no hidden in-memory state or `fit`
is needed.

**If the alias later points to a new version, does an already-loaded Python object change?**
No. `load_model` resolves the alias once and copies that version into memory. The object keeps the old model until the
process loads `models:/telco-churn@candidate` again (restart the service, or reload explicitly). That is why the version
number is kept next to the alias for a reproducible handover.

## 5. Persistence

**Database survived but model files lost?** Runs, parameters, metrics, tags, registry versions and aliases are still
listed, but loading the model or opening any artifact fails: the records point to files that no longer exist.

**Only the model files survived?** The pickle could maybe be loaded directly, but there is no link to its run, parameters,
metrics, data hash, commit, registered version or alias, so no traceability, and `models:/` URIs do not resolve.
