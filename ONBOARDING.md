# Getting set up on this repo (Backend 2)

You already have the datasets and docs as a zip. **The repo is now the source of truth** —
some of those files have changed since your zip, so work from a fresh clone rather than
unzipping on top of anything.

## 1. Clone into a brand-new folder

```bash
git clone https://github.com/PranshuRanjan10/Good_Question_backend.git
cd Good_Question_backend
```

## 2. Copy in only the files you wrote yourself

Your own work — `train_anomaly.py`, ingestion, safety rules, API, DB code — goes straight in.
Those are new paths, so nothing conflicts.

**Do not copy these over from your zip.** The repo versions are newer:

| Path | Why |
|---|---|
| `data/datasets/README.md` | Anomaly detector is now **supervised LightGBM**, no Isolation Forest |
| `data/HANDOFF_backend_teammate.md` | Same change, plus corrected artifact names |
| `docs/` | Unchanged, but the repo is canonical |
| `backend/app/models/` | Pranshu's models — call them, don't edit them |

## 3. Environment

```bash
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt
```

Version match matters: the `.joblib` artifacts were saved with scikit-learn 1.9.1 and
LightGBM 4.7.0. A different major version may fail to load or behave subtly differently.

## 4. Recreate the datasets

They're **not in the repo** (65 MB, and fully reproducible):

```bash
.venv/bin/python data/generate_data.py
```

About 25 seconds, seed 42, byte-identical to the copy in your zip. This also guarantees we're
both training on exactly the same data.

## 5. Check the models load

From the `backend/` folder:

```bash
python -m app.models.task_time
```

Expect `{'task_id': 'T002', 'planned_min': 45, 'p10_min': 50, 'p50_min': 57, 'p90_min': 62,
'remaining_min': 34, ...}`. If you get that, your whole setup is correct.

`backend/app/models/README.md` has the calling conventions for all three functions.

## 6. Work on a branch, never directly on main

```bash
git checkout -b backend2-anomaly
```

Commit as you go, then:

```bash
git push -u origin backend2-anomaly
```

Open a pull request on GitHub. We review and merge, so `main` always stays working.

**Before you start each session:** `git checkout main && git pull`, then branch from there.
Small commits pushed often beat one big one at the end.

## What's already done (don't rebuild it)

- `backend/artifacts/` — trained models, committed, ready to load
- `operator_baselines.json` — per-operator medians for your anomaly explanations
- `predict_task()`, `compute_readiness()`, `operator_profile()` — callable now

## What's yours

Anomaly detector (supervised LightGBM + rules), training recommender, ingestion gateway,
state manager, safety rules engine, SQLite and the REST endpoints. See
`data/HANDOFF_backend_teammate.md` section 2.

**Pranshu is taking the feature builder and decision layer**, since his models sit on both
sides of that boundary. Shared contract: the feature row uses the same column names as
`anomaly_windows_5min.csv`.

Message before changing any field name in the shared interfaces.
