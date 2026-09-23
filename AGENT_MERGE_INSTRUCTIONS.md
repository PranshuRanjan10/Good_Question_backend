# Instructions for the AI agent integrating Backend 2's work

**Read this fully before running any git command.** You are helping Niharika (Backend 2) merge
her locally-built code into a shared repo that already contains Pranshu's (Backend 1) work.

Her code was built from a zip with **no git history**, so her folder contains her own new
files *plus* older copies of files that have since changed in the repo. Your job is to bring
**only her own work** into the repo, leave everything else untouched, and deliver a pull
request. Nothing more.

---

## Non-negotiable rules

1. **Never push to `main`.** All work goes on a branch, then a pull request.
2. **Never use `git push --force`**, `git reset --hard origin/main` on shared history, or any
   command that rewrites published history.
3. **Never modify, "improve", refactor, reformat or reorganise files you did not write.**
   That includes renaming files or moving directories.
4. **Do not "rework the codebase to avoid merge conflicts."** Conflicts are prevented by file
   ownership, which is already settled below. Rewriting working code days before a demo is
   the single most damaging thing you could do here.
5. **If anything is ambiguous, stop and ask the human.** A wrong guess here silently deletes
   a teammate's work. Stopping costs a minute; guessing costs hours.

---

## File ownership

### Protected — the repo version is authoritative, always

If any of these show as **modified** in `git status`, Niharika's older zip copy has overwritten
a newer file. **Restore it; never commit it.**

```
backend/app/models/task_time.py
backend/app/models/readiness.py
backend/app/models/profile.py
backend/app/models/README.md
backend/training/train_time.py
backend/training/train_readiness.py
backend/training/build_profiles.py
backend/tools/replay_sim.py
backend/artifacts/task_time_model.joblib
backend/artifacts/readiness_weights.json
backend/artifacts/operator_profiles.json
backend/artifacts/operator_baselines.json
backend/MODELS.md
data/generate_data.py
data/datasets/README.md
data/HANDOFF_backend_teammate.md
docs/
ONBOARDING.md
MERGE_PLAN.md
AGENT_MERGE_INSTRUCTIONS.md
```

Restore any of them with:

```bash
git checkout origin/main -- <path>
```

**Why this matters concretely:** the zip predates a project decision. The anomaly detector
moved from **unsupervised Isolation Forest to supervised LightGBM**, because the project rule
is labelled data and supervised learning only. `data/datasets/README.md` and
`data/HANDOFF_backend_teammate.md` in the repo describe the supervised approach. The zip
copies still describe Isolation Forest. Committing the zip versions would re-introduce
instructions the team has already rejected.

### Niharika's — bring these in

New paths, so they merge cleanly:

```
backend/app/ingest/          ingestion gateway, WebSocket endpoint, raw logging
backend/app/state/           latest-state store, rolling buffer, derived counters
backend/app/rules/           safety rules engine, machine geometry config
backend/app/api/             REST routes, cab WebSocket
backend/app/db/              SQLite tables and queries
backend/app/models/anomaly.py
backend/app/decision/recommender.py
backend/training/train_anomaly.py
backend/artifacts/anomaly_*.joblib, anomaly_config.json
```

Her actual layout may differ; the principle is what matters: **files she authored come in,
everything else stays as the repo has it.**

### Reserved for Pranshu — do not create

```
backend/app/features/        feature builder      (in progress)
backend/app/decision/assemble.py   decision layer (in progress)
```

### Shared, may need appending

`backend/requirements.txt` — the repo already includes `websockets`, `httpx`,
`python-multipart`, `fastapi`, `uvicorn`, `pydantic`, `lightgbm`, `scikit-learn`.
**Append only** a dependency that is genuinely missing. Never rewrite or reorder the file.

`.gitignore` — keep the repo's version; append lines if needed. Note `data/datasets/*.csv`
and `*.parquet` are ignored on purpose (65 MB, reproducible).

---

## Procedure

### Step 1 — know where you are

```bash
git status
git log --oneline -3
git remote -v
```

If the working tree has uncommitted changes you did not make, **stop and ask the human**
before touching anything.

### Step 2 — get the current main

```bash
git fetch origin
```

### Step 3 — branch, before committing anything

```bash
git checkout -b backend2-anomaly
```

If the branch already exists, use it. **Do not commit on `main`.**

### Step 4 — restore every protected file

For each path in the protected list that `git status` shows as modified or deleted:

```bash
git checkout origin/main -- <path>
```

Then confirm none remain:

```bash
git status --porcelain | grep -E '^( M|M |MM|D )' || echo "clean: no protected files modified"
```

### Step 5 — verify before committing

```bash
git status --porcelain
```

**Expected:** only untracked (`??`) or added (`A`) entries, all under Niharika's paths above.

**Stop and ask the human if you see any of these:**
- a protected file listed as modified (` M`) or deleted (` D`)
- a deletion of any file you did not create
- more than ~40 changed files, which suggests a bulk overwrite
- any change under `docs/`

### Step 6 — commit and push

```bash
git add -A
git commit -m "Backend 2: anomaly detector, ingestion, state, safety rules, API"
git push -u origin backend2-anomaly
```

### Step 7 — open the pull request

Use the link GitHub prints, or `gh pr create --base main --head backend2-anomaly`. In the PR
description, list the files added and state explicitly that no protected file was modified.

**Then stop.** Pranshu reviews and merges. Do not merge your own PR.

---

## Known follow-up — do NOT do it in this PR

If `train_anomaly.py` currently uses `IsolationForest` or any other unsupervised method, it
**must** be changed to a supervised classifier — but on a **separate branch after this PR is
merged**. Keeping the file move and the algorithm change in one PR makes both impossible to
review.

For when that time comes: the dataset carries `is_anomaly` (binary) and `anomaly_label`
(18 classes) on 142k rows in `data/datasets/anomaly_windows_5min.csv`. Train LightGBM on the
labels, use `class_weight="balanced"` (only ~4.9% of windows are anomalous), split by time
(train Nov–Mar, test April — never randomly), and report precision/recall **per label**.
Details in `data/datasets/README.md` and `backend/MODELS.md`.

---

## If something has already gone wrong

Nothing is lost. Everything of Pranshu's is committed on `origin/main`, and a normal push
cannot overwrite it.

- Wrong files committed on the branch? Fix them on the branch and push again. The PR updates.
- Local tree a mess, nothing pushed yet? `git stash` her changes, `git checkout origin/main -- .`
  to restore, then re-apply only her files.
- Unsure at any point? **Stop and ask the human.** Do not attempt a clever recovery.

## Summary

Bring in her files. Leave everything else exactly as the repo has it. Verify with
`git status`. Branch, commit, push, open a PR, stop. Do not refactor anything.
