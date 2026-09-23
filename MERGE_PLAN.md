# Merging Backend 2's work without breaking anything

Niharika built her half from a zip, so her folder has **no git history**. That makes this a
one-time import, not a normal merge. Done in the order below it is routine; done by copying
the whole folder over a clone it silently reverts newer files.

## The real risk is not conflicts

Git conflicts only happen when the *same file* changed on both sides, and git shouts about
them. Her work is almost entirely **new files at new paths**, which merge silently and
cleanly.

The danger is the opposite: her zip contains **older copies** of files that have since
changed here (the anomaly detector switched to supervised LightGBM). If she copies her whole
folder into a clone, git sees "Niharika edited these files" and takes her stale version. No
conflict, no warning, work quietly lost.

**So the rule is: she copies in only the files she wrote herself.**

## Who owns what

| Path | Owner | On merge |
|---|---|---|
| `backend/app/ingest/`, `state/`, `rules/`, `api/`, `db/` | Niharika | New files, merge cleanly |
| `backend/app/models/anomaly.py`, `decision/recommender.py` | Niharika | New files |
| `backend/app/decision/layer.py`, `backend/app/main.py` | Niharika | Hers: already built and working |
| `backend/training/train_anomaly.py` | Niharika | New file |
| `backend/artifacts/anomaly_*.joblib` | Niharika | New file |
| `backend/app/models/task_time.py`, `readiness.py`, `profile.py` | Pranshu | **She must not edit** |
| `backend/app/features/` | Pranshu | Being written now; plugs into her `decision/layer.py` |
| `backend/training/train_time.py`, `train_readiness.py`, `build_profiles.py` | Pranshu | |
| `backend/tools/replay_sim.py` | Pranshu | |
| `data/generate_data.py`, `data/datasets/README.md` | Pranshu | **Repo version is newer than her zip** |
| `docs/` | Shared, frozen | Change only by agreement |

## Files that genuinely can collide

Only four. Everything else is a new path.

| File | Why | Fix |
|---|---|---|
| `backend/requirements.txt` | Both add dependencies | **Already pre-empted**: `websockets`, `httpx`, `python-multipart` are added. She should only append if something is still missing. |
| `backend/app/__init__.py` | Both may have created it | Mine is empty; if hers is too, git merges it silently. |
| `.gitignore` | She may have her own | Keep the repo's. Append her lines if needed. |
| `data/datasets/README.md`, `data/HANDOFF_backend_teammate.md` | Her zip has the pre-supervised version | **Keep the repo's. Never take hers.** |

## Her import procedure (send her this)

```bash
# 1. fresh clone in a NEW folder - do not unzip anything into it
git clone https://github.com/PranshuRanjan10/Good_Question_backend.git
cd Good_Question_backend
git checkout -b backend2-anomaly

# 2. copy in ONLY your own files, e.g.
#    cp -r ~/old_zip_folder/backend/app/ingest      backend/app/
#    cp -r ~/old_zip_folder/backend/app/rules       backend/app/
#    cp    ~/old_zip_folder/backend/training/train_anomaly.py  backend/training/
#    ...and your artifacts:
#    cp    ~/old_zip_folder/backend/artifacts/anomaly_*.joblib backend/artifacts/

# 3. check what git thinks you changed BEFORE committing
git status
git diff --stat
```

**Step 3 is the whole safeguard.** `git status` should show only *new* files (`??` or `A`).
If it lists any file from the "Pranshu" rows above as *modified*, that's a stale copy from the
zip — restore it and re-check:

```bash
git checkout -- <that file>
```

Then:

```bash
git add -A && git commit -m "Backend 2: anomaly detector, ingestion, rules, API"
git push -u origin backend2-anomaly
```

Open a pull request. Do not push to `main`.

## Pranshu's review before merging

```bash
git fetch origin
git diff --stat main..origin/backend2-anomaly   # list every file she touched
```

Scan the list. **Any modified file outside her ownership rows is the thing to question** —
new files are fine, modified ones need a reason. Then merge the PR on GitHub.

## After that: normal workflow, no more drama

Both of you, every session:

```bash
git checkout main && git pull
git checkout -b <short-branch-name>
```

Commit small, push often, open a PR. Because you own different directories, you will almost
never touch the same file twice.

## If a conflict does happen

It'll be `requirements.txt` or similar, and it's a two-minute fix:

```bash
git pull origin main
```

Git marks the file with `<<<<<<<`, `=======`, `>>>>>>>`. Open it, keep **both** sets of lines
(for requirements that's almost always correct), delete the markers, then:

```bash
git add . && git commit -m "Resolve merge conflict" && git push
```

Never resolve a conflict in `backend/app/models/` or `data/generate_data.py` by taking the
incoming side without asking — that's where stale zip copies would show up.
