# Backend 2 (Niharika): next tasks

Everything you need is in this repo. You never have to wait for Pranshu, and he never has to
wait for you: you each own separate files (section 3), and the shared contracts are frozen
(section 4). Pull `main`, set up once (section 1), then pick tasks in order (section 2).

The backend is live on Render at `https://good-question-backend.onrender.com`. Every merge to
`main` redeploys it automatically, so **only merge work that passes the tests**.

---

## 1. One-time setup (≈5 min)

From the repo root:

```bash
git checkout main
git pull origin main
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r backend/requirements.txt pytest
.venv/bin/python data/generate_data.py
```

`generate_data.py` recreates the datasets in `data/datasets/` (~25 s). They are not in git,
and they come out byte-identical to Pranshu's copies (fixed seed).

Check that everything works:

```bash
cd backend
../.venv/bin/python -m pytest -q
```

Expect `21 passed`. To run the server locally:

```bash
cd backend
../.venv/bin/uvicorn app.main:app --reload --port 8000
```

In a second terminal, you can stream a real recorded shift into it:

```bash
.venv/bin/python backend/tools/replay_sim.py --shift EXC001-2025-04-01 --speed 600 --url ws://localhost:8000/ws/telemetry
```

Use `--list --anomaly fuel_theft` (or any label) to find shifts with a given anomaly.

## 2. Your tasks, in order

Each task lists the files you touch, what "done" means, and how to test it. All of them are
inside your own files, so do them in any order you like, but this order gives the demo the
most value first.

### B2-1. Multilingual alerts (demo-critical: our form promises it)

The cab UI speaks and shows alerts in the operator's language.

- New `backend/app/i18n/` package with message catalogues: English (`en`) and Hindi (`hi`)
  at minimum. Keys are `alert_type` / `anomaly_type` (see `app/rules/safety.py`,
  `app/rules/engine.py` and `docs/`), with `{placeholders}` for numbers
  (e.g. `"Worker {distance} m behind you, in blind spot"`).
- The cab socket takes a language: `ws://<host>/ws/cab/EXC001?lang=hi`, default `en`.
- For each connection, the hub translates `message`, `recommended_action` and anomaly
  `explanation` into that language before sending. It also adds `voice_text`: one short
  sentence per alert for the UI's text-to-speech.
- **Keep every existing key and type.** `lang` and `voice_text` are additions only.
  Tell the frontend team about both.
- Files: `app/i18n/*` (new), `app/api/cab_ws.py`, `app/decision/hub.py`,
  `app/rules/safety.py`, `app/rules/engine.py`.
- Done when: `?lang=hi` returns Hindi text for every alert type in the spec section 9
  scenarios, `?lang=xx` falls back to English, and tests pass.

### B2-2. Rule precision + per-label report for the pitch

Offline, the rules alone are too noisy: in `artifacts/evaluation_report.json`, binary
precision is ~0.20 at recall ~0.89.

- Tune `app/rules/thresholds.py` / `app/rules/engine.py` against
  `data/datasets/anomaly_windows_5min.csv`. Train on Nov–Mar, report on April, never a
  random split.
- **Known data limitation. Don't chase it:** the dataset only labels *injected*
  seatbelt episodes. Unbelted digging that happens naturally is labelled `normal`, even though
  the alert is correct. Report `seatbelt_off_while_moving` separately and exclude it from the
  overall precision target.
- Target: rules-only precision ≥ 0.5 with recall ≥ 0.85 (excluding the seatbelt label).
  Rules + model stays at or above the current numbers.
- Output: a per-label precision/recall table (rules only, model only, rules + model) in
  `evaluation_report.json`, plus a copy as `backend/artifacts/anomaly_metrics.md`, ready to
  paste into slides.
- Keep the model's `decision_threshold` (0.8) unless you re-derive it on April.
- Files: `app/rules/thresholds.py`, `app/rules/engine.py`, `training/train_anomaly.py`,
  `artifacts/anomaly_*`.
- Done when: the table exists, the targets are met, and tests pass. If you retrain, commit the
  new `anomaly_model.joblib` + `anomaly_config.json` together.

### B2-3. Training hub API (quiz, content, instructor booking)

The frontend's training hub needs content, not just module IDs.

- New file `backend/seed_data/training_content.json`: for each `module_id` in
  `seed_data/training_modules.json`, give 3 quiz questions (question, 3–4 options,
  correct index, one-line explanation) and a `video_url` placeholder.
  **Don't edit `training_modules.json`**: `data/generate_data.py` overwrites it.
- `GET /api/training/modules/{module_id}`: module + content.
- `POST /api/training/complete`: already exists. Also accept `answers: [int]`, grade them
  server-side, and store the score.
- `POST /api/training/book`: instructor booking
  `{operator_id, module_id, preferred_slot}` → a new `instructor_bookings` table →
  returns a booking id and a confirmed slot.
- `GET /api/training/bookings/{operator_id}`.
- Files: `app/api/rest.py`, `app/db/models.py`, `app/db/writes.py`, `seed_data/training_content.json`.
  New tables are created automatically on startup (`init_db`); no migration needed.
- Done when: all four endpoints work and have tests (B2-5).

### B2-4. Incident detail and alert acknowledgement over REST

- `GET /api/incidents/{id}`: one incident including its `telemetry_window`
  (the ±30 s snapshot that critical alerts store automatically).
- `GET /api/alerts?machine_id=EXC001&limit=50`: recent alerts, newest first.
- `POST /api/alerts/{alert_id}/ack`: same effect as the cab socket's
  `{"msg_type": "ack", "alert_id": ...}` (see `app/db/writes.py:acknowledge_alert`).
- Files: `app/api/rest.py`, `app/db/writes.py`.

### B2-5. API tests

- New `backend/tests/test_api.py` using FastAPI's `TestClient` against a temporary database:
  set `IRONSENSE_DB_PATH` to a temp file **before** importing `app.main`.
- Cover every REST endpoint in spec section 2 plus the new B2-3/B2-4 endpoints. Also cover one
  WebSocket round trip: send `shift_context` + `operation` + a close `proximity` on
  `/ws/telemetry`, then receive a critical alert on `/ws/cab/EXC001`.
- Files: `backend/tests/test_api.py` (new).

### B2-6 (stretch). Supervisor fleet view

- `GET /api/fleet/summary`: for every machine seen today, return the latest Readiness score,
  active alert count, open anomalies, and current task with p50 / remaining time.
  Read it from `app.state.hub.latest` (already kept in memory).
- Files: `app/api/rest.py`.

## 3. Who owns which file

You edit **only your files**. If you need a change in Pranshu's file, message him. Don't edit
it, even for a one-line fix.

| Owner | Files |
|---|---|
| **Niharika (Backend 2)** | `app/ingest/*`, `app/state/*`, `app/rules/*`, `app/api/rest.py`, `app/api/cab_ws.py`, `app/db/*`, `app/decision/*` (hub, layer, recommender), `app/models/anomaly.py`, `app/i18n/*` (new), `training/train_anomaly.py`, `artifacts/anomaly_*`, `artifacts/evaluation_report.json`, `seed_data/training_content.json` (new), `tests/test_api.py` (new) |
| **Pranshu (Backend 1)** | `app/features/*`, `app/models/task_time.py`, `app/models/readiness.py`, `app/models/profile.py`, `app/api/models_api.py`, `app/machines.py`, `app/paths.py`, `app/main.py`, `training/train_time.py`, `training/train_readiness.py`, `training/build_profiles.py`, `tools/*`, `data/*`, `artifacts/task_time_*`, `artifacts/readiness_*`, `artifacts/operator_*`, `Dockerfile`, `render.yaml`, `requirements.txt`, `tests/test_live_pipeline.py` |

- `app/main.py` is Pranshu's. If you add a **new router file**, ask him to add its one
  `include_router` line. Better: put new endpoints in `app/api/rest.py`, which is already included.
- Need a new Python package? Ask Pranshu to add it to `requirements.txt`. It changes the
  deployed image.

## 4. Frozen contracts (change only with both of you + the frontend team)

These are what the two halves and the frontend build against. Additions are fine if
existing keys keep their names and types; renames and removals are not.

1. **Telemetry messages:** `app/ingest/schemas.py` and `docs/frontend_handoff_spec_v1.md`
   section 5. The simulation sends exactly this.
2. **The `assessment` message:** spec section 10 (built in `app/decision/layer.py`).
   The cab UI reads exactly this.
3. **The feature row:** `FEATURE_COLUMNS` in `app/features/builder.py` (Pranshu's) =
   the columns of `anomaly_windows_5min.csv`. Your anomaly model is trained on it.
4. **The function interfaces you call:**
   - `predict_task(task, env, operator, machine, elapsed_min, progress_pct)`
   - `compute_readiness(row, anomalies)`
   - `operator_profile(operator_id)`

   Pranshu may improve what's inside them, but won't change these signatures or their return shapes.

## 5. Daily workflow

```bash
git checkout main && git pull origin main          # start of every session
git checkout -b b2/<task>                          # e.g. b2/i18n, b2/training-hub
# ... work ...
cd backend && ../.venv/bin/python -m pytest -q     # must pass before you push
git push origin b2/<task>                          # open a Pull Request on GitHub into main
```

- Merge your PR only when tests pass. **Merging to `main` redeploys the live server**, so
  don't merge while a Render deploy is still building, and never push broken code to `main`.
- After merging, check `https://good-question-backend.onrender.com/api/health` shows
  `{"status":"ok"}` once the deploy finishes. The free plan sleeps after 15 min idle; the
  first request takes ~1 min to wake it.
- If `git pull` brings in a new `requirements.txt`, re-run the `uv pip install` line from
  section 1. If it brings in a new `generate_data.py`, re-run it.
- Pranshu's branches are named `b1/<task>`, so you'll never collide on branch names.

## 6. Quick reference

| What | Where |
|---|---|
| Message formats, endpoints, scenarios | `docs/frontend_handoff_spec_v1.md` |
| Dataset columns, labels, leakage rules | `data/datasets/README.md` |
| How each model works | `backend/MODELS.md` |
| Live-pipeline tests (examples of driving the backend) | `backend/tests/test_live_pipeline.py` |
| Deployed API | `https://good-question-backend.onrender.com` (`wss://` for sockets) |
