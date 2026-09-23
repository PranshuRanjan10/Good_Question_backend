# Models: how to call them (Backend 2 integration guide)

Three functions, all pure: dicts in, dicts out. No FastAPI, no DB, no network. Load the
artifacts once at startup, then call them per message.

Everything here is already trained. Artifacts live in `backend/artifacts/`.

## Startup

Load once, so a missing or corrupt artifact fails loudly at boot instead of mid-demo:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.models import task_time, readiness, profile

@asynccontextmanager
async def lifespan(app: FastAPI):
    task_time.load()          # artifacts/task_time_model.joblib  (~3.5 MB)
    readiness.load_weights()  # artifacts/readiness_weights.json
    profile.load()            # artifacts/operator_profiles.json
    yield

app = FastAPI(lifespan=lifespan)
```

`readiness` and `profile` fall back to sensible defaults if their artifact is missing, so they
never crash the app. `task_time.load()` will raise — which is what you want, since a silent
fallback there would mean quietly shipping wrong predictions.

---

## 1. `predict_task(...)` — the task_prediction block

Call it at **task start**, on every **weather change**, and **once a minute** while a task runs.

```python
from app.models.task_time import predict_task

predict_task(
    task={"task_id": "T002", "task_type": "Trenching", "planned_estimate_min": 45,
          "target_volume_m3": 36.9, "haul_trucks_assigned": 2},
    env={"weather": "Rainy", "ambient_temp_c": 27, "rain_mm_h": 6.5, "wind_speed_kmh": 18,
         "visibility_m": 400, "light": "day", "ground_condition": "wet"},
    operator={"operator_id": "OP1001", "skill_level": "Intermediate"},
    machine={"machine_id": "EXC001", "machine_age_yrs": 4},
    elapsed_min=21, progress_pct=38.5,
)
```

Real output:

```json
{
  "task_id": "T002",
  "planned_min": 45,
  "p10_min": 50,
  "p50_min": 57,
  "p90_min": 62,
  "remaining_min": 34,
  "factors": [
    {"factor": "task_trenching", "effect_pct": -8},
    {"factor": "weather_rainy", "effect_pct": 4},
    {"factor": "ground_wet", "effect_pct": 2}
  ]
}
```

Drop this straight into `assessment.task_prediction` — the shape already matches the spec.

- **Omit `elapsed_min` / `progress_pct`** before a task starts; `remaining_min` comes back
  `null` and the p10/p50/p90 are the pure forecast.
- **Pass both while it runs** and `remaining_min` blends the forecast with the pace actually
  observed. Early progress is noisy, so the model dominates until the task is properly under
  way; past ~50% done, observed pace takes over.
- **Every field except `task_type` and `planned_estimate_min` is optional.** Missing values
  get reasonable defaults, so a partial state object won't throw.
- `factors` is already sorted by size and capped at 4, ready to display.

---

## 2. `compute_readiness(...)` — the readiness block

Call it **every 10 s**, on the same feature row you build for the anomaly detector.

```python
from app.models.readiness import compute_readiness

compute_readiness(row, anomalies)   # row = your live feature row; anomalies = detect_anomalies(...)
```

Real output:

```json
{
  "readiness_score": 50,
  "readiness_breakdown": {
    "seatbelt": 100, "proximity": 25, "behaviour": 90, "fatigue": 100, "conditions": 80
  },
  "drivers": [
    "Person 6.8 m away, inside the 8 m danger zone",
    "Person in a blind spot",
    "Rainy conditions"
  ]
}
```

- `readiness_score` and `readiness_breakdown` go into `assessment` as-is.
- **`drivers` is an extra I added** beyond the spec: plain-language reasons the score dropped,
  already ordered by importance. Show them under the score in the cab — it turns a number
  nobody trusts into one they do. Ignore it if the UI has no room.
- **One safety rule is built in:** the overall score can never sit more than 25 points above
  its worst sub-score. Without that, four healthy sub-scores would average away a worker
  standing in the danger zone. That's why the example reads 50 and not 74.
- `anomalies` can be `None` or `[]`; the behaviour sub-score just loses that input.
- It reads the same field names as `anomaly_windows_5min.csv`, so if your feature row matches
  that schema, it works with no mapping.

---

## 3. `operator_profile(...)` — the living skill profile

Call it at **shift start**, and after a training module completes.

```python
from app.models.profile import operator_profile, skill_label

operator_profile("OP1004")
```

Real output:

```json
{
  "operator_id": "OP1004",
  "skill_score": 31.9,
  "level": "Beginner",
  "trend": "steady",
  "strengths": [],
  "focus_areas": ["Tasks running over the planned time", "Harsh braking and fast swings"],
  "components": {"efficiency": 26.0, "compliance": 39.8, "safety": 32.2, "smoothness": 30.2},
  "as_of": "2025-04-28",
  "weeks_of_history": 27
}
```

- `focus_areas` feeds your **training recommender** — it's the operator's two weakest areas in
  words, so you can map straight to modules.
- `skill_label(operator_id, static_label)` returns what to pass to `predict_task` as
  `skill_level`: the live level once there are 2+ weeks of history, otherwise the roster label.
  An unknown operator returns the static label you pass in, so it's always safe to call.
- `profile.history(operator_id)` returns weekly `skill_score` points for a trend chart.
- An operator with no history gets a neutral Intermediate profile rather than an error.

**Read `level` and `trend` as separate things.** `level` is where someone stands now; `trend`
is which way they're moving. OP1004 is a Beginner (efficiency 26) whose task speed is
improving faster than anyone else in the fleet — that combination is exactly the closed-loop
story, and it's why both fields exist.

---

## Where this sits in your assessment message

```python
assessment = {
    "msg_type": "assessment",
    "timestamp": now_iso(),
    "machine_id": machine_id,
    **compute_readiness(row, anomalies),        # readiness_score, readiness_breakdown, drivers
    "zones": zones,                              # yours (safety rules)
    "proximity_view": proximity_view,            # yours
    "alerts": alerts,                            # yours
    "anomalies": anomalies,                      # yours
    "task_prediction": predict_task(...),        # mine
    "training_recommendations": recommend_training(...),  # yours
}
```

`drivers` is an extra key; strip it if you'd rather keep the message exactly to spec.

## Performance

All three are microseconds to low milliseconds per call — no threading or caching needed at
10 Hz. `predict_task` is the heaviest (three LightGBM models), so don't call it per telemetry
tick; once a minute per active task is plenty.

## If something looks wrong

- Predictions identical for every task → the feature row isn't varying; print the dict you're
  passing.
- `readiness_score` stuck at 100 → your row's keys don't match `anomaly_windows_5min.csv`
  column names; every lookup is silently defaulting.
- `FileNotFoundError` on `task_time.load()` → run `backend/training/train_time.py`.

Message me before changing any key name in these interfaces.
