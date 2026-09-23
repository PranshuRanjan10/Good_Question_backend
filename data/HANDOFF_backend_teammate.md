# IronSense backend: teammate handoff (Backend 2)

The backend has two owners:

- **Backend 1 (Pranshu):** data generation, the task time estimator, the Readiness Score and the operator skill profile.
- **Backend 2 (you):** anomaly detection and the live backend service, which receives the telemetry and sends results to the cab UI.

We work in parallel and join up through the interfaces in section 4.

---

## 1. Files you get

Share **`ironsense_backend2_pack.zip`**, which contains:

| File | Why you need it |
|---|---|
| `datasets/anomaly_windows_5min.csv` | **Main training file**: supervised anomaly classifier plus rule thresholds (142k five-minute windows, 18 labelled anomaly types) |
| `datasets/hourly_summaries.csv` | Per-operator / per-machine **baselines** used in explanations ("idle 55 min vs your usual 22") |
| `datasets/minute_telemetry.parquet` | Raw one-minute data, for testing the safety rules engine and rebuilding features at other window sizes |
| `datasets/incidents.csv` | Seed data for the incidents table and the `/api/incidents` endpoint |
| `datasets/training_modules.json` | Training catalogue that the recommender maps anomalies to |
| `datasets/training_history.csv` | Seed data for `/api/training/*` |
| `datasets/machines.csv`, `datasets/operators.csv` | Master data; machine geometry for the rules goes on top of this |
| `datasets/README.md` | Column meanings, label vs feature columns, ground truth |
| `docs/telemetry_contract_v1.json`, `docs/telemetry_frequency_config_v1.json`, `docs/frontend_handoff_spec_v1.md` | Message formats the simulation sends; **the spec wins** where they differ |

## 2. Your tasks (in order)

### A. Anomaly detector (models)
1. **Rule engine:** one explainable rule per known pattern, with thresholds from the spec §7: idle > 20 min or idle ratio > 0.6, rpm > 2100, swing > 55 °/s, bucket-raised travel, speed > 6 km/h, pitch/roll > 15°, payload > 2000 kg × size, coolant > 105 °C, fuel drop with engine off, seat empty with engine on, belt off while moving, > 240 min continuous, outside scheduled hours, operator ≠ assigned, sensor dropout.
2. **Supervised classifier (LightGBM):** the project rule is **labelled data and supervised
   learning only**, so no Isolation Forest or other unsupervised method. The dataset carries
   `is_anomaly` and `anomaly_label` for all 18 types, so train on the labels directly:
   - Start binary on `is_anomaly`, then go multiclass on `anomaly_label` so the model **names**
     the anomaly instead of only flagging one.
   - Features: the recommended list in `datasets/README.md`.
   - Only ~4.9% of windows are anomalous, so pass `class_weight="balanced"` (or
     `scale_pos_weight`) and judge on precision/recall, never accuracy.
   - Scale features per machine size, or add `size_factor` from `machines.csv`.
   - Split by time: train Nov-Mar, test April. Never split randomly.
3. **Evaluate:** precision, recall and F1 **per `anomaly_label`**, comparing rules only,
   model only, and rules + model. Per-label matters because `fatigue` is by far the largest
   class and would otherwise flatter the overall number. Save a confusion table for the pitch.
   Note for the panel: production labels would come from confirmed incident reports and
   supervisor review, the same way ours come from the simulator.
4. **Explanations:** compare each flagged feature with the operator's baseline, built from `hourly_summaries.csv` medians per operator. Output 1–3 short sentences, e.g. `"Fuel per cycle 0.34 L vs your baseline 0.15 L"`. For idle anomalies with `truck_wait_min > 0`, add "likely waiting for haul truck".
5. Save `artifacts/anomaly_model.joblib` + `artifacts/anomaly_config.json` (thresholds, feature list, baselines).

### B. Training recommender
Map each anomaly or incident type to a module through `triggers` in `training_modules.json`. Recommend `TRN-INSTR-01` (instructor booking) when the same unsafe type repeats 3+ times in 7 days.

### C. Live backend service (FastAPI)
1. **Ingestion:** `ws /ws/telemetry`. Validate each `msg_type` with Pydantic, return an `error` message on bad input, and append raw messages to `logs/raw/YYYY-MM-DD.jsonl`.
2. **State manager:** latest state per machine, a 15-minute rolling buffer, derived counters (continuous operation, time since break, hourly compliance) and an hourly summary written to the DB.
3. **Safety rules (fast path):**
   - Danger and caution zones from distance and bearing, plus machine geometry. Widen them in rain, fog, dust, storm and at night: 5 m → 8 m, per the spec.
   - Seatbelt escalation: warning → alarm → logged violation.
   - Tilt, geofence, lightning, and refuelling with the engine on.
   - A critical alert creates an incident with ±30 s of telemetry from the buffer.
4. **Feature builder (every 60 s):** turn the buffer into **one row with exactly the columns of `anomaly_windows_5min.csv`**. This is the shared boundary; see section 4.
5. **Decision layer:**
   - Call my Readiness and task-time functions, plus your anomaly model.
   - Suppress repeat alerts within a cooldown.
   - Push `assessment` on `ws /ws/cab/{machine_id}` every 10 s, and immediately when there's a new alert.
6. **REST + DB (SQLite):** the endpoints in spec §2. Store events, alerts, incidents, hourly summaries, task records and model outputs.

**For the first demo, this slice is enough:** ingestion → state → proximity and seatbelt rules → `assessment` to the cab. Models plug in after that.

## 3. What Backend 1 (Pranshu) delivers to you
- `artifacts/task_time_model.joblib` + `predict_task(...)` (section 4)
- `artifacts/readiness_weights.json` + `compute_readiness(...)` (section 4)
- `artifacts/operator_baselines.json`: per-operator medians your anomaly explanations compare against
- `operator_profile(operator_id)`: a live skill profile from behaviour, which replaces the static Beginner / Intermediate / Expert label
- Dataset regenerations if you need new columns. Ask, and it's a quick change in `generate_data.py`.

## 4. Shared interfaces (agree on these, then work independently)

```python
# Feature row: dict with the same keys as a row of anomaly_windows_5min.csv
#   (built by YOU from the live buffer; used by both of us)

# ---- yours
def detect_anomalies(row: dict, baseline: dict) -> list[dict]:
    # -> [{"anomaly_type": "excessive_idling", "method": "rule+model",
    #      "score": 0.81, "explanation": ["Idle 55 min vs your baseline 22 min", "..."]}]

def recommend_training(anomalies: list[dict], incidents: list[dict], history: list[dict]) -> list[dict]:
    # -> [{"module_id": "TRN-IDLE-01", "title": "Cutting idle time", "reason": "excessive_idling"}]

# ---- mine (Pranshu)
def predict_task(task: dict, env: dict, operator: dict, machine: dict,
                 elapsed_min: float | None = None, progress_pct: float | None = None) -> dict:
    # -> {"task_id", "planned_min", "p10_min", "p50_min", "p90_min", "remaining_min",
    #     "factors": [{"factor": "weather_rainy", "effect_pct": 10}, ...]}

def compute_readiness(row: dict, anomalies: list[dict]) -> dict:
    # -> {"readiness_score": 64, "readiness_breakdown":
    #     {"seatbelt": 100, "proximity": 40, "behaviour": 70, "fatigue": 65, "conditions": 55}}
```

Output shapes must match `assessment` in `frontend_handoff_spec_v1.md` §10.

## 5. Folder ownership

```
backend/app/ingest/  state/  rules/  api/  db/  features/  models/anomaly.py  decision/recommender.py   -> you
backend/app/models/task_time.py  models/readiness.py  models/profile.py                              -> Pranshu
backend/training/train_anomaly.py                                                                  -> you
backend/training/train_time.py  train_readiness.py  data/generate_data.py                           -> Pranshu
backend/artifacts/                                                                                  -> shared
```

Message Pranshu before changing any column name or interface above.
