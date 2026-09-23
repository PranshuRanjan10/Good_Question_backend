# IronSense backend: my plan (Backend 1, Pranshu)

**I own:** data generation, the task time estimator, the Readiness Score, the operator skill profile and model packaging.
**Teammate (Backend 2) owns:** the anomaly detector, the training recommender and the live FastAPI service. See `HANDOFF_backend_teammate.md`.

## 1. My files

| File | Use |
|---|---|
| `datasets/task_records.csv` | **Task time estimator** training (7,037 tasks + organizers' T001–T005) |
| `datasets/readiness_training.csv` | **Readiness** logistic regression (label `incident_next_30min`, 1.4% positive) |
| `datasets/incidents.csv` | Readiness sanity checks, pitch numbers |
| `datasets/hourly_summaries.csv` + `anomaly_windows_5min.csv` | Behaviour inputs for the **operator skill profile** |
| `datasets/operators.csv`, `machines.csv`, `shifts.csv` | Features and joins |
| `datasets/operators_ground_truth.csv` | **Validation only**: does the skill profile rediscover the hidden traits? |
| `generate_data.py` | Regenerate / add columns on request |

## 2. Actions (in order)

### A. Task time estimator
1. Features known at task start only (list in `datasets/README.md`). Target: `log(actual / estimated)`.
2. Split by time: train Nov–Mar, test April. Keep T001–T005 in training with a higher weight.
3. Baseline: linear regression on one-hot skill, weather, ground, light and task type, plus age, temperature and trucks. This gives the explainable `factors` (effect_pct = exp(coef) − 1).
4. Main model: LightGBM with quantile objectives α = 0.1 / 0.5 / 0.9, giving p10 / p50 / p90.
5. Metrics: MAE in minutes vs. the planner's estimate, p10–p90 coverage (target ≈ 80%), and results broken down by skill and weather.
6. Live update: blend the prediction with the progress rate → `remaining_min` (at 0% progress use the model; as progress grows, weight observed pace more).
7. Save `artifacts/task_time_model.joblib` plus `predict_task(...)` (interface in the handoff doc §4).

### B. Readiness Score
1. Five sub-scores (0–100) from hand rules: seatbelt, proximity, behaviour (anomalies), fatigue, conditions (weather, light, visibility, heat).
2. Weights: start by hand, then fit logistic regression on `readiness_training.csv` (class_weight="balanced") and use its coefficients to set the weights. Report AUC and precision at the top 5%.
3. Save `artifacts/readiness_model.joblib` plus `compute_readiness(...)`.

### C. Operator skill profile (the "closed loop")
1. Score each operator per week: efficiency (task overrun vs. plan), compliance (belt, alerts), safety (incidents), smoothness (fast swing / harsh events).
2. Show that OP1004 and OP1009 improve over the six months, and check against `operators_ground_truth.csv`.
3. Feed the profile back into `predict_task` as an extra feature (compare with and without it; that result goes on a pitch slide).
4. `operator_profile(operator_id) -> {"skill_score", "level", "strengths", "focus_areas", "trend"}`.

### D. Packaging for the teammate
- Plain Python modules in `backend/app/models/` that load joblib at startup. No FastAPI code on my side.
- A short `models/README.md` with example calls and outputs.
- Metrics table + 2–3 charts for the pitch (overrun by skill/weather, p10–p90 band, readiness vs. incidents).

## 3. What I need from the teammate
- The live **feature row** builder (same columns as `anomaly_windows_5min.csv`) and the `detect_anomalies` output, both of which my Readiness function consumes.
- Where they call `predict_task` (task start, weather change, every 60 s).

## 4. Order of work
1. Linear task-time baseline → factors (fast win for the demo).
2. LightGBM quantiles + metrics.
3. Readiness (hand weights first, then logistic regression).
4. Skill profile + closed-loop comparison.
5. Charts and numbers for the pitch.
