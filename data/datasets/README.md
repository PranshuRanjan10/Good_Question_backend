# IronSense fabricated datasets (v1)

Synthetic history for **8 CAT machines × 12 operators**, **2024-11-01 → 2025-04-30** (Mon–Sat, ~1,170 shifts), simulated minute by minute. It ends the day before the demo day (2025-05-01).
Regenerate identically with `.venv/bin/python data/generate_data.py` (seed 42, ~25 s on an M4 Air).

Units, enums and value ranges follow `docs/telemetry_contract_v1.json` and `docs/frontend_handoff_spec_v1.md` §7, so the model inputs look like what the live simulation will send.

## Which file trains which model

| Model | File | Target | Notes |
|---|---|---|---|
| **Task time estimator** (LightGBM quantile + linear baseline) | `task_records.csv` | `actual_time_min / estimated_time_min` (predict log ratio) | 7,037 rows. First 5 rows are the organizers' T001–T005 verbatim (`source = organizer_sample`, extras empty). |
| **Anomaly detector** (supervised LightGBM + rules) | `anomaly_windows_5min.csv` | `is_anomaly` (binary) or `anomaly_label` (18-class) | 142k 5-min windows, ~4.9% anomalous. **Supervised**: train on the labels, report precision/recall per label. |
| **Readiness Score** (optional logistic regression) | `readiness_training.csv` | `incident_next_30min` (~1.4% positive) | Engine-on windows only. Use class weights. |
| **Operator baselines** (rolling stats) | `hourly_summaries.csv` | — | Per operator/machine medians of idle, fuel per cycle, cycles/hour, compliance. |
| **Training recommender** (rules) | `training_modules.json`, `training_history.csv` | — | Module catalogue with the anomaly/incident types that trigger each module. |
| Activity / anything custom | `minute_telemetry.parquet` | — | Raw 1-minute table everything else is built from. Rebuild features from here if you need a different window size. |

## Files

| File | Rows | What it is |
|---|---|---|
| `machines.csv` | 8 | EXC001–EXC006 (CAT 320/323/330/336), WL001 (CAT 950 loader), BHL001 (CAT 432 backhoe). EXC001 = the demo machine (CAT 320, 4 yrs, primary operator OP1001). |
| `operators.csv` | 12 | Public profile: `skill_level`, `experience_yrs`. **Use this one for features.** |
| `operators_ground_truth.csv` | 12 | Hidden traits the simulator used (belt discipline, aggressiveness, efficiency, improvement trend). **Never use as features**; it exists only to check that the models rediscover them (e.g. the skill profile). |
| `shifts.csv` | 1,173 | One row per machine-day: operator, shift times, night shift, spotter present, trucks assigned, weather, injected anomalies. |
| `task_records.csv` | 7,037 | Task records in the organizers' column format plus extras (see below). |
| `hourly_summaries.csv` | 12,528 | Hourly interval summaries: the organizers' telemetry columns plus derived metrics and labels. |
| `telemetry_organizer_format.csv` | 12,528 | The same hours with the **exact organizers' column names** (Timestamp, Machine ID, … Safety Alert Triggered), for slides and demos. |
| `anomaly_windows_5min.csv` | 142,450 | Feature rows for the anomaly detector (and the source of the readiness file). |
| `readiness_training.csv` | 127,937 | Safety-context features plus the `incident_next_30min` label. |
| `incidents.csv` | 331 | Incident / near-miss log: type, severity, cause, source (auto / operator / supervisor report) and context at the time. |
| `training_modules.json` | 13 | Training hub catalogue: `TRN-IDLE-01` … `TRN-INSTR-01`, format, duration, trigger types. |
| `training_history.csv` | 311 | Which operators were recommended or completed which module, with quiz scores. |
| `minute_telemetry.parquet` | 707,596 | Full 1-minute simulation output including ECU cumulative counters, per-minute safety alert flags and labels. |

## Feature vs label columns (avoid leakage)

**task_records.csv**
- Known at task start (use these): `task_type, weather, operator_skill, machine_age_yrs, estimated_time_min, operator_id, machine_id, machine_model, machine_type, time_of_day_hour, light, ground_condition, ambient_temp_c, rain_mm_h, wind_speed_kmh, visibility_m, haul_trucks_assigned, spotter_present, operator_experience_yrs, night_shift, target_volume_m3`
- Only known after the task (do NOT use as features): `actual_time_min, overrun_pct, idle_min_during_task, incidents_during_task, volume_moved_m3`
- Split by time (e.g. train Nov–Mar, test Apr), not randomly. Two beginners (OP1004, OP1009) get faster over the 6 months, so a time split is the honest test.

**anomaly_windows_5min.csv**
- Labels / leakage: `anomaly_label, anomaly_cause, is_anomaly, anomaly_min, incident, incident_next_30min, safety_alert_min`
- IDs / context: `machine_id, timestamp, operator_id, assigned_operator_id, task_id`
- Everything else is a feature. Recommended features: `idle_ratio, fuel_rate_lph, fuel_per_cycle_l, cycles_per_hour, rpm_max, over_rev_min, load_pct_mean, swing_rate_p95_dps, fast_swing_count, harsh_brake_count, ground_speed_max_kmh, bucket_raised_travel_s, pitch_max_deg, roll_max_deg, bucket_payload_max_kg, overload_count, coolant_max_c, hydraulic_oil_max_c, fuel_drop_engine_off_pct, seat_empty_engine_on_min, seatbelt_off_moving_min, continuous_operation_min, within_scheduled_hours, operator_matches_assigned, sensor_dropout_min`
- Engine-off windows are included, so fuel theft is detectable. `min_person_distance_m = 99` means nobody within 20 m.
- `fuel_per_cycle_l` treats 0 cycles as 1, so an idle window shows its full fuel burn.

**Class balance:** ~4.9% of windows are anomalous, so use `class_weight="balanced"` (or `scale_pos_weight`) and judge on precision/recall per label, never accuracy.

**readiness_training.csv**: label `incident_next_30min`. `is_anomaly` and `anomaly_label` are allowed as inputs (at runtime they come from the anomaly detector).

## Anomaly labels (`anomaly_label`)

| Group | Labels |
|---|---|
| Behaviour | `excessive_idling` (cause: waiting for haul truck / site instructions / engine left running on a personal break), `seatbelt_off_while_moving`, `operator_out_of_seat`, `fast_swing`, `harsh_operation`, `bucket_raised_travel`, `overspeed`, `slope_exceeded`, `overload`, `low_productivity`, `over_rev`, `fatigue` (> 240 min continuous operation) |
| Machine health | `overheating` (coolant > 105 °C, fault `COOLANT_HIGH_TEMP`) |
| Security / operational | `fuel_theft` (fuel falls with engine off), `after_hours_use`, `unauthorized_operator` (operator ID `OP9xxx` doesn't match the assigned operator), `unsafe_refuelling` (engine running while refuelling) |
| Data quality | `sensor_dropout` (NaN rpm / coolant / fuel level) |

`fatigue` is the largest class (~3.9k windows) because it covers long stretches; report metrics per label, not only overall.

## Ground truth built into the data (what the models should find)

**Task time** (log multiplier on the plan): skill Expert −5%, Intermediate +5%, Beginner +26% (+5% more on Loading/Demolition); weather Rainy +7%, Windy +3%, Fog +11%, Extreme Heat +7%, Dust +5%, Storm +30%; ground wet +2%, muddy +9%, loose +3%; dusk +4%, night +8%; machine age +0.6%/yr; heat above 35 °C +1%/°C; Demolition in wind +6%; Grading in rain +6%. Noise is larger for beginners and bad weather, so the p10–p90 range should widen there. Truck tasks add waits that grow as `haul_trucks_assigned` falls.
Result: median overrun Expert −4%, Intermediate +15%, Beginner +45%, which matches the organizers' T001–T005.

**The organizers' seatbelt / idle pattern** is reproduced: operators unbuckle during long idle waits (mostly waiting for haul trucks).

| Idle min in hour | Median load cycles | Median fuel per cycle (L) | Hours "Unfastened" |
|---|---|---|---|
| ≤ 10 | 59 | 0.22 | 7% |
| 10–25 | 86 | 0.14 | 22% |
| 25–40 | 56 | 0.16 | 57% |
| 40–60 | 18 | 0.34 | 54% |

(The ≤ 10 bucket includes partial hours at shift start and end, which lowers its cycle count.)

**Incidents** are more likely with: a person in the danger zone (5 m, widened to 8 m in rain, fog, dust, storm or at night), seatbelt off while moving, > 4 h continuous operation, bad weather, night, harsh braking, fast swing, beginner operator, tilt > 15°, overspeed, cab > 35 °C, or a person in the blind spot.

## Known differences from the organizers' sample
- **Load cycles** here are bucket dumps (per our contract: one dig-swing-dump cycle ≈ 15–25 s), so a working hour has ~60–150. The organizers' sample (1–12 per interval) looks more like truck loads, so `hourly_summaries.csv` also has `truck_loads` (≈ cycles / 6).
- **Fuel** follows real 20 t excavator rates (idle 2–4 L/h, working 10–20 L/h). The organizers' sample has lower absolute numbers, but the ratio between idle-heavy and normal hours holds.
- Weather follows a north-Indian winter/spring (fog Dec–Jan, heat and dust in April). Change `MONTH_WEATHER_P` in the generator for another region.
