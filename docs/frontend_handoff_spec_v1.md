# IronSense: Frontend & Simulation Handoff Spec (v1)

This is everything the simulation and cab UI need to talk to the backend. Attach with it: `telemetry_contract_v1.json` (full field list and allowed values) and `telemetry_frequency_config_v1.json`. **Where this document and the JSON files differ, this document wins.**

## 1. The big picture

- The simulation runs the site and the controlled excavator (EXC001, a CAT 320).
- It sends raw sensor-style data to the backend as JSON over a WebSocket.
- The backend decides everything intelligent: danger zones, alert severity, anomalies, time predictions, Readiness Score, training suggestions.
- The backend pushes results back to the cab UI over a second WebSocket, and the UI fetches screen data (tasks, incidents, digest) over REST.

Rule of thumb: the simulation never decides whether something is dangerous. It just reports what the sensors would see.

## 2. Connections

Base URL during development: `http://<backend-host>:8000`.

| Purpose | Type | Endpoint |
|---|---|---|
| Send all telemetry and events | WebSocket (send) | `ws://<host>:8000/ws/telemetry` |
| Receive live alerts, scores, predictions | WebSocket (receive) | `ws://<host>:8000/ws/cab/EXC001` |
| Today's tasks | GET | `/api/tasks/today?operator_id=OP1001` |
| Incident list | GET | `/api/incidents?operator_id=OP1001` |
| Report manual incident / near miss | POST | `/api/incidents` |
| End-of-shift digest | GET | `/api/digest/OP1001` |
| Training recommendations | GET | `/api/training/recommendations/OP1001` |
| Mark training completed | POST | `/api/training/complete` |
| Readiness history (for charts) | GET | `/api/readiness/history?machine_id=EXC001` |
| Health check | GET | `/api/health` |

If a message is invalid, the backend replies on the telemetry socket with:

```json
{ "msg_type": "error", "ref_msg_type": "operation", "sim_tick": 5421, "detail": "engine.rpm must be a number" }
```

## 3. Conventions (important)

- Every message carries: `msg_type`, `schema_version` ("1.0"), `timestamp`, `sim_tick`, `machine_id`.
- Timestamps are ISO 8601 UTC in sim time, e.g. `"2025-05-01T10:14:07Z"`.
- `sim_tick` increases by 1 every sim second and never goes backwards.
- Sim clock speed is configurable. Send `time_scale` in `shift_context` (e.g. 60 = 1 real second is 1 sim minute). Use 10–60 for demos.
- Units are in the field name (`_m, _kmh, _c, _pct, _l, _lph, _deg, _dps, _kg, _min`).
- Never omit a key. If a value doesn't exist, send `null`.
- Coordinates: site-local metres. Origin (0,0) at the bottom-left corner of the site, x = east, y = north. Site is roughly 400 m × 300 m.
- Angles: `heading_deg` 0 = north, clockwise, 0–359. `bearing_deg` for proximity objects is relative to the machine's front (0 = straight ahead, 180 = directly behind), clockwise.
- Random seed: the sim should accept a seed so every scenario can be replayed identically.

## 4. What to send and how often (sim time)

| msg_type | When to send | Notes |
|---|---|---|
| shift_context | Once at sim start / operator login. Resend only if something changes. | Machine, operator, tasks, site map, zones, hazards, time_scale |
| operation | Every 10 s | Every 2 s if a person is within 20 m. Every 30 s if idle > 2 min. Stop when engine is off. |
| motion_batch | Every 10 s, containing 10 samples taken 1 s apart | Only while engine is on |
| proximity | Every 1 s only while any object is within 20 m (35 m in Storm, Fog or at night) | When everything leaves the range, send one message with an empty list, then stop |
| environment | Every 15 min, and immediately when weather changes | Also send a weather_change event |
| status | Every 60 s (every 300 s when engine is off) | Also acts as a heartbeat. Includes the cumulative counters. |
| event | Immediately when it happens | See section 6 |

Do NOT send `interval_summary` or `task_record`. The backend builds those itself. No deadbands for now: always send full messages at the rates above.

## 5. Message formats

### 5.1 shift_context (once)

```json
{
  "msg_type": "shift_context",
  "schema_version": "1.0",
  "timestamp": "2025-05-01T07:30:00Z",
  "sim_tick": 0,
  "machine_id": "EXC001",
  "time_scale": 60,
  "scenario_id": "demo_rainy_trench",
  "seed": 42,
  "site_id": "SITE01",
  "machine": { "machine_id": "EXC001", "model": "CAT 320", "machine_type": "excavator", "machine_age_yrs": 4 },
  "operator": { "operator_id": "OP1001", "skill_level": "Intermediate", "shift_start": "2025-05-01T07:30:00Z" },
  "daily_tasks": [
    { "task_id": "T002", "task_type": "Trenching", "zone_id": "ZONE_B", "planned_estimate_min": 45,
      "target_volume_m3": 36.9, "scheduled_start": "2025-05-01T08:00:00Z" }
  ],
  "zones": [
    { "zone_id": "ZONE_B", "zone_type": "work_area", "polygon": [[180,60],[260,60],[260,120],[180,120]] },
    { "zone_id": "LOAD_BAY", "zone_type": "loading_bay", "polygon": [[270,70],[300,70],[300,100],[270,100]] },
    { "zone_id": "NOGO_1", "zone_type": "no_go_zone", "polygon": [[320,200],[360,200],[360,240],[320,240]] }
  ],
  "hazards": [
    { "hazard_id": "PL01", "hazard_type": "overhead_power_line", "line": [[150,140],[300,140]], "clearance_height_m": 9.5 },
    { "hazard_id": "TR01", "hazard_type": "trench_edge", "line": [[190,90],[250,90]] },
    { "hazard_id": "UG02", "hazard_type": "underground_utility_gas", "line": [[200,70],[240,70]] }
  ],
  "walkaround_checklist_result": { "completed": true, "issues": [] }
}
```

### 5.2 operation (every 10 s)

```json
{
  "msg_type": "operation",
  "schema_version": "1.0",
  "timestamp": "2025-05-01T10:14:10Z",
  "sim_tick": 9850,
  "machine_id": "EXC001",
  "operator_id": "OP1001",
  "task_id": "T002",
  "engine": { "state": "running", "rpm": 1650, "load_pct": 62, "fuel_rate_lph": 14.8 },
  "motion": {
    "position": { "x_m": 212.4, "y_m": 88.1 },
    "heading_deg": 274, "ground_speed_kmh": 0.0, "travel_direction": "stationary",
    "parking_brake": false, "hydraulic_lockout": false, "travel_alarm_active": false
  },
  "implement": { "work_mode": "dig", "hydraulic_pressure_bar": null },
  "cab": { "seat_occupied": true, "seatbelt_fastened": true, "door_open": null, "controls_active": true }
}
```

### 5.3 motion_batch (every 10 s, 10 samples)

```json
{
  "msg_type": "motion_batch",
  "schema_version": "1.0",
  "timestamp": "2025-05-01T10:14:10Z",
  "sim_tick": 9850,
  "machine_id": "EXC001",
  "start_timestamp": "2025-05-01T10:14:00Z",
  "sample_interval_s": 1,
  "fields": ["pitch_deg", "roll_deg", "longitudinal_accel_ms2", "swing_angle_deg", "swing_rate_dps", "bucket_height_m", "bucket_payload_kg"],
  "samples": [
    [4.2, 1.1, 0.0, 30, 15, 1.2, 1400],
    [4.2, 1.1, 0.0, 35, 18, 1.2, 1450]
  ]
}
```

Each row follows the order in `fields`. Positive swing rate = clockwise.

### 5.4 proximity (1 s, only when something is near)

```json
{
  "msg_type": "proximity",
  "schema_version": "1.0",
  "timestamp": "2025-05-01T10:14:07Z",
  "sim_tick": 9847,
  "machine_id": "EXC001",
  "objects": [
    { "object_id": "W03", "object_type": "person", "role": "labourer", "distance_m": 6.8,
      "bearing_deg": 190, "relative_speed_ms": -0.6, "sensor": "radar", "detection_confidence": null }
  ]
}
```

`relative_speed_ms` negative = moving closer. Do not send `in_blind_spot` or zone colours; the backend computes them.

### 5.5 environment (15 min + on change)

```json
{
  "msg_type": "environment",
  "schema_version": "1.0",
  "timestamp": "2025-05-01T10:00:00Z",
  "sim_tick": 9000,
  "machine_id": "EXC001",
  "environment": {
    "weather": "Rainy", "ambient_temp_c": 27, "humidity_pct": null, "rain_mm_h": 6.5,
    "wind_speed_kmh": 18, "wind_gust_kmh": 32, "visibility_m": 400, "light": "day",
    "dust_index": null, "ground_condition": "wet", "lightning_distance_km": null
  }
}
```

### 5.6 status (60 s, heartbeat)

```json
{
  "msg_type": "status",
  "schema_version": "1.0",
  "timestamp": "2025-05-01T10:15:00Z",
  "sim_tick": 9900,
  "machine_id": "EXC001",
  "engine": {
    "coolant_temp_c": 88, "fuel_level_pct": 54,
    "cumulative_hours": 1524.3, "cumulative_idle_hours": 312.7,
    "cumulative_fuel_used_l": 98234.5, "def_level_pct": null, "battery_voltage_v": null
  },
  "implement": { "hydraulic_oil_temp_c": 64, "cumulative_load_cycles": 48213 },
  "cab": { "cab_temp_c": 31 },
  "task": { "task_id": "T002", "progress_pct": 38.5, "volume_moved_m3": 14.2 },
  "diagnostics": { "active_fault_codes": [] }
}
```

The backend works out continuous operating time and time since break from the data and `break_start` / `break_end` events.

### 5.7 event (immediately)

```json
{
  "msg_type": "event",
  "schema_version": "1.0",
  "timestamp": "2025-05-01T10:15:30Z",
  "sim_tick": 9930,
  "machine_id": "EXC001",
  "operator_id": "OP1001",
  "event_id": "EV-0042",
  "event_type": "seatbelt_unfastened",
  "source": "sensor",
  "details": {}
}
```

`source` is `"sensor"` for normal sim behaviour, `"director_console"` when injected by a button, `"operator"` for manual reports.

## 6. Events the simulation must emit

| event_type | Emit when | details |
|---|---|---|
| engine_start / engine_stop | Engine state changes | `{}` |
| seatbelt_fastened / seatbelt_unfastened | Belt changes | `{}` |
| operator_left_seat / operator_returned | Seat occupancy changes | `{}` |
| geofence_enter / geofence_exit | Machine enters/leaves any zone | `{ "zone_id": "NOGO_1" }` |
| harsh_brake | Deceleration beyond ~2.5 m/s² | `{ "accel_ms2": -3.1 }` |
| collision | Contact with any object | `{ "object_id": "TRK02" }` |
| tilt_warning | Pitch or roll above ~15° | `{ "pitch_deg": 17.2, "roll_deg": 4.0 }` |
| fault_code | New fault appears | `{ "code": "COOLANT_HIGH_TEMP" }` |
| refuel_start / refuel_end | Refuelling begins/ends | `{ "litres": 120 }` on end |
| task_start / task_complete | Task begins/ends | `{ "task_id": "T002", "volume_moved_m3": 36.9 }` |
| break_start / break_end | Operator break | `{}` |
| weather_change | Weather category changes | `{ "from": "Cloudy", "to": "Rainy" }` |
| lightning_nearby | Lightning within 10 km | `{ "distance_km": 6 }` |
| walkaround_completed | Pre-shift checklist done | `{ "issues": [] }` |
| manual_near_miss / manual_incident | Operator reports (UI button or voice) | `{ "note": "Worker walked behind me" }` |

## 7. Realistic value ranges (~20 t excavator)

| Field | Engine off | Idle | Working | Abnormal (anomaly scenarios) |
|---|---|---|---|---|
| engine.rpm | 0 | 800–1000 | 1500–1900 | > 2100 over-rev |
| engine.fuel_rate_lph | 0 | 2–4 | 10–20 | — |
| engine.load_pct | 0 | 5–15 | 40–85 | > 95 sustained |
| engine.coolant_temp_c | ambient | 75–90 | 80–95 | > 105 overheating |
| implement.hydraulic_oil_temp_c | ambient | 40–60 | 50–80 | > 95 |
| motion.ground_speed_kmh | 0 | 0 | 0–5.5 (travel) | > 6 on site |
| implement.swing_rate_dps | 0 | 0 | 10–40 | > 55 fast swing |
| implement.bucket_payload_kg | 0 | 0 | 0–1800 | > 2000 overload |
| motion.pitch_deg / roll_deg | — | — | 0–10 | > 15 tilt risk |
| cab.cab_temp_c | ambient | 22–28 (AC) | 22–30 | > 35 heat stress |

One work cycle (dig → swing loaded → dump → swing empty) takes about 15–25 s. Each completed dump adds 1 to `cumulative_load_cycles`.

## 8. Simulation world checklist

- **Site:** work zones, loading bay, spoil pile, trench, haul road, overhead power line, marked underground utility, a slope, no-go zone, site office, gate, fuel station, parking. City outside the gate is background.
- **Controlled excavator (EXC001):** keyboard control; autopilot mode running dig-swing-dump cycles; toggles for seatbelt, seat occupied, engine, parking brake.
- **NPC agents:** labourers, spotter, surveyor, dump trucks (sometimes late), one other machine on the haul road, occasional visitor, fuel bowser.
- **Weather system:** Sunny, Cloudy, Rainy, Windy, Storm, Fog, Extreme Heat, Dust, plus day/dusk/night. Weather must change the data, not just visuals.
- **Record & replay:** save every outgoing message to `.jsonl` and play it back at recorded timing.

## 9. Director console scenarios (also our test cases)

| # | Scenario | What the sim does | Expected backend response |
|---|---|---|---|
| 1 | Worker in blind spot | Labourer walks behind machine (bearing ~180°, closing to < 5 m) while swinging | Critical proximity alert, incident created |
| 2 | Seatbelt off while moving | seatbelt_unfastened event, then machine travels | Warning → alarm → logged violation |
| 3 | Left cab, engine running | operator_left_seat, engine stays running | High alert |
| 4 | Truck delayed, long idle | No truck in loading bay for 20+ sim min, machine idles, operator unbuckles | Idle anomaly: "likely waiting for haul truck", training suggestion |
| 5 | Rain starts | weather_change to Rainy, ground wet, visibility drops | Danger zones widen, task prediction goes up |
| 6 | Lightning | lightning_nearby at 6 km | Stop-work alert |
| 7 | Steep slope | Drive onto slope, pitch past 15° | Tilt warning |
| 8 | Unsafe operation | Swing rate > 55 °/s, or travel with bucket height > 2.5 m | Unsafe operation anomaly |
| 9 | Fuel theft | Engine off, fuel_level_pct drops 15% over 10 min | Security anomaly |
| 10 | Overheating | Coolant > 105 °C, fault_code event | Machine health alert |
| 11 | Fatigue | Operate 4+ sim hours with no break_start | Fatigue warning, Readiness drops |
| 12 | Near miss report | UI button / voice → manual_near_miss event | Logged incident, shows in incident list |

## 10. What the cab UI receives

`assessment` messages on `ws://<host>:8000/ws/cab/EXC001` (about every 10 s, and immediately on a new alert):

```json
{
  "msg_type": "assessment",
  "timestamp": "2025-05-01T10:15:31Z",
  "machine_id": "EXC001",
  "readiness_score": 64,
  "readiness_breakdown": { "seatbelt": 100, "proximity": 40, "behaviour": 70, "fatigue": 65, "conditions": 55 },
  "zones": { "danger_radius_m": 8.0, "caution_radius_m": 15.0, "reason": "rain, low visibility" },
  "proximity_view": [
    { "object_id": "W03", "object_type": "person", "distance_m": 6.8, "bearing_deg": 190, "zone": "red", "in_blind_spot": true }
  ],
  "alerts": [
    { "alert_id": "AL-0107", "alert_type": "proximity_person_blind_spot", "severity": "critical",
      "message": "Worker 6.8 m behind you, in blind spot", "recommended_action": "Stop swing and sound horn" }
  ],
  "anomalies": [
    { "anomaly_type": "excessive_idling", "score": 0.81,
      "explanation": ["Idle 55 min vs your baseline 22 min", "No truck in loading bay: likely waiting for haul truck"] }
  ],
  "task_prediction": {
    "task_id": "T002", "planned_min": 45, "p10_min": 47, "p50_min": 52, "p90_min": 58, "remaining_min": 31,
    "factors": [ { "factor": "weather_rainy", "effect_pct": 10 }, { "factor": "skill_intermediate", "effect_pct": 5 } ]
  },
  "training_recommendations": [ { "module_id": "TRN-IDLE-01", "title": "Cutting idle time", "reason": "excessive_idling" } ]
}
```

| Severity | Display |
|---|---|
| info | Small banner, no sound |
| warning | Amber banner, single chime |
| high | Red banner, repeating tone until acknowledged |
| critical | Full-width red overlay + alarm sound, stays until condition clears |

Alerts with the same `alert_id` are updates of the same alert. While the machine is moving (ground_speed_kmh > 0 or work_mode not idle/break), the UI shows only the glanceable view; training, incident history and digest screens unlock when idle or parked.

## 11. Priority order for the first demo

1. shift_context, operation, proximity and event messages flowing to the backend.
2. Controlled excavator + one labourer NPC walking behind it (scenario 1).
3. Seatbelt toggle (scenario 2).
4. Cab UI showing the proximity ring and alerts from assessment.
5. Record & replay.

Message the backend team before changing any field name; the backend validates every field.
