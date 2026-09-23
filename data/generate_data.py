#!/usr/bin/env python3
"""
IronSense - fabricated training data generator.

Simulates 6 months (2024-11-01 .. 2025-04-30) of shifts for 8 CAT machines and
12 operators at 1-minute resolution, then derives every dataset the models need.
The history ends the day before the demo day (2025-05-01).

Run:  .venv/bin/python data/generate_data.py            (seed 42, ~1 min)
Out:  data/datasets/*.csv|*.parquet|*.json   (see data/datasets/README.md)

Calibration targets (organizers' sample data):
  * Task table: experts in good weather beat the plan, beginners / rain / wind overrun it
    (T001 -3%, T002 +16%, T003 +40%, T004 -6%, T005 +17%).
  * Telemetry table: idle-heavy intervals coincide with unfastened seatbelts,
    few load cycles and ~3-4x worse fuel per load cycle.
Physical ranges follow docs/frontend_handoff_spec_v1.md section 7, so the
synthetic history matches what the live simulation will send.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START_DATE = pd.Timestamp("2024-11-01")
END_DATE = pd.Timestamp("2025-04-30")
OUT = Path(__file__).resolve().parent / "datasets"

rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------- master data
MACHINES = [
    # machine_id, model, type, age, size factor (fuel/payload/output), tank L, primary operator
    ("EXC001", "CAT 320", "excavator", 4, 1.00, 345, "OP1001"),
    ("EXC002", "CAT 320", "excavator", 2, 1.00, 345, "OP1002"),
    ("EXC003", "CAT 323", "excavator", 6, 1.10, 345, "OP1003"),
    ("EXC004", "CAT 330", "excavator", 9, 1.40, 475, "OP1004"),
    ("EXC005", "CAT 336", "excavator", 1, 1.55, 600, "OP1005"),
    ("EXC006", "CAT 320", "excavator", 5, 1.00, 345, "OP1008"),
    ("WL001", "CAT 950", "wheel_loader", 5, 1.00, 370, "OP1006"),
    ("BHL001", "CAT 432", "backhoe_loader", 7, 0.55, 160, "OP1007"),
]
MACHINES = [dict(zip(["machine_id", "model", "machine_type", "machine_age_yrs", "size_factor",
                      "fuel_tank_l", "primary_operator_id"], m)) for m in MACHINES]

SKILL_EFFECT = {"Expert": -0.05, "Intermediate": 0.05, "Beginner": 0.26}  # log time multiplier
OPERATORS = [
    # id, skill, experience yrs, belt discipline, idle propensity, aggressiveness,
    # break skipper prob, engine-off-on-break prob, improvement over the 6 months (log mult)
    ("OP1001", "Intermediate", 5, 0.70, 1.2, 1.00, 0.10, 0.60, 0.00),
    ("OP1002", "Expert", 14, 0.95, 0.6, 0.90, 0.05, 0.90, 0.00),
    ("OP1003", "Intermediate", 6, 0.80, 1.0, 1.10, 0.15, 0.70, 0.00),
    ("OP1004", "Beginner", 1, 0.55, 1.5, 1.05, 0.10, 0.40, -0.12),  # improving after training
    ("OP1005", "Expert", 11, 0.90, 0.7, 1.20, 0.30, 0.80, 0.00),   # fast but aggressive, skips breaks
    ("OP1006", "Intermediate", 4, 0.75, 1.1, 0.95, 0.10, 0.70, 0.00),
    ("OP1007", "Beginner", 2, 0.50, 1.6, 1.25, 0.20, 0.30, 0.00),  # risky profile
    ("OP1008", "Intermediate", 7, 0.85, 0.9, 1.00, 0.05, 0.80, 0.00),
    ("OP1009", "Beginner", 1, 0.65, 1.4, 0.95, 0.05, 0.50, -0.10),  # improving after training
    ("OP1010", "Expert", 18, 0.98, 0.5, 0.85, 0.05, 0.95, 0.00),
    ("OP1011", "Intermediate", 3, 0.60, 1.3, 1.10, 0.25, 0.50, 0.00),
    ("OP1012", "Beginner", 1, 0.60, 1.4, 1.00, 0.10, 0.50, 0.00),
]
OPERATORS = [dict(zip(["operator_id", "skill_level", "experience_yrs", "belt_discipline",
                       "idle_propensity", "aggressiveness", "break_skip_prob",
                       "engine_off_on_break_prob", "improvement_log"], o)) for o in OPERATORS]
for o in OPERATORS:  # individual efficiency around the skill mean (hidden trait)
    o["efficiency_log"] = SKILL_EFFECT[o["skill_level"]] + rng.normal(0, 0.04)
OP_BY_ID = {o["operator_id"]: o for o in OPERATORS}
SPARE_OPERATORS = ["OP1009", "OP1010", "OP1011", "OP1012"]

TASK_TYPES = ["Earth Excavation", "Trenching", "Material Loading", "Grading", "Demolition"]
TASK_MIX = {
    "excavator": ([0.30, 0.25, 0.20, 0.10, 0.15], TASK_TYPES),
    "wheel_loader": ([0.70, 0.30], ["Material Loading", "Grading"]),
    "backhoe_loader": ([0.30, 0.40, 0.15, 0.15], ["Earth Excavation", "Trenching", "Material Loading", "Grading"]),
}
TASK_PLAN_MIN = {"Earth Excavation": (40, 120), "Trenching": (30, 90), "Material Loading": (20, 60),
                 "Grading": (25, 70), "Demolition": (45, 150)}
TASK_RATE_M3_MIN = {"Earth Excavation": 1.4, "Trenching": 0.7, "Material Loading": 1.8,
                    "Grading": 1.0, "Demolition": 0.9}
CYCLES_PER_MIN = {"Earth Excavation": 2.6, "Trenching": 2.0, "Material Loading": 2.9,
                  "Grading": 0.45, "Demolition": 1.1}
PERSON_NEAR_P = {"Earth Excavation": 0.15, "Trenching": 0.35, "Material Loading": 0.10,
                 "Grading": 0.25, "Demolition": 0.20}
TRUCK_TASKS = {"Earth Excavation", "Material Loading"}

# ---- weather (north-Indian style season: foggy winter, hot dusty April)
WEATHERS = ["Sunny", "Cloudy", "Rainy", "Windy", "Storm", "Fog", "Extreme Heat", "Dust"]
MONTH_WEATHER_P = {
    11: [.45, .25, .05, .10, .01, .12, .00, .02],
    12: [.30, .25, .08, .07, .01, .29, .00, .00],
    1: [.28, .27, .10, .07, .02, .26, .00, .00],
    2: [.40, .25, .12, .12, .03, .08, .00, .00],
    3: [.50, .18, .08, .12, .04, .01, .03, .04],
    4: [.40, .10, .06, .12, .06, .00, .16, .10],
}
MONTH_TEMP = {11: 26, 12: 21, 1: 19, 2: 23, 3: 30, 4: 36}
WEATHER_TEMP_ADJ = {"Sunny": 1, "Cloudy": -2, "Rainy": -4, "Windy": -1, "Storm": -3, "Fog": -5,
                    "Extreme Heat": 7, "Dust": 2}
SUN = {11: (6.6, 17.5), 12: (7.0, 17.4), 1: (7.2, 17.7), 2: (6.9, 18.2), 3: (6.4, 18.5), 4: (5.9, 18.8), 5: (5.5, 19.1)}
WEATHER_SHIFT = {"Sunny": ["Windy", "Cloudy", "Extreme Heat", "Dust"], "Cloudy": ["Rainy", "Sunny", "Windy"],
                 "Rainy": ["Cloudy", "Storm"], "Windy": ["Dust", "Cloudy", "Storm"], "Storm": ["Rainy", "Cloudy"],
                 "Fog": ["Cloudy", "Sunny"], "Extreme Heat": ["Sunny", "Dust"], "Dust": ["Windy", "Sunny"]}
BAD_WEATHER = {"Rainy", "Storm", "Fog", "Dust"}

# ---- ground-truth effects on task time (log multiplier). The model must rediscover these.
WEATHER_EFFECT = {"Sunny": 0.0, "Cloudy": 0.0, "Rainy": 0.07, "Windy": 0.03, "Storm": 0.30,
                  "Fog": 0.11, "Extreme Heat": 0.07, "Dust": 0.05}
GROUND_EFFECT = {"dry": 0.0, "wet": 0.02, "muddy": 0.09, "loose": 0.03}
LIGHT_EFFECT = {"day": 0.0, "dusk": 0.04, "night": 0.08}
AGE_EFFECT_PER_YR = 0.006

# ---- anomaly episodes injected into shifts (labels for evaluation)
SEGMENT_ANOMALIES = ["excessive_idling", "operator_out_of_seat", "fatigue", "after_hours_use",
                     "unauthorized_operator", "fuel_theft", "unsafe_refuelling"]
MINUTE_ANOMALIES = ["over_rev", "fast_swing", "bucket_raised_travel", "overspeed", "harsh_operation",
                    "slope_exceeded", "overload", "overheating", "low_productivity",
                    "seatbelt_off_while_moving", "sensor_dropout"]
ANOMALY_WEIGHTS = {
    "excessive_idling": 3.0, "operator_out_of_seat": 1.0, "fatigue": 1.0, "after_hours_use": 0.5,
    "unauthorized_operator": 0.4, "fuel_theft": 0.5, "unsafe_refuelling": 0.5, "over_rev": 1.0,
    "fast_swing": 1.5, "bucket_raised_travel": 1.0, "overspeed": 0.8, "harsh_operation": 1.0,
    "slope_exceeded": 0.7, "overload": 1.0, "overheating": 0.6, "low_productivity": 1.2,
    "seatbelt_off_while_moving": 1.2, "sensor_dropout": 0.6,
}
EPISODE_P_PER_SHIFT = 0.40


def pick(options, p=None):
    return options[rng.choice(len(options), p=p)]


# --------------------------------------------------------------------------- environment
def draw_environment(weather: str, month: int, prev_rain: bool) -> dict:
    temp = MONTH_TEMP[month] + WEATHER_TEMP_ADJ[weather] + rng.normal(0, 2)
    if weather == "Extreme Heat":
        temp = max(temp, 40.5 + rng.random() * 3)
    rain = {"Rainy": rng.uniform(2, 20), "Storm": rng.uniform(10, 40)}.get(weather, 0.0)
    wind = {"Windy": rng.uniform(30, 60), "Storm": rng.uniform(40, 70), "Dust": rng.uniform(25, 45)}.get(
        weather, rng.uniform(5, 20))
    vis = {"Fog": rng.uniform(40, 100), "Rainy": rng.uniform(300, 1500), "Dust": rng.uniform(200, 800),
           "Storm": rng.uniform(200, 800)}.get(weather, rng.uniform(5000, 10000))
    dust = {"Dust": rng.uniform(6, 9), "Windy": rng.uniform(3, 5)}.get(weather, rng.uniform(0, 2))
    if weather in ("Rainy", "Storm"):
        ground = "muddy" if rain > 10 else "wet"
    elif prev_rain and rng.random() < 0.6:
        ground = "wet"
    elif weather in ("Dust", "Extreme Heat") and rng.random() < 0.5:
        ground = "loose"
    else:
        ground = "dry"
    return {
        "weather": weather, "ambient_temp_c": round(temp, 1), "rain_mm_h": round(rain, 1),
        "wind_speed_kmh": round(wind, 1), "visibility_m": round(vis), "dust_index": round(dust, 1),
        "ground_condition": ground,
        "lightning_distance_km": round(rng.uniform(2, 15), 1) if weather == "Storm" else np.nan,
    }


def light_at(ts: pd.Timestamp) -> str:
    rise, set_ = SUN[ts.month]
    h = ts.hour + ts.minute / 60
    if rise + 0.5 <= h <= set_ - 0.5:
        return "day"
    if rise - 0.5 <= h < rise + 0.5 or set_ - 0.5 < h <= set_ + 0.5:
        return "dusk"
    return "night"


# --------------------------------------------------------------------------- task time ground truth
def true_task_minutes(task_type, weather, ground, light, temp, machine, op, day_frac, trucks):
    skill = op["skill_level"]
    log_m = op["efficiency_log"] + op["improvement_log"] * day_frac
    log_m += WEATHER_EFFECT[weather] + GROUND_EFFECT[ground] + LIGHT_EFFECT[light]
    log_m += AGE_EFFECT_PER_YR * machine["machine_age_yrs"]
    log_m += 0.01 * max(0.0, temp - 35)
    if task_type == "Demolition" and weather == "Windy":
        log_m += 0.06
    if task_type == "Grading" and weather in ("Rainy", "Storm"):
        log_m += 0.06
    if skill == "Beginner" and task_type in ("Material Loading", "Demolition"):
        log_m += 0.05
    sd = 0.05 + (0.04 if skill == "Beginner" else 0) + (0.03 if weather in BAD_WEATHER else 0)
    return log_m + rng.normal(0, sd)


# --------------------------------------------------------------------------- shift -> segments
def seg(mode, n, **kw):
    d = {"mode": mode, "n": int(max(1, round(n))), "task": None, "label": "normal", "cause": "",
         "engine": None, "seat": None}
    d.update(kw)
    return d


def build_shift(date, machine, op, shift_start, night_shift, spotter, trucks, fuel_start_pct,
                weather_plan, env_by_weather, day_frac, task_counter, episodes):
    """Returns (segments, task_records). Segments are expanded to minutes later."""
    segs, tasks = [], []
    shift_len = rng.uniform(480, 630)
    w1, change_min, w2 = weather_plan

    def weather_at(minute):
        return w1 if minute < change_min else w2

    segs.append(seg("walkaround", rng.uniform(8, 12)))
    if fuel_start_pct < 40:
        unsafe = "unsafe_refuelling" in episodes
        segs.append(seg("refuel", rng.uniform(8, 12), engine="idle" if unsafe else "off",
                        label="unsafe_refuelling" if unsafe else "normal",
                        cause="refuelling with engine running" if unsafe else ""))
    segs.append(seg("warmup", rng.uniform(3, 6)))

    # fatigue episode: skip every break and run long; break skippers normally skip only one short break
    if "fatigue" in episodes:
        skipped = {0, 1, 2}
        shift_len += rng.uniform(60, 120)
    elif rng.random() < op["break_skip_prob"]:
        skipped = {2}  # skipping the afternoon break
    else:
        skipped = set()
    idle_ep_at = rng.integers(1, 4) if "excessive_idling" in episodes else -1
    oos_ep_at = rng.integers(1, 4) if "operator_out_of_seat" in episodes else -1
    breaks_done, elapsed, work_since_break, t_idx = 0, sum(s["n"] for s in segs), 0, 0
    mtype = machine["machine_type"]

    while elapsed < shift_len - 20:
        # breaks: ~2h, lunch ~4.5h, ~6.5h
        due = [120, 270, 400]
        if breaks_done < 3 and elapsed >= due[breaks_done]:
            length = 40 if breaks_done == 1 else 15
            if breaks_done not in skipped:
                off = rng.random() < op["engine_off_on_break_prob"]
                segs.append(seg("break", rng.uniform(length - 5, length + 5), engine="off" if off else "idle"))
                work_since_break = 0
            breaks_done += 1
            elapsed = sum(s["n"] for s in segs)
            continue

        p, types = TASK_MIX[mtype]
        task_type = pick(types, p)
        planned = int(round(rng.uniform(*TASK_PLAN_MIN[task_type]) / 5) * 5)
        start_minute = elapsed + 3
        weather = weather_at(start_minute)
        env = env_by_weather[weather]
        start_ts = shift_start + pd.Timedelta(minutes=start_minute)
        light = light_at(start_ts)
        log_m = true_task_minutes(task_type, weather, env["ground_condition"], light, env["ambient_temp_c"],
                                  machine, op, day_frac, trucks)
        work_minutes = planned * np.exp(log_m)
        # truck delay: fewer trucks -> more / longer waits (learnable via haul_trucks_assigned)
        wait = 0.0
        if task_type in TRUCK_TASKS and rng.random() < min(1.0, 1.2 / trucks):
            wait = min(rng.exponential(planned * 0.25 / trucks), 25)  # longer waits are injected anomalies

        task_counter[0] += 1
        task_id = f"TK{task_counter[0]:06d}"
        segs.append(seg("travel", rng.uniform(2, 6), task=t_idx))
        # work is split into chunks; truck tasks have short truck-swap waits between chunks
        # (swaps are part of normal work time, so they come out of work_minutes)
        pieces = []  # (mode, minutes, cause)
        if task_type in TRUCK_TASKS:
            n_swaps = max(1, int(work_minutes / rng.uniform(8, 14)))
            swaps = rng.uniform(1, 3, n_swaps) * (1.3 if trucks <= 1 else 1.0)
            per_chunk = max(work_minutes - swaps.sum(), work_minutes * 0.5) / (n_swaps + 1)
            for s_len in swaps:
                pieces += [("work", per_chunk, ""), ("wait", s_len, "truck swap")]
            pieces.append(("work", per_chunk, ""))
        else:
            pieces.append(("work", work_minutes, ""))
        mid = int(rng.integers(0, len(pieces))) // 2 * 2  # insert extra idles after a work chunk
        before, after = pieces[:mid + 1], pieces[mid + 1:]
        for mode_, mins, c in before:
            segs.append(seg(mode_, mins, task=t_idx, cause=c))
        injected_idle = 0
        if wait >= 1:
            segs.append(seg("wait", wait, task=t_idx, cause="waiting for haul truck"))
        if t_idx == idle_ep_at:
            cause = pick(["waiting for haul truck", "waiting for site instructions",
                          "engine left running during personal break"], [0.6, 0.25, 0.15])
            injected_idle = rng.uniform(20, 60)
            segs.append(seg("wait", injected_idle, task=t_idx, label="excessive_idling", cause=cause,
                            seat=False if "personal break" in cause else None))
        if t_idx == oos_ep_at:
            segs.append(seg("idle", rng.uniform(4, 15), task=t_idx, seat=False, label="operator_out_of_seat",
                            cause="operator left cab with engine running"))
        for mode_, mins, c in after:
            segs.append(seg(mode_, mins, task=t_idx, cause=c))
        actual = sum(s["n"] for s in segs if s["task"] == t_idx and s["mode"] != "travel")
        tasks.append({
            "task_id": task_id, "t_idx": t_idx, "task_type": task_type, "weather": weather,
            "operator_skill": op["skill_level"], "machine_age_yrs": machine["machine_age_yrs"],
            "estimated_time_min": planned, "actual_time_min": actual,
            "operator_id": op["operator_id"], "machine_id": machine["machine_id"], "machine_model": machine["model"],
            "machine_type": mtype, "date": date.date().isoformat(), "start_ts": start_ts,
            "time_of_day_hour": start_ts.hour, "light": light, "ground_condition": env["ground_condition"],
            "ambient_temp_c": env["ambient_temp_c"], "rain_mm_h": env["rain_mm_h"],
            "wind_speed_kmh": env["wind_speed_kmh"], "visibility_m": env["visibility_m"],
            "haul_trucks_assigned": trucks if task_type in TRUCK_TASKS else 0, "spotter_present": spotter,
            "operator_experience_yrs": op["experience_yrs"], "night_shift": night_shift,
            "target_volume_m3": round(planned * TASK_RATE_M3_MIN[task_type] * machine["size_factor"], 1),
            "idle_min_during_task": sum(x["n"] for x in segs if x["task"] == t_idx and x["mode"] in ("wait", "idle")),
        })
        t_idx += 1
        gap = rng.exponential(6 * op["idle_propensity"])
        if gap >= 1:
            segs.append(seg("idle", gap, cause="between tasks"))
        elapsed = sum(s["n"] for s in segs)
        work_since_break += 1

    segs.append(seg("travel", rng.uniform(2, 5), cause="to parking"))

    if "after_hours_use" in episodes or "unauthorized_operator" in episodes:
        lbl = "unauthorized_operator" if "unauthorized_operator" in episodes else "after_hours_use"
        segs.append(seg("parked", rng.uniform(90, 180)))
        segs.append(seg("warmup", 3, label=lbl, cause="engine started outside shift"))
        segs.append(seg("work", rng.uniform(20, 60), label=lbl, cause="machine used outside scheduled hours"))
        segs.append(seg("idle", rng.uniform(5, 15), label=lbl, cause="machine used outside scheduled hours"))
    if "fuel_theft" in episodes:
        segs.append(seg("parked", rng.uniform(20, 60)))
        segs.append(seg("parked", rng.uniform(10, 30), label="fuel_theft", cause="fuel level dropping with engine off"))
        segs.append(seg("parked", rng.uniform(10, 30)))
    elif rng.random() < 0.15:
        segs.append(seg("parked", rng.uniform(30, 60)))  # normal engine-off data
    return segs, tasks


# --------------------------------------------------------------------------- segments -> minutes
MODE_ENGINE = {"walkaround": "off", "refuel": "off", "warmup": "idle", "travel": "running", "work": "running",
               "wait": "idle", "idle": "idle", "break": "off", "parked": "off"}
MODE_SEAT = {"walkaround": False, "refuel": False, "warmup": True, "travel": True, "work": True, "wait": True,
             "idle": True, "break": False, "parked": False}


def expand_shift(segs, tasks, machine, op, shift_start, shift_end_sched, env_by_weather, weather_plan,
                 fuel_start_pct, spotter, minute_episodes, unauthorized_op):
    n = sum(s["n"] for s in segs)
    mode = np.empty(n, dtype=object)
    engine = np.empty(n, dtype=object)
    seat = np.zeros(n, bool)
    task_idx = np.full(n, -1)
    label = np.full(n, "normal", dtype=object)
    cause = np.full(n, "", dtype=object)
    i = 0
    for s in segs:
        sl = slice(i, i + s["n"])
        mode[sl] = s["mode"]
        engine[sl] = s["engine"] or MODE_ENGINE[s["mode"]]
        seat[sl] = MODE_SEAT[s["mode"]] if s["seat"] is None else s["seat"]
        task_idx[sl] = -1 if s["task"] is None else s["task"]
        label[sl] = s["label"]
        cause[sl] = s["cause"]
        i += s["n"]

    ts = shift_start + pd.to_timedelta(np.arange(n), unit="min")
    w1, change_min, w2 = weather_plan
    weather = np.where(np.arange(n) < change_min, w1, w2)
    env_cols = ["ambient_temp_c", "rain_mm_h", "wind_speed_kmh", "visibility_m", "dust_index",
                "ground_condition", "lightning_distance_km"]
    env = {c: np.array([env_by_weather[w][c] for w in weather]) for c in env_cols}
    light = np.array([light_at(t) for t in ts[::15] for _ in range(15)][:n])

    on = engine != "off"
    running = engine == "running"
    idle = engine == "idle"
    work = mode == "work"
    travel = mode == "travel"
    size, age, aggr = machine["size_factor"], machine["machine_age_yrs"], op["aggressiveness"]
    ttype = np.array([tasks[t]["task_type"] if t >= 0 else "" for t in task_idx], dtype=object)

    rpm = np.where(running, rng.uniform(1500, 1900, n), np.where(idle, rng.uniform(800, 1000, n), 0.0))
    rpm = np.where(travel, rng.uniform(1600, 1800, n), rpm)
    rpm_max = np.where(on, rpm + rng.uniform(40, 250, n) * aggr, 0.0)
    load = np.where(work, rng.uniform(40, 85, n), np.where(travel, rng.uniform(35, 55, n),
                                                            np.where(idle, rng.uniform(5, 15, n), 0.0)))
    load = np.where(ttype == "Demolition", np.minimum(load + 8, 92), load)
    fuel_rate = np.where(running, 6 + 0.16 * load, np.where(idle, rng.uniform(2, 4, n), 0.0))
    fuel_rate = fuel_rate * size * (1 + 0.01 * age) * np.where(running, 0.9 + 0.1 * aggr, 1.0)

    base_cpm = np.array([CYCLES_PER_MIN.get(t, 0.0) for t in ttype])
    speed_factor = np.exp(-op["efficiency_log"]) * (0.85 + 0.15 * min(size, 1.3))
    cycles = np.where(work, rng.poisson(np.maximum(base_cpm * speed_factor, 0.01)), 0).astype(float)

    swing_p95 = np.where(work, rng.normal(26, 4, n) * aggr, 0.0)
    swing_p95 = np.where(ttype == "Grading", swing_p95 * 0.4, swing_p95)
    swing_max = np.where(work, swing_p95 + np.abs(rng.normal(4, 3, n)), 0.0)
    fast_swing = np.where(swing_max > 55, rng.poisson(1, n) + 1, 0)
    speed_max = np.where(travel, rng.uniform(2.5, 5.5, n), np.where(work, rng.uniform(0, 1, n), 0.0))
    harsh = np.where(travel, rng.poisson(0.03 * aggr, n), np.where(work, rng.poisson(0.005 * aggr, n), 0))
    accel_min = np.where(harsh > 0, -rng.uniform(2.6, 3.8, n), np.where(on, -rng.uniform(0.2, 1.8, n), 0.0))
    bucket_raised = np.where(travel & (rng.random(n) < 0.03 * aggr), rng.uniform(3, 12, n), 0.0)
    ground_tilt = {"dry": 0.0, "wet": 1.0, "muddy": 2.0, "loose": 1.5}
    tilt_add = np.array([ground_tilt[g] for g in env["ground_condition"]])
    pitch = np.where(on, np.abs(rng.normal(3.5, 2.0, n)) + tilt_add, 0.0)
    roll = np.where(on, np.abs(rng.normal(1.8, 1.2, n)) + tilt_add * 0.5, 0.0)
    payload = np.where(work & (base_cpm > 0.6), rng.uniform(1200, 1800, n) * size, 0.0)
    overload = np.zeros(n, int)

    # proximity (people only; vehicles are implied by truck_in_bay)
    near_p = np.array([PERSON_NEAR_P.get(t, 0.0) for t in ttype])
    near_p = np.where(work, near_p, np.where(on, 0.05, 0.0))
    has_person = rng.random(n) < near_p
    min_dist = np.where(has_person, rng.uniform(8.5, 20, n), np.nan)  # workers normally keep clear
    close_p = 0.012 * (1.0 if spotter else 2.0) * np.where(np.isin(weather, ["Fog", "Dust", "Storm"]), 1.5, 1.0)
    close_p = close_p * np.where(light == "night", 1.5, 1.0) * (1.3 if op["skill_level"] == "Beginner" else 1.0)
    close = has_person & running & (rng.random(n) < close_p)
    min_dist = np.where(close, rng.uniform(1.5, 5, n), min_dist)
    blind = has_person & (rng.random(n) < np.where(close, 0.55, 0.3))
    truck_in_bay = np.where(work & np.isin(ttype, list(TRUCK_TASKS)), 1.0,
                            np.where(mode == "wait", 0.0, np.nan))
    wait_mask = (mode == "wait") & (cause == "waiting for haul truck")

    # ---- minute-level anomaly episodes
    def window(mask, lo, hi):
        idx = np.flatnonzero(mask)
        if len(idx) < lo:
            return None
        length = int(rng.integers(lo, min(hi, len(idx)) + 1))
        start = int(rng.integers(0, len(idx) - length + 1))
        return idx[start:start + length]

    for ep in minute_episodes:
        base = travel if ep in ("bucket_raised_travel", "overspeed") else (on if ep in ("overheating", "sensor_dropout") else work)
        base = base & (label == "normal")
        w = window(base, 3, 25 if ep not in ("overheating",) else 40)
        if w is None:
            continue
        label[w] = ep
        if ep == "over_rev":
            rpm_max[w] = rng.uniform(2150, 2450, len(w)); rpm[w] = rng.uniform(1900, 2150, len(w))
            fuel_rate[w] *= 1.15; cause[w] = "engine over-revving"
        elif ep == "fast_swing":
            swing_p95[w] = rng.uniform(50, 68, len(w)); swing_max[w] = swing_p95[w] + rng.uniform(3, 10, len(w))
            fast_swing[w] = rng.poisson(4, len(w)) + 2; cause[w] = "swing speed above safe limit"
        elif ep == "bucket_raised_travel":
            bucket_raised[w] = rng.uniform(20, 60, len(w)); cause[w] = "travelling with bucket raised"
        elif ep == "overspeed":
            speed_max[w] = rng.uniform(6.3, 9, len(w)); cause[w] = "travel speed above site limit"
        elif ep == "harsh_operation":
            harsh[w] = rng.poisson(3, len(w)) + 1; accel_min[w] = -rng.uniform(3, 5, len(w))
            cause[w] = "harsh braking / jerky controls"
        elif ep == "slope_exceeded":
            pitch[w] = rng.uniform(15, 22, len(w)); roll[w] = rng.uniform(5, 14, len(w))
            cause[w] = "working beyond slope limit"
        elif ep == "overload":
            payload[w] = rng.uniform(2050, 2600, len(w)) * size; overload[w] = rng.poisson(2, len(w)) + 1
            cause[w] = "bucket overloaded"
        elif ep == "low_productivity":
            cycles[w] = np.floor(cycles[w] * rng.uniform(0.25, 0.5)); cause[w] = "work mode active but few cycles"
        elif ep == "seatbelt_off_while_moving":
            cause[w] = "seatbelt unfastened while operating"
        elif ep == "overheating":
            cause[w] = "coolant / hydraulic oil overheating"
        elif ep == "sensor_dropout":
            cause[w] = "sensor data missing"

    # ---- seatbelt state machine (idle -> unbuckle, the organizers' pattern)
    belt = np.zeros(n, bool)
    b, prev_seat, idle_run = False, False, 0
    disc = op["belt_discipline"]
    unbuckle_rate = 0.12 * (1 - disc) + 0.01
    for k in range(n):
        if not seat[k]:
            b, idle_run, prev_seat = False, 0, False
            continue
        if not prev_seat:
            b = rng.random() < disc
        if mode[k] in ("wait", "idle", "warmup"):
            idle_run += 1
            if b and idle_run >= 3 and rng.random() < unbuckle_rate * (1 + idle_run / 8):
                b = False
        else:
            if idle_run > 0 and not b:
                b = rng.random() < 0.5 + 0.45 * disc
            elif not b and rng.random() < 0.05 + 0.2 * disc:  # in-cab reminder eventually works
                b = True
            idle_run = 0
            if b and rng.random() < 0.0005 * (1 - disc):
                b = False
        belt[k] = b
        prev_seat = True
    belt[label == "seatbelt_off_while_moving"] = False

    # ---- continuous operation (reset by >=10 min out of seat)
    cont = np.zeros(n)
    since_break = np.zeros(n)
    c, out_run = 0, 0
    for k in range(n):
        if seat[k]:
            c += 1
            out_run = 0
        else:
            out_run += 1
            if out_run >= 10:
                c = 0
        cont[k] = c
    since_break[:] = cont
    fatigue_mask = (cont > 240) & running & (label == "normal")
    label[fatigue_mask] = "fatigue"
    cause[fatigue_mask] = "long continuous operation without a break"
    cycles[fatigue_mask] = np.floor(cycles[fatigue_mask] * 0.8)
    harsh[fatigue_mask] += rng.poisson(0.15, fatigue_mask.sum())

    # ---- temperatures (first-order lag towards a target)
    amb = env["ambient_temp_c"].astype(float)
    cool_t = np.where(running, 84 + 0.3 * age + 0.1 * np.maximum(0, amb - 30) + 0.05 * (load - 60),
                      np.where(idle, 80.0, amb))
    hyd_t = np.where(running, 58 + 0.25 * load * 0.3 + 0.2 * np.maximum(0, amb - 30),
                     np.where(idle, 48.0, amb))
    overheat = label == "overheating"
    if overheat.any():
        ramp = np.linspace(0, 1, overheat.sum())
        cool_t[overheat] = 100 + 12 * ramp
        hyd_t[overheat] = 88 + 14 * ramp
    coolant = np.empty(n); hyd = np.empty(n)
    cv, hv = amb[0], amb[0]
    for k in range(n):
        cv += 0.12 * (cool_t[k] - cv); hv += 0.08 * (hyd_t[k] - hv)
        coolant[k], hyd[k] = cv, hv
    coolant += rng.normal(0, 0.5, n); hyd += rng.normal(0, 0.5, n)
    cab = np.where(seat & on, rng.uniform(22, 27, n) + np.maximum(0, amb - 36) * (0.4 + 0.08 * age), amb)
    fault = np.where(overheat & (coolant > 104), "COOLANT_HIGH_TEMP", "")

    # ---- fuel level
    fuel_l = fuel_rate / 60
    tank = machine["fuel_tank_l"]
    level = np.empty(n)
    lv = fuel_start_pct
    theft_mask = label == "fuel_theft"
    theft_per_min = rng.uniform(10, 25) / max(theft_mask.sum(), 1)
    refuel_end = None
    if (mode == "refuel").any():
        refuel_end = np.flatnonzero(mode == "refuel")[-1]
    for k in range(n):
        lv -= fuel_l[k] / tank * 100
        if theft_mask[k]:
            lv -= theft_per_min
        if refuel_end is not None and k == refuel_end:
            lv = rng.uniform(92, 99)
        level[k] = lv
    level += rng.normal(0, 0.15, n)

    # sensor dropout -> NaNs
    drop = label == "sensor_dropout"
    for arr in (rpm, coolant, level, fuel_rate, hyd):
        arr[drop] = np.nan

    # operator shown on the machine
    op_id = np.full(n, op["operator_id"], dtype=object)
    if unauthorized_op:
        op_id[label == "unauthorized_operator"] = unauthorized_op
    scheduled = (ts >= shift_start) & (ts <= shift_end_sched)

    return pd.DataFrame({
        "timestamp": ts, "machine_id": machine["machine_id"], "operator_id": op_id,
        "assigned_operator_id": op["operator_id"],
        "task_id": [tasks[t]["task_id"] if t >= 0 else None for t in task_idx], "task_type": ttype,
        "mode": mode, "engine_state": engine, "seat_occupied": seat, "seatbelt_fastened": belt,
        "rpm_mean": rpm.round(0), "rpm_max": rpm_max.round(0), "load_pct": load.round(1),
        "fuel_rate_lph": fuel_rate.round(2), "fuel_used_l": fuel_l.round(4), "fuel_level_pct": level.round(2),
        "load_cycles": cycles, "swing_rate_p95_dps": swing_p95.round(1), "swing_rate_max_dps": swing_max.round(1),
        "fast_swing_count": fast_swing, "ground_speed_max_kmh": speed_max.round(2), "harsh_brake_count": harsh,
        "longitudinal_accel_min_ms2": accel_min.round(2), "bucket_raised_travel_s": bucket_raised.round(1),
        "pitch_max_deg": pitch.round(1), "roll_max_deg": roll.round(1), "bucket_payload_max_kg": payload.round(0),
        "overload_count": overload, "coolant_temp_c": coolant.round(1), "hydraulic_oil_temp_c": hyd.round(1),
        "cab_temp_c": cab.round(1), "fault_code": fault,
        "min_person_distance_m": min_dist.round(1), "person_in_blind_spot": blind,
        "truck_in_bay": truck_in_bay, "waiting_for_truck": wait_mask,
        "continuous_operation_min": cont, "weather": weather, "light": light,
        **{c: env[c] for c in env_cols},
        "within_scheduled_hours": scheduled, "anomaly_label": label, "anomaly_cause": cause,
    })


# --------------------------------------------------------------------------- safety alerts + incidents
def add_safety_and_incidents(df: pd.DataFrame) -> pd.DataFrame:
    on = df.engine_state != "off"
    moving = df["mode"].isin(["work", "travel"])
    bad_vis = df.weather.isin(["Rainy", "Fog", "Storm", "Dust"]) | (df.light == "night")
    danger_r = np.where(bad_vis, 8.0, 5.0)
    df["danger_radius_m"] = danger_r
    df["alert_red_zone"] = (df.min_person_distance_m < danger_r) & on
    df["alert_seatbelt"] = df.seat_occupied & ~df.seatbelt_fastened & moving
    df["alert_out_of_seat"] = on & ~df.seat_occupied & (df["mode"] != "break")  # idling on break = fuel waste, not an alert
    df["alert_tilt"] = (df.pitch_max_deg > 15) | (df.roll_max_deg > 15)
    df["alert_overspeed"] = df.ground_speed_max_kmh > 6
    df["alert_lightning"] = (df.lightning_distance_km < 10) & moving
    df["alert_refuel_engine_on"] = (df["mode"] == "refuel") & on
    df["alert_heat_stress"] = df.cab_temp_c > 35
    df["alert_fast_swing"] = df.fast_swing_count > 0
    alert_cols = [c for c in df.columns if c.startswith("alert_")]
    df["safety_alert"] = df[alert_cols].any(axis=1)

    # incident hazard model (ground truth for the Readiness Score)
    skill = df.operator_id.map(lambda o: OP_BY_ID.get(o, {"skill_level": "Intermediate"})["skill_level"])
    z = (2.6 * df.alert_red_zone + 1.2 * df.alert_seatbelt + 0.9 * (df.continuous_operation_min > 240)
         + 0.6 * df.weather.isin(list(BAD_WEATHER)) + 0.5 * (df.light == "night") + 0.8 * (df.harsh_brake_count > 0)
         + 0.7 * df.alert_fast_swing + 0.5 * (skill == "Beginner") + 1.2 * df.alert_tilt + 0.8 * df.alert_overspeed
         + 0.6 * df.alert_heat_stress + 0.6 * df.person_in_blind_spot)
    p = np.where(on & (df["mode"] != "refuel"), 0.00025 * np.exp(z), 0.0)
    df["incident"] = rng.random(len(df)) < p
    return df


def incident_type(row) -> tuple[str, str]:
    if row.alert_red_zone:
        return (pick(["near_miss_person", "person_struck"], [0.95, 0.05]),
                "worker inside danger zone" + (" (blind spot)" if row.person_in_blind_spot else ""))
    if row.alert_tilt:
        return "tip_over_risk", "machine exceeded slope limit"
    if row.alert_overspeed or row.harsh_brake_count > 0:
        return pick(["near_miss_vehicle", "minor_collision"], [0.7, 0.3]), "harsh or fast travel"
    if row.alert_heat_stress:
        return "heat_stress_symptoms", "high cab temperature"
    if row.alert_fast_swing:
        return pick(["near_miss_person", "property_damage"], [0.5, 0.5]), "fast swing"
    return (pick(["property_damage", "near_miss_vehicle", "utility_strike_risk", "near_miss_person"],
                 [0.35, 0.3, 0.1, 0.25]), "general operating risk")


SEVERITY = {"near_miss_person": "high", "person_struck": "critical", "tip_over_risk": "high",
            "near_miss_vehicle": "warning", "minor_collision": "high", "heat_stress_symptoms": "warning",
            "property_damage": "warning", "utility_strike_risk": "high"}


# --------------------------------------------------------------------------- aggregations
def hourly_summaries(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    m["hour"] = m.timestamp.dt.floor("h")
    on = m.engine_state != "off"
    m["on_min"] = on.astype(int)
    m["idle_min"] = (m.engine_state == "idle").astype(int)
    m["belt_off_on_min"] = (on & m.seat_occupied & ~m.seatbelt_fastened).astype(int)
    m["seated_on_min"] = (on & m.seat_occupied).astype(int)
    g = m.groupby(["machine_id", "hour"])
    h = g.agg(operator_id=("operator_id", lambda s: s.mode().iat[0]), engine_hours=("cum_engine_hours", "max"),
              fuel_used_l=("fuel_used_l", "sum"), load_cycles=("load_cycles", "sum"), idle_min=("idle_min", "sum"),
              on_min=("on_min", "sum"), belt_off_min=("belt_off_on_min", "sum"),
              seated_on_min=("seated_on_min", "sum"), safety_alerts=("safety_alert", "sum"),
              red_zone_min=("alert_red_zone", "sum"), incidents=("incident", "sum"),
              weather=("weather", lambda s: s.mode().iat[0]),
              ambient_temp_c=("ambient_temp_c", "mean"), truck_wait_min=("waiting_for_truck", "sum"),
              anomaly_minutes=("anomaly_label", lambda s: (s != "normal").sum()),
              anomaly_labels=("anomaly_label", lambda s: "|".join(sorted(set(s) - {"normal"})))).reset_index()
    h = h[h.on_min > 0].copy()
    h["truck_loads"] = np.floor(h.load_cycles / 6).astype(int)
    h["idle_ratio"] = (h.idle_min / h.on_min).round(3)
    h["fuel_per_cycle_l"] = (h.fuel_used_l / h.load_cycles.clip(lower=1)).round(3)
    h["cycles_per_engine_hour"] = (h.load_cycles / (h.on_min / 60)).round(1)
    h["seatbelt_compliance_pct"] = (100 * (1 - h.belt_off_min / h.seated_on_min.clip(lower=1))).round(1)
    h["seatbelt_status"] = np.where(h.belt_off_min >= 5, "Unfastened", "Fastened")
    h["safety_alert_triggered"] = np.where(h.safety_alerts > 0, "Yes", "No")
    h["fuel_used_l"] = h.fuel_used_l.round(2)
    h["engine_hours"] = h.engine_hours.round(1)
    h["load_cycles"] = h.load_cycles.astype(int)
    h = h.rename(columns={"hour": "timestamp", "idle_min": "idling_time_min", "on_min": "engine_on_min"})
    return h


def window_features(m: pd.DataFrame, minutes: int = 5) -> pd.DataFrame:
    m = m.copy()
    m["window_start"] = m.timestamp.dt.floor(f"{minutes}min")
    on = m.engine_state != "off"
    m["on"] = on.astype(int)
    m["idle"] = (m.engine_state == "idle").astype(int)
    m["work"] = (m["mode"] == "work").astype(int)
    m["travel"] = (m["mode"] == "travel").astype(int)
    m["over_rev"] = (m.rpm_max > 2100).astype(int)
    m["belt_off_on"] = (on & m.seat_occupied & ~m.seatbelt_fastened).astype(int)
    m["seat_empty_on"] = (on & ~m.seat_occupied).astype(int)
    m["dropout"] = m.rpm_mean.isna().astype(int) * on
    m["off_fuel_drop"] = np.where(~on, -m.groupby("machine_id").fuel_level_pct.diff().fillna(0), 0.0)
    m["is_anom"] = (m.anomaly_label != "normal").astype(int)
    rank = m.anomaly_label.where(m.anomaly_label != "normal")

    g = m.groupby(["machine_id", "window_start"])
    w = g.agg(
        operator_id=("operator_id", "first"), assigned_operator_id=("assigned_operator_id", "first"),
        task_id=("task_id", "first"), task_type=("task_type", lambda s: s.mode().iat[0] if len(s) else ""),
        minutes=("on", "size"), engine_on_min=("on", "sum"), idle_min=("idle", "sum"), work_min=("work", "sum"),
        travel_min=("travel", "sum"), fuel_used_l=("fuel_used_l", "sum"), load_cycles=("load_cycles", "sum"),
        rpm_mean=("rpm_mean", "mean"), rpm_max=("rpm_max", "max"), over_rev_min=("over_rev", "sum"),
        load_pct_mean=("load_pct", "mean"), swing_rate_p95_dps=("swing_rate_p95_dps", "max"),
        fast_swing_count=("fast_swing_count", "sum"), harsh_brake_count=("harsh_brake_count", "sum"),
        ground_speed_max_kmh=("ground_speed_max_kmh", "max"), bucket_raised_travel_s=("bucket_raised_travel_s", "sum"),
        pitch_max_deg=("pitch_max_deg", "max"), roll_max_deg=("roll_max_deg", "max"),
        bucket_payload_max_kg=("bucket_payload_max_kg", "max"), overload_count=("overload_count", "sum"),
        coolant_max_c=("coolant_temp_c", "max"), hydraulic_oil_max_c=("hydraulic_oil_temp_c", "max"),
        cab_temp_max_c=("cab_temp_c", "max"), fuel_level_pct=("fuel_level_pct", "last"),
        fuel_drop_engine_off_pct=("off_fuel_drop", "sum"), seat_empty_engine_on_min=("seat_empty_on", "sum"),
        seatbelt_off_engine_on_min=("belt_off_on", "sum"), seatbelt_off_moving_min=("alert_seatbelt", "sum"),
        min_person_distance_m=("min_person_distance_m", "min"), red_zone_min=("alert_red_zone", "sum"),
        blind_spot_min=("person_in_blind_spot", "sum"), truck_wait_min=("waiting_for_truck", "sum"),
        continuous_operation_min=("continuous_operation_min", "max"),
        within_scheduled_hours=("within_scheduled_hours", "min"), sensor_dropout_min=("dropout", "sum"),
        weather=("weather", "first"), light=("light", "first"), ground_condition=("ground_condition", "first"),
        ambient_temp_c=("ambient_temp_c", "first"), visibility_m=("visibility_m", "first"),
        lightning_distance_km=("lightning_distance_km", "min"), safety_alert_min=("safety_alert", "sum"),
        incident=("incident", "max"), anomaly_min=("is_anom", "sum"),
    ).reset_index()
    lbl = rank.groupby([m.machine_id, m.window_start]).agg(lambda s: s.mode().iat[0] if s.notna().any() else "normal")
    cause = m.anomaly_cause.where(m.anomaly_cause != "").groupby([m.machine_id, m.window_start]).agg(
        lambda s: s.dropna().mode().iat[0] if s.notna().any() else "")
    w["anomaly_label"] = lbl.values
    w["anomaly_cause"] = cause.values
    # a window counts as anomalous only if >= 2 of its minutes are (avoids edge slivers)
    w.loc[w.anomaly_min < 2, ["anomaly_label"]] = "normal"
    w.loc[w.anomaly_label == "normal", "anomaly_cause"] = ""
    w["is_anomaly"] = (w.anomaly_label != "normal").astype(int)

    w["idle_ratio"] = (w.idle_min / w.engine_on_min.clip(lower=1)).round(3)
    w["fuel_rate_lph"] = (w.fuel_used_l / (w.engine_on_min.clip(lower=1) / 60)).round(2)
    w["fuel_per_cycle_l"] = (w.fuel_used_l / w.load_cycles.clip(lower=1)).round(3)
    w["cycles_per_hour"] = (w.load_cycles / (w.engine_on_min.clip(lower=1) / 60)).round(1)
    w["operator_matches_assigned"] = (w.operator_id == w.assigned_operator_id).astype(int)
    w["hour_of_day"] = w.window_start.dt.hour
    w["min_person_distance_m"] = w.min_person_distance_m.fillna(99.0)
    w["within_scheduled_hours"] = w.within_scheduled_hours.astype(int)
    for c in ("rpm_mean", "load_pct_mean", "fuel_used_l"):
        w[c] = w[c].round(2)
    return w.rename(columns={"window_start": "timestamp"})


# --------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(START_DATE, END_DATE, freq="D")
    n_days = len(days)
    minute_frames, task_rows, shifts = [], [], []
    task_counter = [0]
    fuel_pct = {m["machine_id"]: rng.uniform(50, 95) for m in MACHINES}

    prev_rain = False
    for d_i, day in enumerate(days):
        if day.dayofweek == 6:  # Sunday off
            continue
        wp = MONTH_WEATHER_P[day.month]
        w1 = pick(WEATHERS, np.array(wp) / sum(wp))
        change_min, w2 = 10_000, w1
        if rng.random() < 0.2:
            change_min, w2 = int(rng.uniform(120, 480)), pick(WEATHER_SHIFT[w1])
        env_by_weather = {w1: draw_environment(w1, day.month, prev_rain)}
        env_by_weather[w2] = env_by_weather.get(w2) or draw_environment(w2, day.month, prev_rain or w1 in ("Rainy", "Storm"))
        prev_rain = w2 in ("Rainy", "Storm") or w1 in ("Rainy", "Storm")
        spotter = rng.random() < 0.7
        trucks = int(rng.choice([1, 2, 3, 4], p=[0.15, 0.4, 0.3, 0.15]))
        used_spares = set()

        for machine in MACHINES:
            if rng.random() < 0.04:  # maintenance day
                continue
            op_id = machine["primary_operator_id"]
            if rng.random() < 0.12:
                free = [s for s in SPARE_OPERATORS if s not in used_spares and s != op_id]
                if free:
                    op_id = pick(free)
                    used_spares.add(op_id)
            op = OP_BY_ID[op_id]
            night = rng.random() < 0.06
            shift_start = day + pd.Timedelta(hours=19 if night else 7, minutes=int(rng.uniform(0, 45)))
            shift_end_sched = shift_start + pd.Timedelta(hours=10, minutes=30)

            episodes = []
            if rng.random() < EPISODE_P_PER_SHIFT * (1.4 if op["skill_level"] == "Beginner" else 1.0):
                k = 1 + int(rng.random() < 0.3)
                names = list(ANOMALY_WEIGHTS)
                p = np.array([ANOMALY_WEIGHTS[a] for a in names]); p = p / p.sum()
                episodes = list(rng.choice(names, size=k, replace=False, p=p))
            if "after_hours_use" in episodes and "unauthorized_operator" in episodes:
                episodes.remove("after_hours_use")
            if night:  # keep night shifts simple: no after-hours episodes
                episodes = [e for e in episodes if e not in ("after_hours_use", "unauthorized_operator")]
            unauthorized = "OP9" + str(rng.integers(100, 999)) if "unauthorized_operator" in episodes else None

            segs, tasks = build_shift(day, machine, op, shift_start, night, spotter, trucks, fuel_pct[machine["machine_id"]],
                                      (w1, change_min, w2), env_by_weather, d_i / n_days, task_counter,
                                      [e for e in episodes if e in SEGMENT_ANOMALIES])
            mdf = expand_shift(segs, tasks, machine, op, shift_start, shift_end_sched, env_by_weather,
                               (w1, change_min, w2), fuel_pct[machine["machine_id"]], spotter,
                               [e for e in episodes if e in MINUTE_ANOMALIES], unauthorized)
            fuel_pct[machine["machine_id"]] = float(np.nanmax([mdf.fuel_level_pct.iloc[-1], 5.0]))
            mdf["shift_id"] = f"{machine['machine_id']}-{day.date().isoformat()}"
            minute_frames.append(mdf)
            task_rows.extend(tasks)
            shifts.append({"shift_id": mdf.shift_id.iat[0], "date": day.date().isoformat(),
                           "machine_id": machine["machine_id"], "operator_id": op_id,
                           "shift_start": shift_start, "scheduled_end": shift_end_sched,
                           "actual_end": mdf.timestamp.iat[-1], "night_shift": night,
                           "spotter_present": spotter, "haul_trucks_assigned": trucks,
                           "weather_start": w1, "weather_end": w2,
                           "injected_anomalies": "|".join(episodes)})
        if d_i % 30 == 0:
            print(f"  simulated up to {day.date()}  ({len(minute_frames)} shifts)")

    m = pd.concat(minute_frames, ignore_index=True).sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    m = m.drop_duplicates(["machine_id", "timestamp"], keep="first")  # night shifts can overlap the next morning

    # cumulative ECU counters (EXC001 lands on ~1523.5 h on demo day, like the organizers' sample)
    on_h = (m.engine_state != "off") / 60.0
    m["cum_engine_hours"] = on_h.groupby(m.machine_id).cumsum()
    m["cum_idle_hours"] = ((m.engine_state == "idle") / 60.0).groupby(m.machine_id).cumsum()
    m["cum_fuel_used_l"] = m.fuel_used_l.fillna(0).groupby(m.machine_id).cumsum()
    m["cum_load_cycles"] = m.load_cycles.groupby(m.machine_id).cumsum()
    for mach in MACHINES:
        sel = m.machine_id == mach["machine_id"]
        end_h = m.loc[sel, "cum_engine_hours"].iat[-1]
        target = 1523.5 if mach["machine_id"] == "EXC001" else mach["machine_age_yrs"] * 1150 + rng.uniform(0, 300)
        off_h = max(target - end_h, 0)
        m.loc[sel, "cum_engine_hours"] += off_h
        m.loc[sel, "cum_idle_hours"] += off_h * 0.28
        m.loc[sel, "cum_fuel_used_l"] += off_h * 11 * mach["size_factor"]
        m.loc[sel, "cum_load_cycles"] += off_h * 95
    for c in ("cum_engine_hours", "cum_idle_hours", "cum_fuel_used_l"):
        m[c] = m[c].round(3)

    print("adding safety alerts and incidents ...")
    m = add_safety_and_incidents(m)

    # ---- incidents table
    inc = m[m.incident].copy()
    types = inc.apply(incident_type, axis=1, result_type="expand")
    inc["incident_type"], inc["cause"] = types[0], types[1]
    inc["severity"] = inc.incident_type.map(SEVERITY)
    inc["source"] = np.where(inc.alert_red_zone | inc.alert_tilt | (inc.harsh_brake_count > 0), "auto",
                             np.where(rng.random(len(inc)) < 0.6, "operator_report", "supervisor_report"))
    incidents = inc[["timestamp", "machine_id", "operator_id", "task_id", "incident_type", "severity", "cause",
                     "source", "weather", "light", "min_person_distance_m", "person_in_blind_spot",
                     "seatbelt_fastened", "continuous_operation_min", "ground_speed_max_kmh",
                     "pitch_max_deg", "cab_temp_c"]].reset_index(drop=True)
    incidents.insert(0, "incident_id", [f"INC{i:05d}" for i in range(1, len(incidents) + 1)])

    # ---- task records (+ the organizers' 5 rows verbatim)
    tasks = pd.DataFrame(task_rows)
    inc_by_task = m[m.incident & m.task_id.notna()].groupby("task_id").size()
    tasks["incidents_during_task"] = tasks.task_id.map(inc_by_task).fillna(0).astype(int)
    tasks["volume_moved_m3"] = tasks.target_volume_m3  # tasks run to completion
    tasks["overrun_pct"] = (100 * (tasks.actual_time_min / tasks.estimated_time_min - 1)).round(1)
    tasks["source"] = "synthetic"
    tasks["start_ts"] = tasks.start_ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    tasks = tasks.drop(columns=["t_idx"])
    organizer = pd.DataFrame({
        "task_id": ["T001", "T002", "T003", "T004", "T005"],
        "task_type": ["Earth Excavation", "Trenching", "Material Loading", "Grading", "Demolition"],
        "weather": ["Sunny", "Rainy", "Cloudy", "Sunny", "Windy"],
        "operator_skill": ["Expert", "Intermediate", "Beginner", "Expert", "Intermediate"],
        "machine_age_yrs": [2, 4, 3, 5, 6], "estimated_time_min": [60, 45, 30, 35, 90],
        "actual_time_min": [58, 52, 42, 33, 105], "source": "organizer_sample"})
    organizer["overrun_pct"] = (100 * (organizer.actual_time_min / organizer.estimated_time_min - 1)).round(1)
    tasks = pd.concat([organizer, tasks], ignore_index=True)

    print("aggregating hourly summaries and 5-min windows ...")
    hourly = hourly_summaries(m)
    windows = window_features(m)

    # readiness label: incident in the next 30 minutes on the same machine
    inc_times = m.loc[m.incident, ["machine_id", "timestamp"]]
    windows["incident_next_30min"] = 0
    for mid, grp in inc_times.groupby("machine_id"):
        sel = windows.machine_id == mid
        ws = windows.loc[sel, "timestamp"].values
        t = np.sort(grp.timestamp.values)
        nxt = np.searchsorted(t, ws + np.timedelta64(5, "m"))
        has = nxt < len(t)
        delta = np.full(len(ws), np.timedelta64(10**9, "m"))
        delta[has] = t[nxt[has]] - ws[has]
        windows.loc[sel, "incident_next_30min"] = ((delta <= np.timedelta64(35, "m"))).astype(int)

    # ---- write
    print("writing files ...")
    pd.DataFrame(MACHINES).to_csv(OUT / "machines.csv", index=False)
    pd.DataFrame([{k: v for k, v in o.items()} | {"efficiency_log": round(o["efficiency_log"], 4)}
                  for o in OPERATORS]).to_csv(OUT / "operators_ground_truth.csv", index=False)
    pd.DataFrame([{k: o[k] for k in ("operator_id", "skill_level", "experience_yrs")}
                  for o in OPERATORS]).to_csv(OUT / "operators.csv", index=False)
    pd.DataFrame(shifts).to_csv(OUT / "shifts.csv", index=False)
    tasks.to_csv(OUT / "task_records.csv", index=False)
    hourly.to_csv(OUT / "hourly_summaries.csv", index=False)
    org_cols = hourly.assign(**{"Timestamp": hourly.timestamp.dt.strftime("%Y-%m-%d %H:%M:%S")})
    org_cols = org_cols.rename(columns={"machine_id": "Machine ID", "operator_id": "Operator ID",
                                        "engine_hours": "Engine Hours", "fuel_used_l": "Fuel Used (L)",
                                        "load_cycles": "Load Cycles", "idling_time_min": "Idling Time (min)",
                                        "seatbelt_status": "Seatbelt Status",
                                        "safety_alert_triggered": "Safety Alert Triggered"})
    org_cols[["Timestamp", "Machine ID", "Operator ID", "Engine Hours", "Fuel Used (L)", "Load Cycles",
              "Idling Time (min)", "Seatbelt Status", "Safety Alert Triggered"]].to_csv(
        OUT / "telemetry_organizer_format.csv", index=False)
    windows.to_csv(OUT / "anomaly_windows_5min.csv", index=False)
    readiness_cols = ["timestamp", "machine_id", "operator_id", "seatbelt_off_moving_min", "seatbelt_off_engine_on_min",
                      "seat_empty_engine_on_min", "min_person_distance_m", "red_zone_min", "blind_spot_min",
                      "harsh_brake_count", "fast_swing_count", "ground_speed_max_kmh", "pitch_max_deg", "roll_max_deg",
                      "continuous_operation_min", "cab_temp_max_c", "weather", "light", "ground_condition",
                      "visibility_m", "lightning_distance_km", "is_anomaly", "anomaly_label", "incident_next_30min"]
    windows[windows.engine_on_min > 0][readiness_cols].to_csv(OUT / "readiness_training.csv", index=False)
    incidents.to_csv(OUT / "incidents.csv", index=False)
    m.drop(columns=["incident"]).to_parquet(OUT / "minute_telemetry.parquet", index=False)

    write_training_catalog(windows, incidents)
    print_summary(tasks, hourly, windows, incidents, m)


TRAINING_MODULES = [
    ("TRN-IDLE-01", "Cutting idle time", "micro_lesson", 2, ["excessive_idling"]),
    ("TRN-BELT-01", "Why the belt stays on", "micro_lesson", 2, ["seatbelt_off_while_moving"]),
    ("TRN-CAB-01", "Leaving the cab safely", "micro_lesson", 3, ["operator_out_of_seat", "unsafe_refuelling"]),
    ("TRN-SWING-01", "Smooth swing control", "simulation", 5, ["fast_swing", "harsh_operation"]),
    ("TRN-TRAVEL-01", "Safe travel on site", "micro_lesson", 3, ["bucket_raised_travel", "overspeed"]),
    ("TRN-SLOPE-01", "Working on slopes", "simulation", 6, ["slope_exceeded"]),
    ("TRN-LOAD-01", "Bucket fill and load limits", "video", 4, ["overload", "low_productivity"]),
    ("TRN-ENGINE-01", "Engine care: RPM and temperatures", "video", 4, ["over_rev", "overheating"]),
    ("TRN-FATIGUE-01", "Managing fatigue and breaks", "micro_lesson", 3, ["fatigue"]),
    ("TRN-PROX-01", "Blind spots and the danger zone", "simulation", 6, ["near_miss_person", "person_struck"]),
    ("TRN-SEC-01", "Machine security and access", "micro_lesson", 2,
     ["fuel_theft", "after_hours_use", "unauthorized_operator"]),
    ("TRN-WX-01", "Working in rain, fog and dust", "video", 5, ["bad_weather"]),
    ("TRN-INSTR-01", "Instructor session: operating review", "instructor_booking", 60, ["repeated_unsafe"]),
]


def write_training_catalog(windows, incidents):
    catalog = [{"module_id": m, "title": t, "format": f, "duration_min": d, "triggers": trg,
                "quiz_questions": 3 if f != "instructor_booking" else 0} for m, t, f, d, trg in TRAINING_MODULES]
    (OUT / "training_modules.json").write_text(json.dumps(catalog, indent=2))
    trigger_map = {trg: m for m, _, _, _, trgs in TRAINING_MODULES for trg in trgs}
    rows = []
    ev = windows[windows.is_anomaly == 1][["timestamp", "operator_id", "anomaly_label"]].rename(
        columns={"anomaly_label": "reason"})
    ev = pd.concat([ev, incidents[["timestamp", "operator_id", "incident_type"]].rename(
        columns={"incident_type": "reason"})])
    ev["day"] = ev.timestamp.dt.date
    ev = ev.drop_duplicates(["operator_id", "reason", "day"])
    for _, r in ev.sort_values("timestamp").iterrows():
        mod = trigger_map.get(r.reason)
        if not mod or rng.random() > 0.35:
            continue
        op = OP_BY_ID.get(r.operator_id)
        if not op:
            continue
        done = r.timestamp + pd.Timedelta(days=float(rng.uniform(0.5, 5)))
        base = {"Beginner": 68, "Intermediate": 78, "Expert": 88}[op["skill_level"]]
        rows.append({"operator_id": r.operator_id, "module_id": mod, "recommended_at": r.timestamp,
                     "reason": r.reason, "completed_at": done if rng.random() < 0.75 else pd.NaT,
                     "quiz_score_pct": int(np.clip(rng.normal(base, 10), 30, 100))})
    th = pd.DataFrame(rows)
    th.loc[th.completed_at.isna(), "quiz_score_pct"] = np.nan
    th.to_csv(OUT / "training_history.csv", index=False)


def print_summary(tasks, hourly, windows, incidents, m):
    print("\n=== summary ===")
    print(f"minute rows: {len(m):,}   shifts: {m.shift_id.nunique():,}")
    print(f"task records: {len(tasks):,}   hourly: {len(hourly):,}   windows: {len(windows):,}   incidents: {len(incidents):,}")
    syn = tasks[tasks.source == "synthetic"]
    print("\nmedian overrun % by skill:\n", syn.groupby("operator_skill").overrun_pct.median().round(1).to_string())
    print("\nmedian overrun % by weather:\n", syn.groupby("weather").overrun_pct.median().round(1).to_string())
    h = hourly
    print("\nhourly by seatbelt status (organizer pattern check):")
    print(h.groupby("seatbelt_status")[["idling_time_min", "load_cycles", "fuel_per_cycle_l"]].median().to_string())
    print(f"\nwindows anomalous: {windows.is_anomaly.mean():.2%}")
    print(windows[windows.is_anomaly == 1].anomaly_label.value_counts().to_string())
    print(f"\nreadiness positives (incident within 30 min): {windows.incident_next_30min.mean():.2%}")


if __name__ == "__main__":
    main()
