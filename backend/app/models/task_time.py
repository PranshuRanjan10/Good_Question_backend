"""
IronSense - task time estimator (inference).

Public API (called by the live backend):

    from app.models.task_time import predict_task
    predict_task(task, env, operator, machine, elapsed_min=21, progress_pct=38.5)

Returns the `task_prediction` block of the `assessment` message:

    {"task_id", "planned_min", "p10_min", "p50_min", "p90_min", "remaining_min",
     "factors": [{"factor": "weather_rainy", "effect_pct": 10}, ...]}

Call it at task start, on every weather change, and once a minute while a task runs.
Nothing here touches FastAPI, the DB or the network: it is a pure function over dicts.
"""
from __future__ import annotations

import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "task_time_model.joblib"

_BUNDLE = None
_LOCK = threading.Lock()

# How much to trust observed pace over the model, by fraction of the task completed.
# Early progress is noisy, so the model dominates until the task is properly under way.
_PACE_TRUST = [(0.10, 0.0), (0.25, 0.35), (0.50, 0.65), (0.75, 0.85), (1.01, 0.95)]

_FACTOR_LABELS = {
    "task_type": "task", "weather": "weather", "operator_skill": "skill",
    "light": "light", "ground_condition": "ground",
}


def load(path: Path | str = ARTIFACT) -> dict:
    """Load the trained bundle once per process. Call at startup to fail fast."""
    global _BUNDLE
    with _LOCK:
        if _BUNDLE is None:
            _BUNDLE = joblib.load(path)
    return _BUNDLE


def _row(task: dict, env: dict, operator: dict, machine: dict, b: dict) -> pd.DataFrame:
    """Build one feature row, filling anything the caller didn't send."""
    d = b["defaults"]
    weather = env.get("weather") or "Sunny"
    skill = operator.get("skill_level") or "Intermediate"
    planned = float(task.get("planned_estimate_min") or task.get("estimated_time_min") or 0.0)
    ground = env.get("ground_condition") or b["weather_ground"].get(weather, d["ground_condition"])
    exp_yrs = operator.get("experience_yrs")
    if exp_yrs is None:
        exp_yrs = d["operator_experience_yrs"].get(skill, 5.0)
    hour = task.get("time_of_day_hour")
    if hour is None and task.get("scheduled_start"):
        hour = pd.Timestamp(task["scheduled_start"]).hour
    vol = task.get("target_volume_m3")

    vals = {
        "task_type": task.get("task_type") or "Earth Excavation",
        "weather": weather,
        "operator_skill": skill,
        "light": env.get("light") or d["light"],
        "ground_condition": ground,
        "estimated_time_min": planned,
        "machine_age_yrs": float(machine.get("machine_age_yrs") or 4),
        "ambient_temp_c": _num(env.get("ambient_temp_c"), d["ambient_temp_c"]),
        "rain_mm_h": _num(env.get("rain_mm_h"), d["rain_mm_h"]),
        "wind_speed_kmh": _num(env.get("wind_speed_kmh"), d["wind_speed_kmh"]),
        "visibility_m": _num(env.get("visibility_m"), d["visibility_m"]),
        "haul_trucks_assigned": _num(task.get("haul_trucks_assigned"), d["haul_trucks_assigned"]),
        "operator_experience_yrs": float(exp_yrs),
        "time_of_day_hour": _num(hour, d["time_of_day_hour"]),
        "target_volume_m3": _num(vol, planned * 1.2),
        "spotter_present": 1.0 if env.get("spotter_present", True) else 0.0,
        "night_shift": 1.0 if env.get("night_shift") else 0.0,
    }
    X = pd.DataFrame([vals])[b["features"]]
    for c in b["cat_features"]:
        X[c] = X[c].astype("category")
    return X


def _num(v, fallback):
    return float(fallback) if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def _factors(X: pd.DataFrame, b: dict, top: int = 4) -> list[dict]:
    """The biggest named effects for this task, from the linear model's coefficients."""
    out = []
    for key, levels in b["factors"].items():
        if key not in X:
            continue
        level = str(X[key].iloc[0])
        pct = levels.get(level)
        if pct is None or abs(pct) < 1.0:
            continue
        out.append({"factor": f"{_FACTOR_LABELS.get(key, key)}_{level}".lower().replace(" ", "_"),
                    "effect_pct": round(pct)})
    out.sort(key=lambda f: -abs(f["effect_pct"]))
    return out[:top]


def _pace_weight(progress_pct: float) -> float:
    frac = max(0.0, min(progress_pct, 100.0)) / 100.0
    for edge, wgt in _PACE_TRUST:
        if frac < edge:
            return wgt
    return _PACE_TRUST[-1][1]


def predict_task(task: dict, env: dict, operator: dict, machine: dict,
                 elapsed_min: float | None = None, progress_pct: float | None = None,
                 bundle: dict | None = None) -> dict:
    """
    task     : {task_id, task_type, planned_estimate_min, target_volume_m3?, haul_trucks_assigned?,
                scheduled_start?}
    env      : {weather, ambient_temp_c, rain_mm_h, wind_speed_kmh, visibility_m, light,
                ground_condition, spotter_present?, night_shift?}
    operator : {operator_id?, skill_level, experience_yrs?}
    machine  : {machine_id?, machine_age_yrs}

    Pass elapsed_min and progress_pct while the task is running to get a live
    remaining_min that blends the model with the pace actually observed so far.
    """
    b = bundle or load()
    X = _row(task, env, operator, machine, b)
    planned = float(X["estimated_time_min"].iloc[0])
    off = b.get("offsets", {})

    p = {}
    for tag in ("p10", "p50", "p90"):
        p[tag] = float(planned * np.exp(b["lgbm"][tag].predict(X)[0] + off.get(tag, 0.0)))
    lo, mid, hi = sorted((p["p10"], p["p50"], p["p90"]))  # quantile models can cross

    remaining = None
    if elapsed_min is not None:
        if progress_pct and progress_pct >= 1:
            observed_total = elapsed_min / (min(progress_pct, 99.0) / 100.0)
            w = _pace_weight(progress_pct)
            # Blend all three with the observed pace. p50 must move too, or a fast/slow pace
            # drags p10/p90 across it (seen live: p10 37, p50 49, p90 42).
            mid = (1 - w) * mid + w * observed_total
            lo = (1 - w) * lo + w * observed_total * 0.9
            hi = (1 - w) * hi + w * observed_total * 1.1
        # The task can't finish before the time already spent.
        lo, mid, hi = (max(v, elapsed_min) for v in (lo, mid, hi))
        lo, mid, hi = sorted((lo, mid, hi))
        remaining = max(0.0, mid - elapsed_min)

    return {
        "task_id": task.get("task_id"),
        "planned_min": round(planned),
        "p10_min": round(lo),
        "p50_min": round(mid),
        "p90_min": round(hi),
        "remaining_min": None if remaining is None else round(remaining),
        "factors": _factors(X, b),
    }


if __name__ == "__main__":  # smoke test: the T002 example from the contract
    print(predict_task(
        task={"task_id": "T002", "task_type": "Trenching", "planned_estimate_min": 45,
              "target_volume_m3": 36.9},
        env={"weather": "Rainy", "ambient_temp_c": 27, "rain_mm_h": 6.5, "wind_speed_kmh": 18,
             "visibility_m": 400, "light": "day", "ground_condition": "wet"},
        operator={"operator_id": "OP1001", "skill_level": "Intermediate"},
        machine={"machine_id": "EXC001", "machine_age_yrs": 4},
        elapsed_min=21, progress_pct=38.5))
