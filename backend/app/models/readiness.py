"""
IronSense - Readiness Score (inference).

    from app.models.readiness import compute_readiness
    compute_readiness(row, anomalies)

`row` is the live feature row (same keys as a row of anomaly_windows_5min.csv), `anomalies`
is the list returned by detect_anomalies(). Returns the `readiness_*` block of `assessment`:

    {"readiness_score": 64,
     "readiness_breakdown": {"seatbelt": 100, "proximity": 40, "behaviour": 70,
                             "fatigue": 65, "conditions": 55},
     "drivers": ["Worker inside the danger zone", ...]}

Each sub-score is a transparent rule (an operator can always be told why a number dropped).
Only the five *weights* are learned, by train_readiness.py, from how often each sub-score
preceded a real incident; without the artifact the hand-set weights below are used, so the
function works before any training has run.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "readiness_weights.json"

SUBSCORES = ("seatbelt", "proximity", "behaviour", "fatigue", "conditions")
HAND_WEIGHTS = {"seatbelt": 0.20, "proximity": 0.30, "behaviour": 0.20,
                "fatigue": 0.15, "conditions": 0.15}

_WEIGHTS = None
_LOCK = threading.Lock()

BAD_WEATHER = {"Rainy", "Fog", "Storm", "Dust"}


def load_weights(path: Path | str = ARTIFACT) -> dict:
    global _WEIGHTS
    with _LOCK:
        if _WEIGHTS is None:
            try:
                _WEIGHTS = json.loads(Path(path).read_text())["weights"]
            except (OSError, KeyError, ValueError):
                _WEIGHTS = dict(HAND_WEIGHTS)
    return _WEIGHTS


def _g(row: dict, key: str, default=0.0) -> float:
    v = row.get(key, default)
    if v is None:
        return float(default)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if f != f else f  # NaN -> default


def danger_radius_m(row: dict) -> float:
    """Danger zone widens in poor visibility, matching the safety rules in the spec."""
    bad = row.get("weather") in BAD_WEATHER or row.get("light") == "night"
    return 8.0 if bad else 5.0


def subscores(row: dict, anomalies: list[dict] | None = None) -> tuple[dict, list[str]]:
    """Five 0-100 sub-scores (100 = good) plus the plain-language reasons they dropped."""
    why: list[str] = []
    anomalies = anomalies or []
    kinds = {a.get("anomaly_type") for a in anomalies}

    # --- seatbelt / cab compliance
    s = 100.0
    if _g(row, "seatbelt_off_moving_min") > 0:
        s -= 60; why.append("Seatbelt off while operating")
    elif _g(row, "seatbelt_off_engine_on_min") > 0:
        s -= 25; why.append("Seatbelt off with the engine running")
    if _g(row, "seat_empty_engine_on_min") > 0:
        s -= 35; why.append("Engine running with nobody in the seat")
    seatbelt = s

    # --- proximity
    s = 100.0
    dist = _g(row, "min_person_distance_m", 99.0)
    danger = danger_radius_m(row)
    if dist < danger:
        s -= 55; why.append(f"Person {dist:.1f} m away, inside the {danger:.0f} m danger zone")
    elif dist < danger * 2:
        s -= 20; why.append(f"Person {dist:.1f} m away, in the caution zone")
    if _g(row, "blind_spot_min") > 0:
        s -= 20; why.append("Person in a blind spot")
    if _g(row, "red_zone_min") > 2:
        s -= 15
    proximity = s

    # --- behaviour
    s = 100.0
    if _g(row, "harsh_brake_count") > 0:
        s -= 15 * min(_g(row, "harsh_brake_count"), 2); why.append("Harsh braking")
    if _g(row, "fast_swing_count") > 0:
        s -= 20; why.append("Swing speed above the safe limit")
    if _g(row, "ground_speed_max_kmh") > 6:
        s -= 20; why.append("Travel speed above the site limit")
    if max(_g(row, "pitch_max_deg"), _g(row, "roll_max_deg")) > 15:
        s -= 30; why.append("Working beyond the slope limit")
    behaviour_anoms = kinds & {"fast_swing", "harsh_operation", "over_rev", "overload",
                               "bucket_raised_travel", "overspeed", "slope_exceeded",
                               "excessive_idling", "low_productivity"}
    s -= 10 * len(behaviour_anoms)
    behaviour = s

    # --- fatigue
    s = 100.0
    cont = _g(row, "continuous_operation_min")
    if cont > 240:
        s -= 45; why.append(f"{cont / 60:.1f} h operating without a break")
    elif cont > 180:
        s -= 25; why.append("Approaching 4 h without a break")
    elif cont > 120:
        s -= 10
    if row.get("light") == "night":
        s -= 10
    if "fatigue" in kinds:
        s -= 10
    fatigue = s

    # --- conditions
    s = 100.0
    weather = row.get("weather")
    if weather == "Storm":
        s -= 40; why.append("Storm on site")
    elif weather in BAD_WEATHER:
        s -= 20; why.append(f"{weather} conditions")
    if _g(row, "visibility_m", 9999) < 200:
        s -= 25; why.append("Visibility under 200 m")
    lightning = _g(row, "lightning_distance_km", 999)
    if 0 < lightning < 10:
        s -= 40; why.append(f"Lightning {lightning:.0f} km away: stop work")
    if _g(row, "cab_temp_max_c") > 35:
        s -= 20; why.append("Cab above 35 C: take a cool-down break")
    if row.get("light") == "night":
        s -= 10
    conditions = s

    scores = {"seatbelt": seatbelt, "proximity": proximity, "behaviour": behaviour,
              "fatigue": fatigue, "conditions": conditions}
    return {k: int(max(0.0, min(100.0, v))) for k, v in scores.items()}, why


def compute_readiness(row: dict, anomalies: list[dict] | None = None,
                      weights: dict | None = None) -> dict:
    """Overall 0-100 Readiness Score plus its five parts and the reasons behind them."""
    parts, why = subscores(row, anomalies)
    w = weights or load_weights()
    total = sum(w.values()) or 1.0
    score = sum(parts[k] * w.get(k, 0.0) for k in SUBSCORES) / total
    # A critical safety failure must not be averaged away by four healthy sub-scores.
    score = min(score, min(parts.values()) + 25)
    return {"readiness_score": int(round(max(0.0, min(100.0, score)))),
            "readiness_breakdown": parts,
            "drivers": why[:4]}


if __name__ == "__main__":
    demo = {"seatbelt_off_moving_min": 0, "seatbelt_off_engine_on_min": 0, "seat_empty_engine_on_min": 0,
            "min_person_distance_m": 6.8, "blind_spot_min": 1, "red_zone_min": 1, "harsh_brake_count": 0,
            "fast_swing_count": 0, "ground_speed_max_kmh": 0, "pitch_max_deg": 4.2, "roll_max_deg": 1.1,
            "continuous_operation_min": 112, "cab_temp_max_c": 31, "weather": "Rainy", "light": "day",
            "visibility_m": 400, "lightning_distance_km": None}
    print(compute_readiness(demo, [{"anomaly_type": "excessive_idling"}]))
