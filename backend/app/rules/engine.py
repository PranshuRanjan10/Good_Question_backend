"""Explainable rule engine: one function per known anomaly pattern.

Each rule takes a feature `row` (dict with the same keys as a row of
anomaly_windows_5min.csv - see datasets/README.md) and returns a finding
dict or None. Runs in the fast path (per telemetry tick) and also feeds the
offline evaluation in training/train_anomaly.py so "rules-only" numbers are
comparable to "rules + LightGBM".

anomaly_type values match datasets/README.md's `anomaly_label` vocabulary
exactly, so rule hits and model predictions can be compared/merged 1:1.
"""

from __future__ import annotations
from typing import Callable

from app.rules.thresholds import THRESHOLDS, PAYLOAD_KG_PER_SIZE_FACTOR


def _finding(anomaly_type: str, message: str, severity: str = "warning") -> dict:
    return {"anomaly_type": anomaly_type, "method": "rule", "severity": severity, "message": message}


def rule_excessive_idling(row: dict) -> dict | None:
    # A 5-minute window can't hold a 20-minute idle, and short truck-swap waits push the
    # window ratio over 0.6 all the time, so the rule reads the unbroken idle streak.
    # Idling during a declared break is fuel waste, not this anomaly (training agrees).
    if row.get("on_break"):
        return None
    streak = row.get("idle_streak_min", 0) or 0
    if streak >= THRESHOLDS["idle_min"] and row.get("idle_ratio", 0) > THRESHOLDS["idle_ratio"]:
        return _finding("excessive_idling", f"Idling for {streak:.0f} min")
    return None


def rule_over_rev(row: dict) -> dict | None:
    if row.get("rpm_max", 0) > THRESHOLDS["rpm_max"]:
        return _finding("over_rev", f"RPM {row['rpm_max']:.0f} > {THRESHOLDS['rpm_max']}", "danger")
    return None


def rule_fast_swing(row: dict) -> dict | None:
    if row.get("swing_rate_p95_dps", 0) > THRESHOLDS["swing_deg_s_max"]:
        return _finding("fast_swing", f"Swing rate p95 {row['swing_rate_p95_dps']:.0f} deg/s")
    return None


def rule_harsh_operation(row: dict) -> dict | None:
    if row.get("harsh_brake_count", 0) > 0 and row.get("fast_swing_count", 0) > 0:
        return _finding("harsh_operation", "Harsh braking combined with fast swings")
    return None


def rule_bucket_raised_travel(row: dict) -> dict | None:
    if row.get("bucket_raised_travel_s", 0) > THRESHOLDS["bucket_raised_travel_s"] and row.get("ground_speed_max_kmh", 0) > 0:
        return _finding("bucket_raised_travel", f"{row['bucket_raised_travel_s']:.0f}s traveling with bucket raised", "danger")
    return None


def rule_overspeed(row: dict) -> dict | None:
    if row.get("ground_speed_max_kmh", 0) > THRESHOLDS["speed_kmh_max"]:
        return _finding("overspeed", f"Speed {row['ground_speed_max_kmh']:.1f} km/h")
    return None


def rule_slope_exceeded(row: dict) -> dict | None:
    pitch = abs(row.get("pitch_max_deg", 0))
    roll = abs(row.get("roll_max_deg", 0))
    if pitch > THRESHOLDS["pitch_roll_deg_max"] or roll > THRESHOLDS["pitch_roll_deg_max"]:
        return _finding("slope_exceeded", f"Pitch/roll {max(pitch, roll):.0f} deg", "danger")
    return None


def rule_overload(row: dict) -> dict | None:
    size_factor = row.get("size_factor", 1.0)
    limit = PAYLOAD_KG_PER_SIZE_FACTOR * size_factor
    if row.get("bucket_payload_max_kg", 0) > limit or row.get("overload_count", 0) > 0:
        return _finding("overload", f"Payload {row.get('bucket_payload_max_kg', 0):.0f} kg > {limit:.0f} kg limit")
    return None


def rule_overheating(row: dict) -> dict | None:
    if row.get("coolant_max_c", 0) > THRESHOLDS["coolant_c_max"]:
        return _finding("overheating", f"Coolant {row['coolant_max_c']:.0f} C", "danger")
    return None


def rule_fuel_theft(row: dict) -> dict | None:
    if row.get("fuel_drop_engine_off_pct", 0) > THRESHOLDS["fuel_drop_off_pct"]:
        return _finding("fuel_theft", f"Fuel dropped {row['fuel_drop_engine_off_pct']:.1f}% with engine off", "danger")
    return None


def rule_operator_out_of_seat(row: dict) -> dict | None:
    # Break minutes inside the same window don't count (engine-idling on a break is normal in
    # training); the live row carries the out-of-seat time outside breaks separately.
    empty = row.get("seat_empty_outside_break_min", row.get("seat_empty_engine_on_min", 0))
    if empty > 0 and not row.get("on_break"):
        return _finding("operator_out_of_seat", f"{empty:.1f} min seat empty, engine on", "danger")
    return None


def rule_seatbelt_off_while_moving(row: dict) -> dict | None:
    if row.get("seatbelt_off_moving_min", 0) > 0:
        return _finding("seatbelt_off_while_moving", f"{row['seatbelt_off_moving_min']:.1f} min unbelted while moving", "danger")
    return None


def rule_fatigue(row: dict) -> dict | None:
    if row.get("continuous_operation_min", 0) > THRESHOLDS["continuous_op_min_max"]:
        return _finding("fatigue", f"{row['continuous_operation_min']:.0f} min without a break")
    return None


def rule_after_hours_use(row: dict) -> dict | None:
    # Arrives as 0/1. Overtime straight after the shift is normal (training agrees); the
    # pattern is the engine being started again outside the schedule.
    if (not row.get("within_scheduled_hours", 1) and row.get("engine_started_outside_hours", True)
            and row.get("engine_on", True)):
        return _finding("after_hours_use", "Operating outside scheduled hours")
    return None


def rule_unauthorized_operator(row: dict) -> dict | None:
    if not row.get("operator_matches_assigned", 1):
        return _finding("unauthorized_operator", "Operator differs from assigned operator")
    return None


def rule_unsafe_refuelling(row: dict) -> dict | None:
    if row.get("engine_on", False) and row.get("fuel_rising", False):
        return _finding("unsafe_refuelling", "Refuelling with engine running", "danger")
    return None


def rule_sensor_dropout(row: dict) -> dict | None:
    if row.get("sensor_dropout_min", 0) > 0:
        return _finding("sensor_dropout", "Sensor dropout in this window", "info")
    return None


RULES: list[Callable[[dict], dict | None]] = [
    rule_excessive_idling,
    rule_over_rev,
    rule_fast_swing,
    rule_harsh_operation,
    rule_bucket_raised_travel,
    rule_overspeed,
    rule_slope_exceeded,
    rule_overload,
    rule_overheating,
    rule_fuel_theft,
    rule_operator_out_of_seat,
    rule_seatbelt_off_while_moving,
    rule_fatigue,
    rule_after_hours_use,
    rule_unauthorized_operator,
    rule_unsafe_refuelling,
    rule_sensor_dropout,
]


def run_rules(row: dict) -> list[dict]:
    findings = []
    for rule in RULES:
        result = rule(row)
        if result is not None:
            findings.append(result)
    return findings
