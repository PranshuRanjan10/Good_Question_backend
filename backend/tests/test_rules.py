"""B2-2: the tuned rules. Thresholds were picked on Nov-Mar windows and checked on April (see
training/train_anomaly.py --eval-only and artifacts/anomaly_metrics.md); these tests pin the
behaviour so a later edit cannot quietly bring the old false alarms back."""
from __future__ import annotations

from app.rules.engine import run_rules
from app.rules.thresholds import THRESHOLDS


def types(**row) -> set[str]:
    return {f["anomaly_type"] for f in run_rules(row)}


def test_over_rev_needs_a_sustained_peak_not_a_blip():
    assert "over_rev" not in types(rpm_max=2150, over_rev_min=1)       # noise: normal windows reach 2199
    assert "over_rev" not in types(rpm_max=2300, over_rev_min=1)       # one minute is a blip
    assert "over_rev" not in types(rpm_max=2150, over_rev_min=4)       # never got past the peak limit
    assert "over_rev" in types(rpm_max=2300, over_rev_min=2)
    assert "over_rev" in types(rpm_max=2210, over_rev_min=5)
    assert THRESHOLDS["rpm_max"] == 2100                               # the spec limit still counts the minutes


def test_harsh_operation_is_repeated_harsh_braking():
    assert "harsh_operation" not in types(harsh_brake_count=4)
    assert "harsh_operation" not in types(harsh_brake_count=3, fast_swing_count=2)
    assert "harsh_operation" in types(harsh_brake_count=5)
    assert "harsh_operation" in types(harsh_brake_count=16, fast_swing_count=0)


def test_fuel_theft_ignores_gauge_noise():
    assert "fuel_theft" not in types(fuel_drop_engine_off_pct=1.4)
    assert "fuel_theft" in types(fuel_drop_engine_off_pct=1.6)


def test_fatigue_is_not_raised_while_on_a_break():
    assert "fatigue" in types(continuous_operation_min=300)
    assert "fatigue" not in types(continuous_operation_min=300, on_break=True)
    assert "fatigue" not in types(continuous_operation_min=240)


def test_out_of_seat_and_idling_stay_quiet_on_a_declared_break():
    assert "operator_out_of_seat" in types(seat_empty_engine_on_min=4)
    assert "operator_out_of_seat" not in types(seat_empty_engine_on_min=4, on_break=True)
    assert "operator_out_of_seat" not in types(seat_empty_engine_on_min=4, seat_empty_outside_break_min=0)
    idle = dict(idle_streak_min=25, idle_ratio=1.0)
    assert "excessive_idling" in types(**idle)
    assert "excessive_idling" not in types(on_break=True, **idle)


def test_after_hours_needs_the_engine_started_outside_the_schedule():
    outside = dict(within_scheduled_hours=0, engine_on=True)
    assert "after_hours_use" in types(engine_started_outside_hours=True, **outside)
    assert "after_hours_use" not in types(engine_started_outside_hours=False, **outside)   # overtime from the shift
    assert "after_hours_use" not in types(within_scheduled_hours=1, engine_started_outside_hours=True, engine_on=True)


def test_a_quiet_normal_window_raises_nothing():
    normal = dict(rpm_max=1950, over_rev_min=0, harsh_brake_count=1, fuel_drop_engine_off_pct=0.3,
                  continuous_operation_min=70, idle_ratio=0.1, idle_streak_min=0, coolant_max_c=85,
                  within_scheduled_hours=1, operator_matches_assigned=1, ground_speed_max_kmh=3.0)
    assert types(**normal) == set()
