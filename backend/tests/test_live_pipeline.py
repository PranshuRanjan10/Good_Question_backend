"""Live-pipeline regression tests: the director-console scenarios (spec section 9) plus the
bugs fixed on fix/live-pipeline. Run from backend/:

    ../.venv/bin/python -m pytest -q

They drive StateManager + DecisionLayer directly (no server, no DB), so they take seconds.
The replay parity test needs data/datasets/ and is skipped when it isn't there.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.decision.layer import DecisionLayer
from app.ingest.schemas import parse_message
from app.models.anomaly import detect_anomalies
from app.models.task_time import predict_task
from app.state.manager import StateManager

ROOT = Path(__file__).resolve().parents[2]
SPEC = (ROOT / "docs" / "frontend_handoff_spec_v1.md").read_text()
EXAMPLES = {}
for block in re.findall(r"```json\n(\{.*?\})\n```", SPEC, re.S):
    try:
        d = json.loads(block)
        EXAMPLES.setdefault(d["msg_type"], d)
    except (ValueError, KeyError):
        pass
T0 = datetime(2025, 5, 1, 8, 0, tzinfo=timezone.utc)


def at(sec: float) -> str:
    return (T0 + timedelta(seconds=sec)).strftime("%Y-%m-%dT%H:%M:%SZ")


def msg(kind: str, sec: float, **updates) -> dict:
    m = copy.deepcopy(EXAMPLES[kind])
    m["timestamp"], m["sim_tick"] = at(sec), int(sec)
    for path, value in updates.items():
        cur = m
        *parents, last = path.split("__")
        for p in parents:
            cur = cur[p]
        cur[last] = value
    return m


def event(sec: float, event_type: str, **details) -> dict:
    return {"msg_type": "event", "schema_version": "1.0", "timestamp": at(sec), "sim_tick": int(sec),
            "machine_id": "EXC001", "operator_id": "OP1001", "event_id": f"EV{sec}",
            "event_type": event_type, "source": "director_console", "details": details}


class Site:
    """A running machine at the start of a shift, plus helpers to feed it messages."""

    def __init__(self, weather: str = "Cloudy"):
        self.sm = StateManager()
        self.dl = DecisionLayer(predict_task=predict_task)
        self.sec = 0
        sc = msg("shift_context", 0, operator__shift_start=at(0))
        sc["daily_tasks"][0].update(task_id="T002", task_type="Trenching", scheduled_start=at(60))
        env = msg("environment", 1, environment__weather=weather, environment__rain_mm_h=0,
                  environment__ground_condition="dry", environment__visibility_m=8000)
        self.feed(sc, env, event(2, "engine_start"), event(60, "task_start", task_id="T002"))
        self.sec = 60
        for _ in range(6):
            self.op()

    def feed(self, *messages) -> None:
        async def go():
            for m in messages:
                await self.sm.ingest(parse_message(m), m)
        asyncio.run(go())

    def op(self, dt: float = 10, **updates) -> None:
        self.sec += dt
        self.feed(msg("operation", self.sec, **updates))

    @property
    def state(self):
        return self.sm.get("EXC001")

    def assess(self):
        return self.dl.assess(self.state, {})

    def alerts(self, alert_type: str | None = None) -> list[dict]:
        out = self.assess().payload["alerts"]
        return [a for a in out if alert_type is None or a["alert_type"] == alert_type]


def person(distance: float, bearing: float = 185) -> dict:
    return {"object_id": "W03", "object_type": "person", "role": "labourer", "distance_m": distance,
            "bearing_deg": bearing, "relative_speed_ms": -0.8, "sensor": "radar", "detection_confidence": None}


# ---------------------------------------------------------------- director scenarios

def test_s1_worker_in_blind_spot_is_critical_and_keeps_its_id():
    site = Site()
    site.sec += 1
    site.feed(msg("proximity", site.sec, objects=[person(3.0)]))
    first = site.alerts("proximity_person_blind_spot")
    assert first and first[0]["severity"] == "critical"
    site.sec += 1
    site.feed(msg("proximity", site.sec, objects=[person(2.5)]))
    again = site.alerts("proximity_person_blind_spot")
    assert again[0]["alert_id"] == first[0]["alert_id"], "same condition must keep the same alert_id"


def test_stale_proximity_clears_the_alert():
    site = Site()
    site.sec += 1
    site.feed(msg("proximity", site.sec, objects=[person(3.0)]))
    assert site.alerts("proximity_person_blind_spot")
    site.op(dt=30)                          # no proximity for 30 s: nobody is in range
    assert not site.alerts("proximity_person_blind_spot")


def test_danger_zone_widens_in_rain():
    site = Site(weather="Rainy")
    site.sec += 1
    site.feed(msg("proximity", site.sec, objects=[person(6.5, bearing=0)]))
    payload = site.assess().payload
    assert payload["zones"]["danger_radius_m"] == 8.0
    assert any(a["alert_type"] == "proximity_person_danger" for a in payload["alerts"])


def test_s2_seatbelt_escalates_while_digging_in_place():
    site = Site()
    seen = []
    for _ in range(8):
        site.op(cab__seatbelt_fastened=False)          # stationary, work_mode "dig"
        seen += site.alerts("seatbelt_off_moving")
    severities = [a["severity"] for a in seen]
    assert severities[0] == "warning" and "high" in severities and severities[-1] == "critical"
    assert len({a["alert_id"] for a in seen}) == 1


def test_s3_out_of_cab_with_engine_running_but_not_on_break():
    site = Site()
    site.op(cab__seat_occupied=False, engine__state="idle", implement__work_mode="idle")
    assert site.alerts("operator_out_of_cab")
    site.feed(event(site.sec + 1, "break_start"))
    site.op(cab__seat_occupied=False, engine__state="idle", implement__work_mode="break")
    assert not site.alerts("operator_out_of_cab"), "a declared break is not an out-of-cab alert"


def test_engine_stop_ends_engine_running_alerts():
    site = Site()
    site.op(cab__seat_occupied=False, engine__state="idle", implement__work_mode="idle")
    assert site.alerts("operator_out_of_cab")
    site.feed(event(site.sec + 5, "engine_stop"))      # sim stops sending operation messages now
    assert not site.alerts("operator_out_of_cab")


def test_s6_lightning_event_stops_work():
    site = Site()
    site.feed(event(site.sec + 1, "lightning_nearby", distance_km=6))
    assert site.alerts("lightning_nearby")[0]["severity"] == "critical"


def test_s7_tilt_from_motion_batch():
    site = Site()
    site.sec += 10
    mb = msg("motion_batch", site.sec, start_timestamp=at(site.sec - 10))
    mb["samples"] = [[17.5, 3.0, 0, 30, 10, 1.0, 0]] * 10
    site.feed(mb)
    assert site.alerts("tilt_warning")


def test_digging_is_not_idling():
    site = Site()
    for _ in range(30):
        site.op()                                # 5 minutes of "dig" at 1650 rpm, 0 km/h
    row = site.state.to_feature_row()
    assert row["idle_ratio"] == 0.0
    assert not [a for a in detect_anomalies(row, {}) if a["anomaly_type"] == "excessive_idling"]


def test_s4_long_idle_without_truck_is_flagged_with_reason():
    site = Site()
    site.state.shift_context.daily_tasks[0].task_type = "Material Loading"   # a truck-served task
    site.state.buffer.context["daily_tasks"][0]["task_type"] = "Material Loading"
    for _ in range(6 * 25):                      # 25 min idling, no truck in range
        site.op(engine__state="idle", engine__rpm=900, implement__work_mode="idle")
    anomalies = detect_anomalies(site.state.to_feature_row(), {})
    idle = [a for a in anomalies if a["anomaly_type"] == "excessive_idling"]
    assert idle and "Likely waiting for haul truck" in idle[0]["explanation"]


def test_fatigue_clock_survives_short_idles():
    site = Site()
    for i in range(6 * 60):                      # an hour: mostly digging, idle every 10 min
        site.op(**({"engine__state": "idle", "implement__work_mode": "idle"} if i % 60 == 0 else {}))
    assert site.state.to_feature_row()["continuous_operation_min"] >= 59


def test_s9_fuel_theft_with_engine_off():
    site = Site()
    site.feed(event(site.sec + 1, "engine_stop"))
    level = 60.0
    for k in range(4):                           # status every ~5 min while off (with jitter)
        site.feed(msg("status", site.sec + 60 + k * 307, engine__fuel_level_pct=level))
        level -= 3.0
    row = site.state.to_feature_row()
    assert row["fuel_drop_engine_off_pct"] >= 2.5
    assert any(a["anomaly_type"] == "fuel_theft" for a in detect_anomalies(row, {}))


def test_refuel_with_engine_off_is_not_unsafe():
    site = Site()
    site.feed(msg("status", site.sec + 1, engine__fuel_level_pct=20.0))
    site.feed(event(site.sec + 2, "engine_stop"))
    site.feed(msg("status", site.sec + 300, engine__fuel_level_pct=95.0))   # refuelled while off
    site.feed(event(site.sec + 310, "engine_start"))
    site.sec += 320
    site.op()
    assert not site.alerts("refuel_engine_on")


# ---------------------------------------------------------------- task time

@pytest.mark.parametrize("elapsed,progress", [(None, None), (2, 21), (10, 20), (20, 80), (40, 30), (60, 95)])
def test_task_prediction_range_is_ordered(elapsed, progress):
    r = predict_task({"task_id": "T002", "task_type": "Trenching", "planned_estimate_min": 45},
                     {"weather": "Cloudy", "light": "day", "ground_condition": "dry"},
                     {"skill_level": "Intermediate"}, {"machine_age_yrs": 4},
                     elapsed_min=elapsed, progress_pct=progress)
    assert r["p10_min"] <= r["p50_min"] <= r["p90_min"]
    if elapsed is not None:
        assert r["p10_min"] >= elapsed and r["remaining_min"] is not None


def test_assessment_has_remaining_time_once_task_started():
    site = Site()
    site.feed(msg("status", site.sec + 1, task__task_id="T002", task__progress_pct=20.0))
    pred = site.assess().payload["task_prediction"]
    assert pred and pred["remaining_min"] is not None


# ---------------------------------------------------------------- live row == training row

REPLAY = ROOT / "backend" / "tools" / "replay_sim.py"
WINDOWS = ROOT / "data" / "datasets" / "anomaly_windows_5min.csv"


@pytest.mark.skipif(not WINDOWS.exists(), reason="needs data/datasets (run data/generate_data.py)")
def test_live_feature_row_matches_training(tmp_path):
    import subprocess
    import sys

    import pandas as pd

    shift = "EXC001-2024-11-28"
    out = tmp_path / "shift.jsonl"
    subprocess.run([sys.executable, str(REPLAY), "--shift", shift, "--out", str(out)], check=True,
                   capture_output=True)
    sm, rows, last = StateManager(), {}, None

    async def go():
        nonlocal last
        for line in out.read_text().splitlines():
            raw = json.loads(line)
            st = await sm.ingest(parse_message(raw), raw)
            if st.latest_operation is None:
                continue
            bucket = pd.Timestamp(st.sim_now()).floor("5min")
            if last is not None and bucket != last:
                rows[last.tz_localize(None)] = st.to_feature_row()
            last = bucket
    asyncio.run(go())

    w = pd.read_csv(WINDOWS, parse_dates=["timestamp"])
    w = w[(w.machine_id == "EXC001") & (w.timestamp.dt.date == pd.Timestamp("2024-11-28").date())]
    w = w.set_index("timestamp")
    common = [ts for ts in rows if ts in w.index]
    assert len(common) > 90
    for feature, min_match in (("idle_ratio", 0.99), ("continuous_operation_min", 0.9),
                               ("seatbelt_off_moving_min", 0.99), ("within_scheduled_hours", 0.99)):
        live = pd.Series([rows[t][feature] for t in common], dtype=float)
        train = w.loc[common, feature].astype(float).reset_index(drop=True)
        match = ((live - train).abs() <= 0.05 * train.abs().clip(lower=1)).mean()
        assert match >= min_match, f"{feature}: only {match:.0%} of windows match training"
