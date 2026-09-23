#!/usr/bin/env python3
"""
IronSense - replay harness: pretend to be the simulation.

Reads a shift out of data/datasets/minute_telemetry.parquet and emits it as the exact message
types the frontend simulation will send (frontend_handoff_spec_v1.md section 5), either to a
WebSocket or to a .jsonl file.

Why this exists:
  * the backend can be built and tested today, without waiting for the simulation;
  * every shift in six months of history is a test case, including ones with rare anomalies;
  * it is the demo fallback -- if the live sim dies on stage, replay a recorded shift.

The simulation remains the source of truth for the contract. If this file and the spec ever
disagree, the spec wins and this should be fixed.

Examples:
    # list shifts that contain a given anomaly
    python backend/tools/replay_sim.py --list --anomaly excessive_idling

    # stream a shift to the backend at 60x speed (1 real sec = 1 sim min)
    python backend/tools/replay_sim.py --shift EXC001-2025-04-14 --speed 60

    # dump to a file instead, for offline tests
    python backend/tools/replay_sim.py --shift EXC001-2025-04-14 --out shift.jsonl

Known gap: the synthetic history has no GPS track, so positions here are a simple synthetic
walk inside the work zone. Everything else is real generated telemetry.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "datasets" / "minute_telemetry.parquet"
SCHEMA = "1.0"
PROXIMITY_TRIGGER_M = 20.0
SUBSTEP_S = 10          # operation / proximity cadence, in sim seconds

# Site geometry from the handoff spec's example zones, so replayed positions land somewhere
# plausible on the frontend's map.
ZONE_B = (180, 60, 260, 120)


def iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def jnum(v, nd=2):
    """JSON-safe number: NaN -> None (the contract says null, never a missing key)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, nd)


def jbool(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else bool(v)


class Replayer:
    def __init__(self, df: pd.DataFrame, machine: dict, speed: float, seed: int = 42):
        self.df = df.reset_index(drop=True)
        self.machine = machine
        self.speed = speed
        self.rng = random.Random(seed)
        self.tick = 0
        self.event_no = 0
        self.prev = None
        self.t0 = pd.Timestamp(self.df.timestamp.iloc[0])
        x0, y0, x1, y1 = ZONE_B
        self.pos = [(x0 + x1) / 2, (y0 + y1) / 2]
        self.heading = 90.0

    # ---------------------------------------------------------------- helpers
    def _advance_tick(self, ts) -> int:
        self.tick = int((pd.Timestamp(ts) - self.t0).total_seconds())
        return self.tick

    def _head(self, msg_type: str, ts) -> dict:
        return {"msg_type": msg_type, "schema_version": SCHEMA, "timestamp": iso(ts),
                "sim_tick": self._advance_tick(ts), "machine_id": self.machine["machine_id"]}

    def _step_position(self, row):
        """Synthetic walk: the history has no GPS, but the message shape needs a position."""
        speed = jnum(row.ground_speed_max_kmh, 2) or 0.0
        if speed > 0.1:
            self.heading = (self.heading + self.rng.uniform(-25, 25)) % 360
            d = speed / 3.6 * SUBSTEP_S
            self.pos[0] += d * math.sin(math.radians(self.heading))
            self.pos[1] += d * math.cos(math.radians(self.heading))
            x0, y0, x1, y1 = ZONE_B
            self.pos[0] = min(max(self.pos[0], x0), x1)
            self.pos[1] = min(max(self.pos[1], y0), y1)
        return {"x_m": round(self.pos[0], 1), "y_m": round(self.pos[1], 1)}

    # ---------------------------------------------------------------- messages
    def shift_context(self, row) -> dict:
        tasks = (self.df[self.df.task_id.notna()]
                 .groupby("task_id", sort=False)
                 .agg(task_type=("task_type", "first"), start=("timestamp", "min"),
                      minutes=("timestamp", "size")).reset_index())
        msg = self._head("shift_context", row.timestamp)
        msg.update({
            "time_scale": self.speed,
            "scenario_id": f"replay_{row.shift_id}",
            "seed": 42,
            "site_id": "SITE01",
            "machine": self.machine,
            "operator": {"operator_id": row.operator_id, "skill_level": self.machine.pop("_skill", None)
                         or "Intermediate", "shift_start": iso(row.timestamp)},
            "daily_tasks": [
                {"task_id": t.task_id, "task_type": t.task_type, "zone_id": "ZONE_B",
                 "planned_estimate_min": int(t.minutes), "target_volume_m3": None,
                 "scheduled_start": iso(t.start)}
                for t in tasks.itertuples()
            ],
            "zones": [
                {"zone_id": "ZONE_B", "zone_type": "work_area",
                 "polygon": [[180, 60], [260, 60], [260, 120], [180, 120]]},
                {"zone_id": "LOAD_BAY", "zone_type": "loading_bay",
                 "polygon": [[270, 70], [300, 70], [300, 100], [270, 100]]},
                {"zone_id": "NOGO_1", "zone_type": "no_go_zone",
                 "polygon": [[320, 200], [360, 200], [360, 240], [320, 240]]},
            ],
            "hazards": [
                {"hazard_id": "PL01", "hazard_type": "overhead_power_line",
                 "line": [[150, 140], [300, 140]], "clearance_height_m": 9.5},
                {"hazard_id": "TR01", "hazard_type": "trench_edge", "line": [[190, 90], [250, 90]]},
            ],
            "walkaround_checklist_result": {"completed": True, "issues": []},
            "replay_note": "generated by backend/tools/replay_sim.py, not the live simulation",
        })
        return msg

    def operation(self, row, sub: int) -> dict:
        ts = pd.Timestamp(row.timestamp) + pd.Timedelta(seconds=sub * SUBSTEP_S)
        engine = row.engine_state
        moving = row.mode in ("travel",)
        msg = self._head("operation", ts)
        msg.update({
            "operator_id": row.operator_id,
            "task_id": None if pd.isna(row.task_id) else row.task_id,
            "engine": {"state": engine, "rpm": jnum(row.rpm_mean, 0), "load_pct": jnum(row.load_pct, 1),
                       "fuel_rate_lph": jnum(row.fuel_rate_lph, 2)},
            "motion": {
                "position": self._step_position(row),
                "heading_deg": round(self.heading),
                "ground_speed_kmh": jnum(row.ground_speed_max_kmh, 2) if moving else 0.0,
                "travel_direction": "forward" if moving else "stationary",
                "parking_brake": engine == "off",
                "hydraulic_lockout": False,
                "travel_alarm_active": bool(moving),
            },
            "implement": {"work_mode": self._work_mode(row), "hydraulic_pressure_bar": None},
            "cab": {"seat_occupied": jbool(row.seat_occupied),
                    "seatbelt_fastened": jbool(row.seatbelt_fastened),
                    "door_open": None, "controls_active": jbool(row.seat_occupied)},
        })
        return msg

    @staticmethod
    def _work_mode(row) -> str:
        return {"work": "dig", "travel": "travel", "break": "break"}.get(
            row.mode, "idle" if row.engine_state != "off" else "idle")

    def motion_batch(self, row) -> dict:
        """10 one-second samples. The history is per-minute, so values hold with light jitter."""
        fields = ["pitch_deg", "roll_deg", "longitudinal_accel_ms2", "swing_angle_deg",
                  "swing_rate_dps", "bucket_height_m", "bucket_payload_kg"]
        pitch, roll = jnum(row.pitch_max_deg, 1) or 0.0, jnum(row.roll_max_deg, 1) or 0.0
        swing = jnum(row.swing_rate_p95_dps, 1) or 0.0
        payload = jnum(row.bucket_payload_max_kg, 0) or 0.0
        # Only a digging cycle lifts the bucket (samples 3-5 = swing loaded). Travelling carries
        # it low unless the history recorded bucket-raised travel for this minute.
        digging = row.mode == "work"
        raised_s = int(min(10, round((jnum(row.bucket_raised_travel_s, 1) or 0) / 6)))
        samples, angle = [], 0.0
        for i in range(10):
            if digging:
                height = 1.2 + (2.0 if i in (3, 4, 5) else 0.0)
            else:
                height = 3.0 if (row.mode == "travel" and i < raised_s) else 0.8
            angle = (angle + swing) % 360
            samples.append([
                round(pitch + self.rng.uniform(-0.3, 0.3), 1),
                round(roll + self.rng.uniform(-0.2, 0.2), 1),
                round(jnum(row.longitudinal_accel_min_ms2, 2) or 0.0, 2) if i == 5 else 0.0,
                round(angle, 1), round(swing, 1),
                round(height, 1),
                payload if (digging and i < 6) else 0.0,
            ])
        msg = self._head("motion_batch", row.timestamp)
        msg.update({"start_timestamp": iso(row.timestamp), "sample_interval_s": 1,
                    "fields": fields, "samples": samples})
        return msg

    def proximity(self, row, sub: int) -> dict | None:
        dist = jnum(row.min_person_distance_m, 1)
        if dist is None or dist >= PROXIMITY_TRIGGER_M:
            return None
        ts = pd.Timestamp(row.timestamp) + pd.Timedelta(seconds=sub * SUBSTEP_S)
        bearing = 180 + self.rng.uniform(-25, 25) if jbool(row.person_in_blind_spot) else self.rng.uniform(0, 359)
        msg = self._head("proximity", ts)
        msg["objects"] = [{
            "object_id": "W03", "object_type": "person", "role": "labourer",
            "distance_m": dist, "bearing_deg": round(bearing),
            "relative_speed_ms": round(self.rng.uniform(-1.2, 0.4), 2),
            "sensor": "radar", "detection_confidence": None,
        }]
        return msg

    def environment(self, row) -> dict:
        msg = self._head("environment", row.timestamp)
        msg["environment"] = {
            "weather": row.weather, "ambient_temp_c": jnum(row.ambient_temp_c, 1),
            "humidity_pct": None, "rain_mm_h": jnum(row.rain_mm_h, 1),
            "wind_speed_kmh": jnum(row.wind_speed_kmh, 1), "wind_gust_kmh": None,
            "visibility_m": jnum(row.visibility_m, 0), "light": row.light,
            "dust_index": jnum(row.dust_index, 1), "ground_condition": row.ground_condition,
            "lightning_distance_km": jnum(row.lightning_distance_km, 1),
        }
        return msg

    def status(self, row) -> dict:
        msg = self._head("status", row.timestamp)
        msg.update({
            "engine": {"coolant_temp_c": jnum(row.coolant_temp_c, 1),
                       "fuel_level_pct": jnum(row.fuel_level_pct, 1),
                       "cumulative_hours": jnum(row.cum_engine_hours, 2),
                       "cumulative_idle_hours": jnum(row.cum_idle_hours, 2),
                       "cumulative_fuel_used_l": jnum(row.cum_fuel_used_l, 1),
                       "def_level_pct": None, "battery_voltage_v": None},
            "implement": {"hydraulic_oil_temp_c": jnum(row.hydraulic_oil_temp_c, 1),
                          "cumulative_load_cycles": int(jnum(row.cum_load_cycles, 0) or 0)},
            "cab": {"cab_temp_c": jnum(row.cab_temp_c, 1)},
            "task": {"task_id": None if pd.isna(row.task_id) else row.task_id,
                     "progress_pct": None, "volume_moved_m3": None},
            "diagnostics": {"active_fault_codes": [row.fault_code] if row.fault_code else []},
        })
        return msg

    def event(self, ts, event_type: str, details: dict, operator_id) -> dict:
        self.event_no += 1
        msg = self._head("event", ts)
        msg.update({"operator_id": operator_id, "event_id": f"EV-{self.event_no:04d}",
                    "event_type": event_type, "source": "sensor", "details": details})
        return msg

    def events_for(self, row) -> list[dict]:
        """Diff this minute against the previous one and emit the state changes."""
        out, p = [], self.prev
        ts, op = row.timestamp, row.operator_id
        if p is None:
            if row.engine_state != "off":
                out.append(self.event(ts, "engine_start", {}, op))
        else:
            if (p.engine_state == "off") != (row.engine_state == "off"):
                out.append(self.event(ts, "engine_start" if row.engine_state != "off" else "engine_stop", {}, op))
            if jbool(p.seatbelt_fastened) != jbool(row.seatbelt_fastened):
                out.append(self.event(ts, "seatbelt_fastened" if row.seatbelt_fastened
                                      else "seatbelt_unfastened", {}, op))
            if jbool(p.seat_occupied) != jbool(row.seat_occupied):
                out.append(self.event(ts, "operator_returned" if row.seat_occupied
                                      else "operator_left_seat", {}, op))
            if (p.mode == "break") != (row.mode == "break"):   # spec section 6: break_start / break_end
                out.append(self.event(ts, "break_start" if row.mode == "break" else "break_end", {}, op))
            if (p.mode == "refuel") != (row.mode == "refuel"):
                out.append(self.event(ts, "refuel_start" if row.mode == "refuel" else "refuel_end", {}, op))
            if p.weather != row.weather:
                out.append(self.event(ts, "weather_change", {"from": p.weather, "to": row.weather}, op))
            if str(p.task_id) != str(row.task_id):
                if not pd.isna(p.task_id):
                    out.append(self.event(ts, "task_complete", {"task_id": p.task_id}, op))
                if not pd.isna(row.task_id):
                    out.append(self.event(ts, "task_start",
                                          {"task_id": row.task_id, "task_type": row.task_type}, op))
            if row.fault_code and not p.fault_code:
                out.append(self.event(ts, "fault_code", {"code": row.fault_code}, op))
        if (jnum(row.pitch_max_deg, 1) or 0) > 15 or (jnum(row.roll_max_deg, 1) or 0) > 15:
            out.append(self.event(ts, "tilt_warning",
                                  {"pitch_deg": jnum(row.pitch_max_deg, 1),
                                   "roll_deg": jnum(row.roll_max_deg, 1)}, op))
        if (row.harsh_brake_count or 0) > 0:
            out.append(self.event(ts, "harsh_brake",
                                  {"accel_ms2": jnum(row.longitudinal_accel_min_ms2, 2)}, op))
        lightning = jnum(row.lightning_distance_km, 1)
        if lightning is not None and lightning < 10 and (p is None or (jnum(p.lightning_distance_km, 1) or 99) >= 10):
            out.append(self.event(ts, "lightning_nearby", {"distance_km": lightning}, op))
        return out

    # ---------------------------------------------------------------- driver
    def messages(self):
        """Yield (delay_in_sim_seconds_since_previous, message) in send order."""
        last_tick = 0
        for i, row in enumerate(self.df.itertuples()):
            if i == 0:
                yield 0, self.shift_context(row)
            for ev in self.events_for(row):
                yield max(0, ev["sim_tick"] - last_tick), ev
                last_tick = ev["sim_tick"]
            if i % 15 == 0:
                m = self.environment(row)
                yield max(0, m["sim_tick"] - last_tick), m
                last_tick = m["sim_tick"]
            m = self.status(row)
            yield max(0, m["sim_tick"] - last_tick), m
            last_tick = m["sim_tick"]
            if row.engine_state != "off":
                for sub in range(60 // SUBSTEP_S):
                    m = self.operation(row, sub)
                    yield max(0, m["sim_tick"] - last_tick), m
                    last_tick = m["sim_tick"]
                    prox = self.proximity(row, sub)
                    if prox:
                        yield 0, prox
                    if sub == 0:
                        yield 0, self.motion_batch(row)
            self.prev = row


def load_shift(shift_id: str | None, machine_id: str | None, anomaly: str | None,
               limit_minutes: int | None) -> tuple[pd.DataFrame, dict]:
    if not PARQUET.exists():
        sys.exit(f"missing {PARQUET}\nRun: python data/generate_data.py")
    df = pd.read_parquet(PARQUET)
    if shift_id:
        df = df[df.shift_id == shift_id]
        if df.empty:
            sys.exit(f"no shift {shift_id!r}. Use --list to see options.")
    else:
        pool = df
        if machine_id:
            pool = pool[pool.machine_id == machine_id]
        if anomaly:
            ids = pool[pool.anomaly_label == anomaly].shift_id.unique()
            if len(ids) == 0:
                sys.exit(f"no shift contains {anomaly!r}")
            pool = pool[pool.shift_id == ids[0]]
        else:
            pool = pool[pool.shift_id == pool.shift_id.iloc[0]]
        df = pool
    df = df.sort_values("timestamp")
    if limit_minutes:
        df = df.head(limit_minutes)
    machines = pd.read_csv(ROOT / "data" / "datasets" / "machines.csv")
    m = machines[machines.machine_id == df.machine_id.iloc[0]].iloc[0]
    ops = pd.read_csv(ROOT / "data" / "datasets" / "operators.csv")
    skill = ops[ops.operator_id == df.operator_id.iloc[0]]
    return df, {"machine_id": m.machine_id, "model": m.model, "machine_type": m.machine_type,
                "machine_age_yrs": int(m.machine_age_yrs),
                "_skill": (skill.skill_level.iloc[0] if len(skill) else "Intermediate")}


def list_shifts(machine_id: str | None, anomaly: str | None, n: int = 25):
    df = pd.read_parquet(PARQUET, columns=["shift_id", "machine_id", "operator_id", "timestamp",
                                           "anomaly_label"])
    if machine_id:
        df = df[df.machine_id == machine_id]
    if anomaly:
        keep = df[df.anomaly_label == anomaly].shift_id.unique()
        df = df[df.shift_id.isin(keep)]
    g = df.groupby("shift_id").agg(
        operator=("operator_id", "first"), start=("timestamp", "min"), minutes=("timestamp", "size"),
        anomalies=("anomaly_label", lambda s: ",".join(sorted(set(s) - {"normal"})) or "-"))
    print(g.sort_values("start").head(n).to_string())
    print(f"\n{len(g)} shifts match. Replay one with --shift <id>")


async def stream(msgs, url: str, speed: float, verbose: bool):
    try:
        import websockets
    except ImportError:
        sys.exit("pip install websockets  (or use --out to write a file instead)")
    sent = 0
    async with websockets.connect(url, max_size=None) as ws:
        print(f"connected to {url} (speed {speed}x sim time)")
        for delay, msg in msgs:
            if delay:
                await asyncio.sleep(delay / speed)
            await ws.send(json.dumps(msg))
            sent += 1
            if verbose:
                print(f"  -> {msg['msg_type']:<14} tick={msg['sim_tick']}")
            try:                                    # surface backend rejections immediately
                reply = await asyncio.wait_for(ws.recv(), timeout=0.001)
                print(f"  <- {reply[:200]}")
            except (asyncio.TimeoutError, Exception):
                pass
    print(f"done: {sent} messages")


def main():
    ap = argparse.ArgumentParser(description="Replay recorded telemetry as simulation messages.")
    ap.add_argument("--shift", help="shift_id, e.g. EXC001-2025-04-14")
    ap.add_argument("--machine", help="pick a shift for this machine")
    ap.add_argument("--anomaly", help="pick a shift containing this anomaly_label")
    ap.add_argument("--list", action="store_true", help="list shifts and exit")
    ap.add_argument("--speed", type=float, default=60.0, help="sim seconds per real second (default 60)")
    ap.add_argument("--url", default="ws://localhost:8000/ws/telemetry")
    ap.add_argument("--out", help="write .jsonl instead of streaming")
    ap.add_argument("--limit-minutes", type=int, help="only the first N minutes of the shift")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if a.list:
        list_shifts(a.machine, a.anomaly)
        return

    df, machine = load_shift(a.shift, a.machine, a.anomaly, a.limit_minutes)
    print(f"shift {df.shift_id.iloc[0]}  {df.machine_id.iloc[0]}/{df.operator_id.iloc[0]}  "
          f"{len(df)} minutes  anomalies: "
          f"{','.join(sorted(set(df.anomaly_label) - {'normal'})) or 'none'}")
    msgs = list(Replayer(df, machine, a.speed).messages())
    print(f"{len(msgs)} messages to send")

    if a.out:
        with open(a.out, "w") as fh:
            for _, m in msgs:
                fh.write(json.dumps(m) + "\n")
        print(f"wrote {a.out}")
        counts = pd.Series([m["msg_type"] for _, m in msgs]).value_counts()
        print(counts.to_string())
        return

    asyncio.run(stream(msgs, a.url, a.speed, a.verbose))


if __name__ == "__main__":
    main()
