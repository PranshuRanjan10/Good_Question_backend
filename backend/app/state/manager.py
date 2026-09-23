"""Per-machine state: shift context, rolling buffers for each message type,
event-driven counters (continuous operation, seatbelt/seat compliance), and
the feature-row builder that turns all of it into one row shaped exactly
like anomaly_windows_5min.csv (see datasets/README.md), which is the shared
boundary consumed by the rule engine, the anomaly model, and Pranshu's
readiness/task-time functions.
"""

from __future__ import annotations
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta, UTC

from app.ingest.schemas import (
    ShiftContextMessage, OperationMessage, MotionBatchMessage,
    ProximityMessage, EnvironmentMessage, StatusMessage, EventMessage,
)

WINDOW = timedelta(minutes=5)
IDLE_SPEED_THRESHOLD_KMH = 0.5
OVER_REV_RPM = 2100
FAST_SWING_DPS = 55
BUCKET_RAISED_M = 2.5
OVERLOAD_KG = 2000
PROXIMITY_STALE_S = 5.0
BLIND_SPOT_BEARING = (135.0, 225.0)  # roughly behind the machine, per spec bearing convention

_FRESH = lambda ts, now: (now - ts) <= WINDOW


class MachineState:
    def __init__(self, machine_id: str):
        self.machine_id = machine_id

        self.shift_context: ShiftContextMessage | None = None

        self.operation_ticks: deque[tuple[datetime, OperationMessage]] = deque()
        self.motion_samples: deque[tuple[datetime, dict, bool]] = deque()  # (ts, sample, traveling)
        self.status_readings: deque[tuple[datetime, StatusMessage]] = deque()
        self.events: deque[tuple[datetime, EventMessage]] = deque()

        self.latest_proximity: ProximityMessage | None = None
        self.latest_proximity_ts: datetime | None = None
        self.latest_environment: EnvironmentMessage | None = None
        self.latest_status: StatusMessage | None = None
        self.latest_operation: OperationMessage | None = None

        self.continuous_op_since: datetime | None = None
        self.last_break_end: datetime | None = None
        self._sim_now: datetime | None = None  # latest timestamp seen across any message type

    # ---- ingestion ----

    def _trim(self, buf: deque, now: datetime) -> None:
        while buf and (now - buf[0][0]) > WINDOW:
            buf.popleft()

    def _touch(self, ts: datetime) -> None:
        """Every message advances sim time; this is a simulated site running
        on sim_tick, not the wall clock, so freshness/cooldown/window math
        must all key off the latest message timestamp, not datetime.now()."""
        if self._sim_now is None or ts > self._sim_now:
            self._sim_now = ts

    def sim_now(self) -> datetime:
        return self._sim_now or datetime.now(UTC)

    def on_shift_context(self, msg: ShiftContextMessage) -> None:
        self.shift_context = msg
        self._touch(msg.timestamp)

    def on_operation(self, msg: OperationMessage) -> None:
        self.latest_operation = msg
        now = msg.timestamp
        self._touch(now)
        self.operation_ticks.append((now, msg))
        self._trim(self.operation_ticks, now)
        self._update_continuous_op(msg)

    def on_motion_batch(self, msg: MotionBatchMessage) -> None:
        now = msg.timestamp
        self._touch(now)
        traveling = bool(self.latest_operation and (self.latest_operation.motion.ground_speed_kmh or 0) > IDLE_SPEED_THRESHOLD_KMH)
        for i, record in enumerate(msg.as_records()):
            ts = msg.start_timestamp + timedelta(seconds=msg.sample_interval_s * i)
            self.motion_samples.append((ts, record, traveling))
        self._trim(self.motion_samples, now)

    def on_proximity(self, msg: ProximityMessage) -> None:
        self.latest_proximity = msg
        self.latest_proximity_ts = msg.timestamp
        self._touch(msg.timestamp)

    def on_environment(self, msg: EnvironmentMessage) -> None:
        self.latest_environment = msg
        self._touch(msg.timestamp)

    def on_status(self, msg: StatusMessage) -> None:
        self.latest_status = msg
        now = msg.timestamp
        self._touch(now)
        self.status_readings.append((now, msg))
        self._trim(self.status_readings, now)

    def on_event(self, msg: EventMessage) -> None:
        now = msg.timestamp
        self._touch(now)
        self.events.append((now, msg))
        self._trim(self.events, now)
        if msg.event_type == "engine_start" and self.continuous_op_since is None:
            self.continuous_op_since = now
        elif msg.event_type in ("engine_stop", "break_start"):
            self.continuous_op_since = None
            self.last_break_end = now if msg.event_type == "break_start" else self.last_break_end
        elif msg.event_type == "break_end":
            self.continuous_op_since = now
            self.last_break_end = now

    def _update_continuous_op(self, msg: OperationMessage) -> None:
        """Fallback for when engine_start/break events aren't sent: infer from
        engine state directly on the operation tick."""
        if msg.engine.state == "running":
            if self.continuous_op_since is None:
                self.continuous_op_since = msg.timestamp
        else:
            self.continuous_op_since = None

    def continuous_op_min(self, now: datetime) -> float:
        if self.continuous_op_since is None:
            return 0.0
        return (now - self.continuous_op_since).total_seconds() / 60.0

    # ---- feature row ----

    def to_feature_row(self, now: datetime | None = None) -> dict:
        """One row with the columns of anomaly_windows_5min.csv (datasets/README.md),
        plus the extra fields app/models/readiness.py reads (min_person_distance_m,
        blind_spot_min, red_zone_min, cab_temp_max_c, weather, light, visibility_m,
        lightning_distance_km, seatbelt_off_engine_on_min)."""
        now = now or self.sim_now()
        if self.latest_operation is None:
            return {"machine_id": self.machine_id, "timestamp": now}

        ticks = [m for _, m in self.operation_ticks]
        n_ticks = len(ticks) or 1
        tick_min = 10.0 / 60.0  # operation sent every 10s per spec section 4

        def tick_minutes(pred) -> float:
            return sum(tick_min for m in ticks if pred(m))

        idle_ticks = tick_minutes(lambda m: m.engine.state != "off" and (m.motion.ground_speed_kmh or 0) <= IDLE_SPEED_THRESHOLD_KMH)
        window_min = max(n_ticks * tick_min, tick_min)

        rpms = [m.engine.rpm for m in ticks if m.engine.rpm is not None]
        load_pcts = [m.engine.load_pct for m in ticks if m.engine.load_pct is not None]
        fuel_rates = [m.engine.fuel_rate_lph for m in ticks if m.engine.fuel_rate_lph is not None]
        speeds = [m.motion.ground_speed_kmh or 0 for m in ticks]

        motion_in_window = [(ts, rec) for ts, rec, _ in self.motion_samples]
        swing_rates = sorted(rec.get("swing_rate_dps", 0) or 0 for _, rec in motion_in_window)
        pitches = [abs(rec.get("pitch_deg", 0) or 0) for _, rec in motion_in_window]
        rolls = [abs(rec.get("roll_deg", 0) or 0) for _, rec in motion_in_window]
        payloads = [rec.get("bucket_payload_kg", 0) or 0 for _, rec in motion_in_window]
        heights = [(ts, rec.get("bucket_height_m", 0) or 0, trav) for ts, rec, trav in self.motion_samples]

        swing_p95 = swing_rates[int(0.95 * (len(swing_rates) - 1))] if swing_rates else 0.0
        fast_swing_count = sum(1 for r in swing_rates if r > FAST_SWING_DPS)
        overload_count = sum(1 for p in payloads if p > OVERLOAD_KG)
        bucket_raised_travel_s = sum(
            (self.motion_samples[i][0] - self.motion_samples[i - 1][0]).total_seconds() if i > 0 else 1.0
            for i, (ts, h, trav) in enumerate(heights) if h > BUCKET_RAISED_M and trav
        )

        harsh_brake_count = sum(1 for _, e in self.events if e.event_type == "harsh_brake")
        tilt_events = sum(1 for _, e in self.events if e.event_type == "tilt_warning")

        status_in_window = [s for _, s in self.status_readings]
        coolant_vals = [s.engine.coolant_temp_c for s in status_in_window if s.engine.coolant_temp_c is not None]
        hyd_oil_vals = [s.implement.hydraulic_oil_temp_c for s in status_in_window if s.implement.hydraulic_oil_temp_c is not None]
        cab_temp_vals = [s.cab.cab_temp_c for s in status_in_window if s.cab.cab_temp_c is not None]

        cycles_delta = 0
        if len(status_in_window) >= 2:
            first, last = status_in_window[0], status_in_window[-1]
            if first.implement.cumulative_load_cycles is not None and last.implement.cumulative_load_cycles is not None:
                cycles_delta = max(0, last.implement.cumulative_load_cycles - first.implement.cumulative_load_cycles)
        cycles_per_hour = cycles_delta / (window_min / 60.0) if window_min else 0.0
        fuel_per_cycle_l = (sum(fuel_rates) / len(fuel_rates) * window_min / 60.0) / max(cycles_delta, 1) if fuel_rates else 0.0

        fuel_levels = [(ts, s.engine.fuel_level_pct) for ts, s in self.status_readings if s.engine.fuel_level_pct is not None]
        fuel_drop_engine_off_pct = 0.0
        if len(fuel_levels) >= 2 and self.latest_operation.engine.state == "off":
            first_level, last_level = fuel_levels[0][1], fuel_levels[-1][1]
            if last_level < first_level:
                fuel_drop_engine_off_pct = first_level - last_level

        seat_empty_engine_on_min = tick_minutes(lambda m: m.engine.state != "off" and m.cab.seat_occupied is False)
        seatbelt_off_moving_min = tick_minutes(
            lambda m: (m.motion.ground_speed_kmh or 0) > IDLE_SPEED_THRESHOLD_KMH and m.cab.seatbelt_fastened is False
        )
        seatbelt_off_engine_on_min = tick_minutes(
            lambda m: m.engine.state != "off" and m.cab.seatbelt_fastened is False
        )

        sensor_dropout_min = tick_minutes(lambda m: m.engine.rpm is None or (m.engine.load_pct is None))

        assigned_operator_id = self.shift_context.operator.operator_id if self.shift_context else None
        operator_id = self.latest_operation.operator_id
        operator_matches_assigned = (assigned_operator_id is None) or (operator_id == assigned_operator_id)

        hour = now.hour
        within_scheduled_hours = 6 <= hour < 20  # TODO: replace with shift_context.operator.shift_start + shift length once spec'd

        # proximity
        min_person_distance_m = 99.0
        blind_spot_min = 0.0
        red_zone_min = 0.0
        if self.latest_proximity and self.latest_proximity_ts and _FRESH(self.latest_proximity_ts, now):
            people = [o for o in self.latest_proximity.objects if o.object_type == "person"]
            if people:
                min_person_distance_m = min(o.distance_m for o in people)
                if any(BLIND_SPOT_BEARING[0] <= o.bearing_deg <= BLIND_SPOT_BEARING[1] for o in people):
                    blind_spot_min = 1.0 / 60.0  # one proximity tick (~1s), per spec 5.4 cadence
                danger = 8.0 if self._degraded_visibility() else 5.0
                if any(o.distance_m < danger for o in people):
                    red_zone_min = 1.0 / 60.0

        env = self.latest_environment.environment if self.latest_environment else None

        return {
            "machine_id": self.machine_id,
            "operator_id": operator_id,
            "assigned_operator_id": assigned_operator_id,
            "task_id": self.latest_operation.task_id,
            "timestamp": now,
            "size_factor": 1.0,  # TODO: join against machines.csv once loaded at startup

            "idle_ratio": min(idle_ticks / window_min, 1.0) if window_min else 0.0,
            "fuel_rate_lph": sum(fuel_rates) / len(fuel_rates) if fuel_rates else 0.0,
            "fuel_per_cycle_l": fuel_per_cycle_l,
            "cycles_per_hour": cycles_per_hour,
            "rpm_max": max(rpms) if rpms else 0.0,
            "over_rev_min": tick_minutes(lambda m: (m.engine.rpm or 0) > OVER_REV_RPM),
            "load_pct_mean": sum(load_pcts) / len(load_pcts) if load_pcts else 0.0,
            "swing_rate_p95_dps": swing_p95,
            "fast_swing_count": fast_swing_count,
            "harsh_brake_count": harsh_brake_count,
            "ground_speed_max_kmh": max(speeds) if speeds else 0.0,
            "bucket_raised_travel_s": bucket_raised_travel_s,
            "pitch_max_deg": max(pitches) if pitches else 0.0,
            "roll_max_deg": max(rolls) if rolls else 0.0,
            "bucket_payload_max_kg": max(payloads) if payloads else 0.0,
            "overload_count": overload_count,
            "coolant_max_c": max(coolant_vals) if coolant_vals else 0.0,
            "hydraulic_oil_max_c": max(hyd_oil_vals) if hyd_oil_vals else 0.0,
            "fuel_drop_engine_off_pct": fuel_drop_engine_off_pct,
            "seat_empty_engine_on_min": seat_empty_engine_on_min,
            "seatbelt_off_moving_min": seatbelt_off_moving_min,
            "seatbelt_off_engine_on_min": seatbelt_off_engine_on_min,
            "continuous_operation_min": self.continuous_op_min(now),
            "within_scheduled_hours": within_scheduled_hours,
            "operator_matches_assigned": operator_matches_assigned,
            "sensor_dropout_min": sensor_dropout_min,

            # readiness.py / rule engine extras
            "min_person_distance_m": min_person_distance_m,
            "blind_spot_min": blind_spot_min,
            "red_zone_min": red_zone_min,
            "cab_temp_max_c": max(cab_temp_vals) if cab_temp_vals else 0.0,
            "weather": env.weather if env else None,
            "light": env.light if env else None,
            "visibility_m": env.visibility_m if env else None,
            "lightning_distance_km": env.lightning_distance_km if env else None,
            "fuel_rising": self._fuel_rising(),
            "engine_on": self.latest_operation.engine.state != "off",
            "truck_wait_min": 0.0,  # TODO: needs haul-truck proximity/assignment context from shift_context.daily_tasks
            "tilt_warning_count": tilt_events,
        }

    def _degraded_visibility(self) -> bool:
        env = self.latest_environment.environment if self.latest_environment else None
        if not env:
            return False
        return env.weather in ("Rainy", "Fog", "Storm", "Dust") or env.light == "night"

    def _fuel_rising(self) -> bool:
        fuel_levels = [(ts, s.engine.fuel_level_pct) for ts, s in self.status_readings if s.engine.fuel_level_pct is not None]
        if len(fuel_levels) < 2 or self.latest_operation is None:
            return False
        return self.latest_operation.engine.state == "running" and fuel_levels[-1][1] > fuel_levels[0][1] + 1.0


class StateManager:
    def __init__(self):
        self._machines: dict[str, MachineState] = defaultdict(lambda: None)
        self._lock = asyncio.Lock()

    async def ingest(self, msg) -> MachineState:
        async with self._lock:
            state = self._machines.get(msg.machine_id)
            if state is None:
                state = MachineState(msg.machine_id)
                self._machines[msg.machine_id] = state

            if isinstance(msg, ShiftContextMessage):
                state.on_shift_context(msg)
            elif isinstance(msg, OperationMessage):
                state.on_operation(msg)
            elif isinstance(msg, MotionBatchMessage):
                state.on_motion_batch(msg)
            elif isinstance(msg, ProximityMessage):
                state.on_proximity(msg)
            elif isinstance(msg, EnvironmentMessage):
                state.on_environment(msg)
            elif isinstance(msg, StatusMessage):
                state.on_status(msg)
            elif isinstance(msg, EventMessage):
                state.on_event(msg)

            return state

    def get(self, machine_id: str) -> MachineState | None:
        return self._machines.get(machine_id)

    def all_machine_ids(self) -> list[str]:
        return list(self._machines.keys())
