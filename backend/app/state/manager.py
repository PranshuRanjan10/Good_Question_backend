"""Per-machine state: shift context, rolling buffers for each message type,
event-driven counters, and the live feature row.

The feature row comes from app.features.WindowBuffer -- the one builder that reproduces the
training data exactly (anomaly_windows_5min.csv). It used to be assembled by hand here, and the
two had drifted: idle was read from ground speed (so a digging excavator looked idle), and the
fatigue clock reset every time the engine dropped to idle. One builder, used everywhere, keeps
live rows and training rows identical.

The typed per-message buffers below stay, because the fast-path safety checks need the latest
raw readings (last proximity scan, last few motion samples), not 5-minute aggregates.
"""

from __future__ import annotations
import asyncio
from collections import deque
from datetime import datetime, timedelta, UTC

from app.features import WindowBuffer
from app.machines import size_factor
from app.ingest.schemas import (
    ShiftContextMessage, OperationMessage, MotionBatchMessage,
    ProximityMessage, EnvironmentMessage, StatusMessage, EventMessage,
)

WINDOW = timedelta(minutes=5)
EVENT_KEEP = timedelta(minutes=60)
SCHEDULED_SHIFT = timedelta(hours=10, minutes=30)   # same window the training data used  # geofence entries / lightning must outlive a 5-min window
PROXIMITY_STALE_S = 3.0      # proximity arrives at 1 Hz; older than this means nothing is near
BLIND_SPOT_BEARING = (135.0, 225.0)  # roughly behind the machine, per spec bearing convention
MOVING_MODES = {"dig", "swing_loaded", "dump", "swing_empty", "travel", "grade"}


class HourAccumulator:
    """Builds the organizers'-format hourly interval summary from live ticks."""

    def __init__(self, hour: datetime):
        self.hour = hour
        self.operator_id: str | None = None
        self.engine_on_min = self.idle_min = self.belt_off_min = self.seated_on_min = 0.0
        self.alerts = 0
        self.first_status: StatusMessage | None = None
        self.last_status: StatusMessage | None = None

    def add_operation(self, msg: OperationMessage, dt_min: float) -> None:
        self.operator_id = msg.operator_id or self.operator_id
        if msg.engine.state == "off":
            return
        self.engine_on_min += dt_min
        if msg.engine.state == "idle" or msg.implement.work_mode in ("idle", "break"):
            self.idle_min += dt_min
        if msg.cab.seat_occupied:
            self.seated_on_min += dt_min
            if msg.cab.seatbelt_fastened is False:
                self.belt_off_min += dt_min

    def add_status(self, msg: StatusMessage) -> None:
        self.first_status = self.first_status or msg
        self.last_status = msg

    def summary(self, machine_id: str) -> dict | None:
        if self.engine_on_min <= 0:
            return None

        def delta(get):
            try:
                a, b = get(self.first_status), get(self.last_status)
                return None if a is None or b is None else max(0.0, b - a)
            except AttributeError:
                return None

        cycles = delta(lambda s: s.implement.cumulative_load_cycles) or 0
        fuel = delta(lambda s: s.engine.cumulative_fuel_used_l) or 0.0
        hours = self.last_status.engine.cumulative_hours if self.last_status else None
        return {
            "machine_id": machine_id, "operator_id": self.operator_id, "timestamp": self.hour,
            "engine_hours": hours, "fuel_used_l": round(fuel, 2), "load_cycles": int(cycles),
            "idling_time_min": round(self.idle_min), "engine_on_min": round(self.engine_on_min, 1),
            "seatbelt_status": "Unfastened" if self.belt_off_min >= 5 else "Fastened",
            "seatbelt_compliance_pct": round(100 * (1 - self.belt_off_min / max(self.seated_on_min, 1e-6)), 1),
            "safety_alert_triggered": "Yes" if self.alerts else "No",
            "safety_alerts": self.alerts,
            "fuel_per_cycle_l": round(fuel / max(cycles, 1), 3),
        }


class MachineState:
    def __init__(self, machine_id: str):
        self.machine_id = machine_id

        self.shift_context: ShiftContextMessage | None = None
        self.buffer = WindowBuffer(machine_id)

        self.operation_ticks: deque[tuple[datetime, OperationMessage]] = deque()
        self.motion_samples: deque[tuple[datetime, dict, bool]] = deque()  # (ts, sample, traveling)
        self.status_readings: deque[tuple[datetime, StatusMessage]] = deque()
        self.events: deque[tuple[datetime, EventMessage]] = deque()

        self.latest_proximity: ProximityMessage | None = None
        self.latest_proximity_ts: datetime | None = None
        self.latest_environment: EnvironmentMessage | None = None
        self.latest_status: StatusMessage | None = None
        self.latest_operation: OperationMessage | None = None

        self.task_started_at: dict[str, datetime] = {}
        self.completed_tasks: list[dict] = []          # drained by the hub into the DB
        self.manual_reports: list[EventMessage] = []   # manual_near_miss / manual_incident
        self.finished_hours: list[dict] = []           # hourly summaries, drained by the hub
        self._hour: HourAccumulator | None = None
        self._sim_now: datetime | None = None  # latest timestamp seen across any message type
        self._row_cache: tuple[tuple, dict] | None = None
        self.last_engine_start: datetime | None = None
        self.last_engine_stop: datetime | None = None

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
        hour = ts.replace(minute=0, second=0, microsecond=0)
        if self._hour is None:
            self._hour = HourAccumulator(hour)
        elif hour > self._hour.hour:
            done = self._hour.summary(self.machine_id)
            if done:
                self.finished_hours.append(done)
            self._hour = HourAccumulator(hour)

    def sim_now(self) -> datetime:
        return self._sim_now or datetime.now(UTC)

    def on_shift_context(self, msg: ShiftContextMessage) -> None:
        self.shift_context = msg
        self.buffer.size_factor = size_factor(self.machine_id, msg.machine.model)
        self._touch(msg.timestamp)

    def started_outside_schedule(self) -> bool:
        """Engine started after the scheduled shift end (or well before its start). Running on
        into overtime is normal; a fresh start out of hours is the after-hours-use pattern."""
        sc = self.shift_context
        if not sc or not sc.operator.shift_start or not self.last_engine_start:
            return False
        start = sc.operator.shift_start
        return self.last_engine_start > start + SCHEDULED_SHIFT or self.last_engine_start < start - timedelta(hours=1)

    def on_operation(self, msg: OperationMessage) -> None:
        prev = self.latest_operation
        self.latest_operation = msg
        now = msg.timestamp
        self._touch(now)
        self.operation_ticks.append((now, msg))
        self._trim(self.operation_ticks, now)
        dt_min = 10 / 60
        if prev is not None:
            dt_min = min(max((now - prev.timestamp).total_seconds(), 0.0), 60.0) / 60
        self._hour.add_operation(msg, dt_min)
        if msg.task_id and msg.task_id not in self.task_started_at:
            self.task_started_at[msg.task_id] = now   # fallback when no task_start event arrives

    def on_motion_batch(self, msg: MotionBatchMessage) -> None:
        now = msg.timestamp
        self._touch(now)
        traveling = bool(self.latest_operation and (self.latest_operation.motion.ground_speed_kmh or 0) > 0.5)
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
        self._hour.add_status(msg)

    def on_event(self, msg: EventMessage) -> None:
        now = msg.timestamp
        self._touch(now)
        self.events.append((now, msg))
        while self.events and (now - self.events[0][0]) > EVENT_KEEP:
            self.events.popleft()
        task_id = (msg.details or {}).get("task_id")
        if msg.event_type == "engine_start":
            self.last_engine_start = now
        elif msg.event_type == "engine_stop":
            self.last_engine_stop = now
        if msg.event_type == "task_start" and task_id:
            self.task_started_at[task_id] = now
        elif msg.event_type == "task_complete" and task_id:
            self.completed_tasks.append(self._task_record(task_id, now, msg.details or {}))
        elif msg.event_type in ("manual_near_miss", "manual_incident"):
            self.manual_reports.append(msg)

    def _task_record(self, task_id: str, done_at: datetime, details: dict) -> dict:
        task = self._task_def(task_id)
        started = self.task_started_at.get(task_id)
        return {
            "task_id": task_id, "machine_id": self.machine_id,
            "operator_id": self.latest_operation.operator_id if self.latest_operation else None,
            "task_type": task.task_type if task else None,
            "planned_min": task.planned_estimate_min if task else None,
            "actual_min": round((done_at - started).total_seconds() / 60, 1) if started else None,
            "started_at": started, "completed_at": done_at,
        }

    def _task_def(self, task_id: str | None):
        if not self.shift_context or not task_id:
            return None
        return next((t for t in self.shift_context.daily_tasks if t.task_id == task_id), None)

    def note_alert(self) -> None:
        if self._hour:
            self._hour.alerts += 1

    # ---- derived helpers for the fast path ----

    def engine_running(self) -> bool:
        """The sim stops sending operation messages once the engine is off, so the last one
        says "idle" forever. An engine_stop event after it means the engine is off."""
        op = self.latest_operation
        if op is None or op.engine.state == "off":
            return False
        return not (self.last_engine_stop and self.last_engine_stop >= op.timestamp)

    @property
    def on_break(self) -> bool:
        return self.buffer.on_break

    def fresh_proximity(self) -> ProximityMessage | None:
        """The last proximity scan, only if it is current. The sim stops sending when nothing
        is in range, so an old scan must not keep a worker 'behind the machine' forever."""
        if not self.latest_proximity or not self.latest_proximity_ts:
            return None
        if (self.sim_now() - self.latest_proximity_ts).total_seconds() > PROXIMITY_STALE_S:
            return None
        return self.latest_proximity

    def is_moving(self) -> bool:
        op = self.latest_operation
        if not self.engine_running():
            return False
        return (op.motion.ground_speed_kmh or 0) > 0.5 or op.implement.work_mode in MOVING_MODES

    def recent_motion(self, seconds: float = 15) -> list[dict]:
        cutoff = self.sim_now() - timedelta(seconds=seconds)
        return [rec for ts, rec, _ in self.motion_samples if ts >= cutoff]

    def task_elapsed_min(self, task_id: str | None, now: datetime | None = None) -> float | None:
        started = self.task_started_at.get(task_id) if task_id else None
        if started is None:
            return None
        return max(0.0, ((now or self.sim_now()) - started).total_seconds() / 60)

    # ---- feature row ----

    def to_feature_row(self, now: datetime | None = None) -> dict:
        """The training-schema row from the shared builder, plus the context fields the rules,
        readiness and decision layer read (ids, size_factor, idle streak, break flag...)."""
        now = now or self.sim_now()
        key = (now, len(self.buffer.messages), self.buffer.last_ts)
        if self._row_cache and self._row_cache[0] == key:
            return dict(self._row_cache[1])

        row = self.buffer.feature_row(now)
        ident = self.buffer.identity()
        sc = self.shift_context
        op = self.latest_operation
        tilt_events = sum(1 for ts, e in self.events if e.event_type == "tilt_warning" and now - ts <= WINDOW)
        row.update({
            "machine_id": self.machine_id,
            "operator_id": ident["operator_id"],
            "assigned_operator_id": ident["assigned_operator_id"],
            "task_id": ident["task_id"],
            "timestamp": now,
            "size_factor": size_factor(self.machine_id, sc.machine.model if sc else None),
            "idle_streak_min": round(self.buffer.idle_streak_min, 1),
            "on_break": self.buffer.on_break,
            "engine_on": self.engine_running(),
            "fuel_rising": self._fuel_rising(),
            "tilt_warning_count": tilt_events,
            "engine_started_outside_hours": self.started_outside_schedule(),
            "seat_empty_outside_break_min": round(sum(
                1 / 6 for ts, m in self.operation_ticks
                if now - ts <= WINDOW and m.engine.state != "off" and m.cab.seat_occupied is False
                and m.implement.work_mode != "break"), 2),
        })
        self._row_cache = (key, row)
        return dict(row)

    def _engine_on_at(self, ts: datetime) -> bool:
        state = None
        for t, op in self.operation_ticks:
            if t > ts:
                break
            state = op.engine.state
        return state is not None and state != "off"

    def _fuel_rising(self) -> bool:
        """Fuel went up between two status readings with the engine running at both. A rise
        from a normal engine-off refuel must not count once the engine restarts afterwards."""
        levels = [(ts, s.engine.fuel_level_pct) for ts, s in self.status_readings if s.engine.fuel_level_pct is not None]
        return any(l2 > l1 + 1.0 and self._engine_on_at(t1) and self._engine_on_at(t2)
                   for (t1, l1), (t2, l2) in zip(levels, levels[1:]))


class StateManager:
    def __init__(self):
        self._machines: dict[str, MachineState] = {}
        self._lock = asyncio.Lock()

    async def ingest(self, msg, raw: dict | None = None) -> MachineState:
        async with self._lock:
            state = self._machines.get(msg.machine_id)
            if state is None or isinstance(msg, ShiftContextMessage):
                # A shift_context starts a new session. Without a clean slate, a sim restarted
                # from the same start time sends timestamps "older" than what we've seen, so
                # every new reading looks stale and alerts stop firing.
                state = MachineState(msg.machine_id)
                self._machines[msg.machine_id] = state

            state.buffer.add(raw if raw is not None else msg.model_dump(mode="json"))
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
