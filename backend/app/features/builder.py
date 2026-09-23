"""
IronSense - live feature builder.

Turns the rolling buffer of incoming telemetry messages into **one feature row with exactly
the 48 columns of data/datasets/anomaly_windows_5min.csv**. That row is the shared contract:
the anomaly detector and the Readiness Score both read it, and both models were trained on
rows built the same way. If the live row and the training row disagree, the models degrade
silently -- which is why this is one file, used by everyone, rather than each caller
assembling its own dict.

Usage in the service:

    from app.features import WindowBuffer

    buffers = {}                                   # machine_id -> WindowBuffer
    buf = buffers.setdefault(machine_id, WindowBuffer(machine_id))
    buf.add(msg)                                   # every incoming message

    row = buf.feature_row()                        # every 60 s
    anomalies = detect_anomalies(row, baseline)    # Niharika
    readiness = compute_readiness(row, anomalies)  # Pranshu

`feature_row()` is pure: it never mutates the buffer, so it is safe to call at any cadence.

Windowing: the last WINDOW_MINUTES of messages (default 5, matching the training data).
Messages older than KEEP_MINUTES are dropped on insert so memory stays flat.

**Labelling convention.** A live row is labelled by the window's END (you can only aggregate
the past). The training rows in anomaly_windows_5min.csv are labelled by the window's START.
So a live row at 08:25 corresponds to the training row at 08:20. Shift by one window before
comparing the two, or everything will look one step out of phase.

Validated against the training data via tools/replay_sim.py: after that alignment, idle_ratio,
continuous_operation_min and proximity match exactly, and engine/idle/work minutes agree to
within 0.2 min. Three columns still differ, all because of what the replay can reproduce
rather than a fault here:

  * rpm_max (up to ~250 off) - the replay only has per-minute mean rpm, so it cannot emit the
    instantaneous peak. The live simulation sends real rpm, so this closes by itself.
  * cycles_per_hour - load cycles come from differencing cumulative counters recorded per
    minute, so short windows quantise.
  * min_person_distance_m (<1 m) - the replay samples proximity every 10 s against per-minute
    stored values.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

WINDOW_MINUTES = 5
KEEP_MINUTES = 20          # buffer horizon; incidents need +/-30 s of context, rules need more

# The exact column order of anomaly_windows_5min.csv, minus the label and id columns.
FEATURE_COLUMNS = [
    "task_type", "minutes", "engine_on_min", "idle_min", "work_min", "travel_min", "fuel_used_l",
    "load_cycles", "rpm_mean", "rpm_max", "over_rev_min", "load_pct_mean", "swing_rate_p95_dps",
    "fast_swing_count", "harsh_brake_count", "ground_speed_max_kmh", "bucket_raised_travel_s",
    "pitch_max_deg", "roll_max_deg", "bucket_payload_max_kg", "overload_count", "coolant_max_c",
    "hydraulic_oil_max_c", "cab_temp_max_c", "fuel_level_pct", "fuel_drop_engine_off_pct",
    "seat_empty_engine_on_min", "seatbelt_off_engine_on_min", "seatbelt_off_moving_min",
    "min_person_distance_m", "red_zone_min", "blind_spot_min", "truck_wait_min",
    "continuous_operation_min", "within_scheduled_hours", "sensor_dropout_min", "weather", "light",
    "ground_condition", "ambient_temp_c", "visibility_m", "lightning_distance_km", "idle_ratio",
    "fuel_rate_lph", "fuel_per_cycle_l", "cycles_per_hour", "operator_matches_assigned",
    "hour_of_day",
]

# Training-data conventions that must be reproduced exactly.
NO_PERSON_DISTANCE_M = 99.0     # "nobody within sensor range", not a missing value
OVER_REV_RPM = 2100
FAST_SWING_DPS = 55
OVERLOAD_KG = 2000
BUCKET_RAISED_M = 2.5
MOVING_MODES = {"dig", "swing_loaded", "dump", "swing_empty", "travel", "grade"}
TRUCK_TASKS = {"Earth Excavation", "Material Loading"}
TRUCK_SEEN_S = 20           # a dump truck seen within this many seconds counts as present
IDLE_MODES = {"idle", "break"}


def _ts(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _num(value, default=None):
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if f != f else f       # NaN -> default


def _dig(msg: dict, *path, default=None):
    """Nested lookup that tolerates the nulls the contract mandates."""
    cur = msg
    for key in path:
        if not isinstance(cur, dict) or cur.get(key) is None:
            return default
        cur = cur[key]
    return cur


class WindowBuffer:
    """Rolling per-machine buffer of raw messages, plus the shift context they belong to."""

    def __init__(self, machine_id: str, window_minutes: int = WINDOW_MINUTES,
                 keep_minutes: int = KEEP_MINUTES):
        self.machine_id = machine_id
        self.window = timedelta(minutes=window_minutes)
        self.keep = timedelta(minutes=keep_minutes)
        self.messages: deque[tuple[datetime, dict]] = deque()
        self.context: dict = {}          # last shift_context
        # Environment arrives every 15 min, so a 5-minute window usually contains none.
        # The models were trained with weather always present, so the last known value is
        # carried forward rather than letting the row go null.
        self.environment: dict = {}
        self.last_ts: datetime | None = None
        # Counters the simulation does not send; the backend derives them (spec section 5.6).
        self.continuous_operation_min = 0.0
        self.minutes_since_break = 0.0
        self._seat_since: datetime | None = None
        self._last_break_end: datetime | None = None
        # Current unbroken idle stretch (engine idling / idle mode). A 5-minute window can't
        # see a 40-minute idle on its own, so the rules read this instead.
        self.idle_streak_min = 0.0
        self._idle_since: datetime | None = None
        self.on_break = False
        self._break_by_event = False
        self.size_factor = 1.0         # set by the state manager from the machine registry

    # ------------------------------------------------------------------ ingest
    def add(self, msg: dict) -> None:
        """Append one validated message. Safe to call for every message type."""
        if not isinstance(msg, dict):
            return
        mtype = msg.get("msg_type")
        ts = _ts(msg.get("timestamp")) or self.last_ts or datetime.now(timezone.utc)
        self.last_ts = max(ts, self.last_ts) if self.last_ts else ts

        if mtype == "environment":
            env = _dig(msg, "environment", default=None)
            if env:
                self.environment = env
        if mtype == "shift_context":
            self.context = msg
            self._seat_since = None
            self.continuous_operation_min = 0.0
            self.minutes_since_break = 0.0
        elif mtype == "event":
            self._apply_event(msg, ts)

        self.messages.append((ts, msg))
        cutoff = self.last_ts - self.keep
        while self.messages and self.messages[0][0] < cutoff:
            self.messages.popleft()

        if mtype == "operation":
            self._track_seat(msg, ts)
            self._track_idle(msg, ts)

    def _apply_event(self, msg: dict, ts: datetime) -> None:
        etype = msg.get("event_type")
        if etype == "break_start":
            self.on_break = True
            self._break_by_event = True
        if etype in ("break_start", "engine_stop"):
            self._seat_since = None
            self.continuous_operation_min = 0.0
        if etype == "break_end":
            self.on_break = False
            self._break_by_event = False
            self._last_break_end = ts
            self.continuous_operation_min = 0.0
            self.minutes_since_break = 0.0
        if etype == "engine_stop":
            self._idle_since = None
            self.idle_streak_min = 0.0

    def _track_seat(self, msg: dict, ts: datetime) -> None:
        """Continuous operation: time seated with the engine on. Like the training data, it is
        reset by a declared break or by 10+ minutes out of the seat -- not by a short step-out."""
        seated = _dig(msg, "cab", "seat_occupied", default=False)
        engine_on = _dig(msg, "engine", "state", default="off") != "off"
        if seated and engine_on:
            self._out_since = None
            if self._seat_since is None:
                self._seat_since = ts
            self.continuous_operation_min = (ts - self._seat_since).total_seconds() / 60
        else:
            out_since = getattr(self, "_out_since", None) or ts
            self._out_since = out_since
            if (ts - out_since) >= timedelta(minutes=10):
                self._seat_since = None
                self.continuous_operation_min = 0.0
        base = self._last_break_end or _ts(_dig(self.context, "operator", "shift_start"))
        if base:
            self.minutes_since_break = max(0.0, (ts - base).total_seconds() / 60)

    def _track_idle(self, msg: dict, ts: datetime) -> None:
        state = _dig(msg, "engine", "state", default="off")
        mode = _dig(msg, "implement", "work_mode", default="idle")
        # A break is declared by break_start/break_end events, or by work_mode "break" -- accept
        # either, so a sim that forgets the events doesn't turn every break into an alert.
        if mode == "break":
            self.on_break = True
        elif self.on_break and not self._break_by_event:
            self.on_break = False
        if state != "off" and (state == "idle" or mode in IDLE_MODES):
            if self._idle_since is None:
                self._idle_since = ts
            self.idle_streak_min = (ts - self._idle_since).total_seconds() / 60
        else:
            self._idle_since = None
            self.idle_streak_min = 0.0

    # ------------------------------------------------------------------ query
    def in_window(self, now: datetime | None = None) -> list[dict]:
        end = now or self.last_ts
        if end is None:
            return []
        start = end - self.window
        return [m for ts, m in self.messages if start <= ts <= end]

    def feature_row(self, now: datetime | None = None) -> dict:
        """One row, exactly FEATURE_COLUMNS. Pure: does not mutate the buffer."""
        end = now or self.last_ts or datetime.now(timezone.utc)
        # With the engine off, status comes only every 5 min, so a 5-min window may hold a single
        # reading. The last status before the window anchors the engine-off fuel-drop check.
        anchor = next((m for ts, m in reversed(self.messages)
                       if ts < end - self.window and m.get("msg_type") == "status"), None)
        return build_feature_row(self.in_window(now), self.context, end,
                                 self.continuous_operation_min, self.environment, self.size_factor,
                                 anchor_status=anchor)

    def identity(self) -> dict:
        """The id columns the feature row deliberately omits, for logging and DB writes."""
        ops = [m for m in self.in_window() if m.get("operator_id")]
        return {
            "machine_id": self.machine_id,
            "timestamp": (self.last_ts or datetime.now(timezone.utc)).isoformat(),
            "operator_id": (ops[-1].get("operator_id") if ops
                            else _dig(self.context, "operator", "operator_id")),
            "assigned_operator_id": _dig(self.context, "operator", "operator_id"),
            "task_id": next((m.get("task_id") for m in reversed(self.in_window())
                             if m.get("task_id")), None),
        }


def build_feature_row(messages: list[dict], context: dict, now: datetime,
                      continuous_operation_min: float = 0.0,
                      latest_env: dict | None = None, size_factor: float = 1.0,
                      anchor_status: dict | None = None) -> dict:
    """
    Aggregate a window of raw messages into the training schema.

    Minute-equivalents: the training data counted whole minutes per state, so each 10 s
    `operation` message contributes 1/6 of a minute. Keeping that convention is what makes a
    live row comparable to a trained one.
    """
    ops = [m for m in messages if m.get("msg_type") == "operation"]
    task_types = {t.get("task_id"): t.get("task_type") for t in (context or {}).get("daily_tasks") or []}
    batches = [m for m in messages if m.get("msg_type") == "motion_batch"]
    statuses = [m for m in messages if m.get("msg_type") == "status"]
    proximity = [m for m in messages if m.get("msg_type") == "proximity"]
    envs = [m for m in messages if m.get("msg_type") == "environment"]
    events = [m for m in messages if m.get("msg_type") == "event"]

    per_op_min = 1.0 / 6 if ops else 0.0          # a 10 s sample is 1/6 of a minute

    # ---- engine / mode time
    engine_on = idle_min = work_min = travel_min = 0.0
    rpms, loads, fuel_rates, speeds = [], [], [], []
    over_rev = dropouts = 0
    seat_empty_on = belt_off_on = belt_off_moving = 0.0
    fuel_level_last = None
    fuel_level_first_off = last_off_level = None
    fuel_drop_off = 0.0
    task_type = None

    for m in ops:
        state = _dig(m, "engine", "state", default="off")
        mode = _dig(m, "implement", "work_mode", default="idle")
        on = state != "off"
        seated = bool(_dig(m, "cab", "seat_occupied", default=False))
        belted = bool(_dig(m, "cab", "seatbelt_fastened", default=True))
        speed = _num(_dig(m, "motion", "ground_speed_kmh"), 0.0)
        rpm = _num(_dig(m, "engine", "rpm"))
        moving = mode in MOVING_MODES or speed > 0.5

        if on:
            engine_on += per_op_min
            if state == "idle" or mode in IDLE_MODES:
                idle_min += per_op_min
            elif mode == "travel":
                travel_min += per_op_min
            else:
                work_min += per_op_min
            if not seated:
                seat_empty_on += per_op_min
            elif not belted:
                belt_off_on += per_op_min
                if moving:
                    belt_off_moving += per_op_min
            if rpm is None:
                dropouts += per_op_min          # missing sensor value on a running machine
            elif rpm > OVER_REV_RPM:
                over_rev += per_op_min

        if rpm is not None:
            rpms.append(rpm)
        load = _num(_dig(m, "engine", "load_pct"))
        if load is not None:
            loads.append(load)
        rate = _num(_dig(m, "engine", "fuel_rate_lph"))
        if rate is not None:
            fuel_rates.append(rate)
        speeds.append(speed)
        task_type = m.get("task_type") or task_types.get(m.get("task_id")) or task_type

    # ---- fuel theft: fuel falling between status readings while the engine is off.
    # With the engine off the sim stops sending operation messages and slows status to 5 min,
    # so engine state is tracked from ops + engine events in time order. Only falls count:
    # a rise is refuelling, not negative theft.
    engine_off = not ops or _dig(ops[0], "engine", "state", default="off") == "off"
    if anchor_status is not None and engine_off:
        last_off_level = _num(_dig(anchor_status, "engine", "fuel_level_pct"))
    for m in sorted(messages, key=lambda x: _ts(x.get("timestamp")) or now):
        mt = m.get("msg_type")
        if mt == "operation":
            engine_off = _dig(m, "engine", "state", default="off") == "off"
        elif mt == "event" and m.get("event_type") in ("engine_start", "engine_stop"):
            engine_off = m.get("event_type") == "engine_stop"
        elif mt == "status":
            lvl = _num(_dig(m, "engine", "fuel_level_pct"))
            if lvl is None:
                continue
            if engine_off and last_off_level is not None and lvl < last_off_level:
                fuel_drop_off += last_off_level - lvl
            last_off_level = lvl if engine_off else None

    # ---- status: cumulative counters and temperatures
    coolant = [_num(_dig(m, "engine", "coolant_temp_c")) for m in statuses]
    hyd = [_num(_dig(m, "implement", "hydraulic_oil_temp_c")) for m in statuses]
    cab = [_num(_dig(m, "cab", "cab_temp_c")) for m in statuses]
    levels = [_num(_dig(m, "engine", "fuel_level_pct")) for m in statuses]
    levels = [v for v in levels if v is not None]
    if levels:
        fuel_level_last = levels[-1]
    cycles_series = [_num(_dig(m, "implement", "cumulative_load_cycles")) for m in statuses]
    cycles_series = [v for v in cycles_series if v is not None]
    load_cycles = (cycles_series[-1] - cycles_series[0]) if len(cycles_series) > 1 else 0.0
    fuel_series = [_num(_dig(m, "engine", "cumulative_fuel_used_l")) for m in statuses]
    fuel_series = [v for v in fuel_series if v is not None]
    fuel_used = (fuel_series[-1] - fuel_series[0]) if len(fuel_series) > 1 else 0.0
    if fuel_used <= 0 and fuel_rates:
        fuel_used = sum(fuel_rates) / len(fuel_rates) * (engine_on / 60)   # fall back to rate
    for m in statuses:
        if _dig(m, "diagnostics", "active_fault_codes") is None:
            dropouts += 1
    for m in statuses:
        task_type = task_type or task_types.get(_dig(m, "task", "task_id"))

    # ---- motion batches: attitude, swing, payload
    pitch = roll = swing_p95 = payload_max = 0.0
    fast_swing = overload = 0
    bucket_raised_s = 0.0
    swings: list[float] = []
    for m in batches:
        fields = m.get("fields") or []
        idx = {name: i for i, name in enumerate(fields)}
        interval = _num(m.get("sample_interval_s"), 1.0) or 1.0
        for sample in m.get("samples") or []:
            def val(name, default=0.0):
                i = idx.get(name)
                return _num(sample[i], default) if i is not None and i < len(sample) else default
            pitch = max(pitch, abs(val("pitch_deg")))
            roll = max(roll, abs(val("roll_deg")))
            sw = abs(val("swing_rate_dps"))
            swings.append(sw)
            if sw > FAST_SWING_DPS:
                fast_swing += 1
            pl = val("bucket_payload_kg")
            payload_max = max(payload_max, pl)
            if pl > OVERLOAD_KG * size_factor:     # training: payload > 2000 kg x size_factor
                overload += 1
            if val("bucket_height_m") > BUCKET_RAISED_M and max(speeds or [0]) > 0.5:
                bucket_raised_s += interval
    if swings:
        swings.sort()
        swing_p95 = swings[min(len(swings) - 1, int(0.95 * len(swings)))]

    # ---- proximity
    min_dist = NO_PERSON_DISTANCE_M
    red_zone = blind_spot = 0.0
    env = _dig(envs[-1], "environment", default=None) if envs else None
    env = env or dict(latest_env or {})
    weather, light = env.get("weather"), env.get("light")
    danger_r = 8.0 if (weather in ("Rainy", "Fog", "Storm", "Dust") or light == "night") else 5.0
    for m in proximity:
        for obj in m.get("objects") or []:
            if obj.get("object_type") != "person":
                continue
            d = _num(obj.get("distance_m"), NO_PERSON_DISTANCE_M)
            min_dist = min(min_dist, d)
            if d < danger_r:
                red_zone += 1 / 60          # proximity arrives at 1 Hz -> minutes
            bearing = _num(obj.get("bearing_deg"), 0.0)
            if 135 <= bearing <= 225:       # behind the machine
                blind_spot += 1 / 60

    # ---- truck wait: idle minutes during a truck-served task with no dump truck in range.
    # Proximity is only sent while something is within range, so no message = no truck.
    truck_seen = sorted(_ts(m.get("timestamp")) for m in proximity
                        if any(o.get("role") == "dump_truck" for o in m.get("objects") or []))
    truck_wait = 0.0
    if task_type in TRUCK_TASKS:
        for m in ops:
            state = _dig(m, "engine", "state", default="off")
            mode = _dig(m, "implement", "work_mode", default="idle")
            if state == "off" or not (state == "idle" or mode in IDLE_MODES):
                continue
            t = _ts(m.get("timestamp"))
            if t and not any(abs((t - s).total_seconds()) <= TRUCK_SEEN_S for s in truck_seen if s):
                truck_wait += per_op_min

    # ---- events inside the window
    harsh = sum(1 for m in events if m.get("event_type") == "harsh_brake")

    # ---- shift schedule
    shift_start = _ts(_dig(context, "operator", "shift_start"))
    within_hours = 1
    if shift_start:
        end = shift_start + timedelta(hours=10, minutes=30)
        within_hours = int(shift_start <= now <= end)
    assigned = _dig(context, "operator", "operator_id")
    current = next((m.get("operator_id") for m in reversed(ops) if m.get("operator_id")), assigned)

    minutes = max(1.0, round(engine_on + (WINDOW_MINUTES - engine_on if not ops else 0), 0)) if not ops \
        else float(WINDOW_MINUTES)
    engine_on = min(engine_on, float(WINDOW_MINUTES))   # 10 s samples can overshoot by a fraction
    idle_min, work_min, travel_min = (min(v, engine_on) for v in (idle_min, work_min, travel_min))
    engine_on_safe = max(engine_on, 1e-6)

    row = {
        "task_type": task_type,
        "minutes": minutes,
        "engine_on_min": round(engine_on, 2),
        "idle_min": round(idle_min, 2),
        "work_min": round(work_min, 2),
        "travel_min": round(travel_min, 2),
        "fuel_used_l": round(max(fuel_used, 0.0), 3),
        "load_cycles": round(max(load_cycles, 0.0), 1),
        # Training averaged over every minute in the window, counting engine-off minutes as 0.
        # Averaging only the samples we received would read ~500 rpm high in a window where
        # the engine starts partway through, which is exactly when a false anomaly hurts.
        "rpm_mean": round(sum(rpms) * per_op_min / minutes, 1) if rpms else 0.0,
        "rpm_max": round(max(rpms), 1) if rpms else 0.0,
        "over_rev_min": round(over_rev, 2),
        "load_pct_mean": round(sum(loads) * per_op_min / minutes, 1) if loads else 0.0,
        "swing_rate_p95_dps": round(swing_p95, 1),
        "fast_swing_count": fast_swing,
        "harsh_brake_count": harsh,
        "ground_speed_max_kmh": round(max(speeds), 2) if speeds else 0.0,
        "bucket_raised_travel_s": round(bucket_raised_s, 1),
        "pitch_max_deg": round(pitch, 1),
        "roll_max_deg": round(roll, 1),
        "bucket_payload_max_kg": round(payload_max, 0),
        "overload_count": overload,
        "coolant_max_c": _max_or(coolant, 0.0),
        "hydraulic_oil_max_c": _max_or(hyd, 0.0),
        "cab_temp_max_c": _max_or(cab, 0.0),
        "fuel_level_pct": fuel_level_last if fuel_level_last is not None else 0.0,
        "fuel_drop_engine_off_pct": round(fuel_drop_off, 2),
        "seat_empty_engine_on_min": round(seat_empty_on, 2),
        "seatbelt_off_engine_on_min": round(belt_off_on, 2),
        "seatbelt_off_moving_min": round(belt_off_moving, 2),
        "min_person_distance_m": round(min_dist, 1),
        "red_zone_min": round(red_zone, 2),
        "blind_spot_min": round(blind_spot, 2),
        "truck_wait_min": round(truck_wait, 2),
        "continuous_operation_min": round(continuous_operation_min, 1),
        "within_scheduled_hours": within_hours,
        "sensor_dropout_min": round(dropouts, 2),
        "weather": env.get("weather"),
        "light": env.get("light"),
        "ground_condition": env.get("ground_condition"),
        "ambient_temp_c": _num(env.get("ambient_temp_c"), 0.0),
        "visibility_m": _num(env.get("visibility_m"), 9999.0),
        "lightning_distance_km": _num(env.get("lightning_distance_km")),
        "idle_ratio": round(idle_min / engine_on_safe, 3) if engine_on > 0 else 0.0,
        "fuel_rate_lph": round(sum(fuel_rates) / len(fuel_rates), 2) if fuel_rates else 0.0,
        # Training used max(cycles, 1) so an idle window shows its whole fuel burn.
        "fuel_per_cycle_l": round(max(fuel_used, 0.0) / max(load_cycles, 1), 3),
        "cycles_per_hour": round(load_cycles / (engine_on / 60), 1) if engine_on > 0 else 0.0,
        "operator_matches_assigned": int(current == assigned) if assigned else 1,
        "hour_of_day": now.hour,
    }
    return {k: row[k] for k in FEATURE_COLUMNS}      # fixed order, no extra keys


def _max_or(values, default=0.0):
    vals = [v for v in values if v is not None]
    return round(max(vals), 1) if vals else default
