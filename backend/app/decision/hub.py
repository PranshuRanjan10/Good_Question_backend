"""CabHub: the single place that assesses a machine, stores the results, and pushes
`assessment` to every cab screen watching that machine.

Three triggers (spec section 4/10):
  * every PUSH_INTERVAL_S real seconds, for every machine (the routine refresh; this is also the
    only place Readiness history is written, so it is stored once, not once per open screen);
  * every 60 s of *sim* time as telemetry arrives (the window path for anomalies);
  * immediately, from the telemetry socket, when an incoming message raises a new or escalated
    safety alert. At time_scale 60 a worker can cross the blind spot in well under 10 real
    seconds, so waiting for the timer would miss it.

Side effects on a fresh assessment: new alerts -> alerts table; new critical alerts -> an
incident with the 30 s of telemetry before it (the 30 s after is attached once sim time has
moved on); new anomalies -> anomaly_events. Completed tasks, manual near-miss reports and hourly
summaries collected by the state manager are written as they arrive.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from fastapi import WebSocket

from app.db import writes
from app.i18n import DEFAULT_LANG, localize
from app.models.anomaly import baseline_for

log = logging.getLogger("ironsense.hub")

PUSH_INTERVAL_S = 10
SNAPSHOT_S = 30
SIM_ASSESS_EVERY = timedelta(seconds=60)
FAST_PATH_TYPES = {"operation", "proximity", "motion_batch", "event", "environment"}


class CabHub:
    def __init__(self, state_manager, decision_layer):
        self.state = state_manager
        self.decision = decision_layer
        self.clients: dict[str, dict[WebSocket, str]] = {}      # machine -> {socket: lang}
        self.latest: dict[str, dict] = {}
        self._pending_after: list[tuple[int, str, object]] = []   # (incident_id, machine_id, until)
        self._task: asyncio.Task | None = None
        self._last_sim_assess: dict[str, object] = {}

    # ------------------------------------------------------------------ connections
    async def connect(self, machine_id: str, ws: WebSocket, lang: str = DEFAULT_LANG) -> None:
        self.clients.setdefault(machine_id, {})[ws] = lang
        if machine_id in self.latest:                 # don't make a new screen wait 10 s
            await self._send(ws, localize(self.latest[machine_id], lang))

    def disconnect(self, machine_id: str, ws: WebSocket) -> None:
        self.clients.get(machine_id, {}).pop(ws, None)

    async def _send(self, ws: WebSocket, payload: dict) -> bool:
        try:
            await ws.send_json(payload)
            return True
        except Exception:                            # closed socket: drop it quietly
            return False

    async def broadcast(self, machine_id: str, payload: dict) -> None:
        """Send `payload` (English, as assessed) to every screen, rendered in its own language."""
        rendered: dict[str, dict] = {}
        dead = []
        for ws, lang in list(self.clients.get(machine_id, {}).items()):
            msg = rendered.get(lang) or rendered.setdefault(lang, localize(payload, lang))
            if not await self._send(ws, msg):
                dead.append(ws)
        for ws in dead:
            self.disconnect(machine_id, ws)

    # ------------------------------------------------------------------ triggers
    async def on_ingest(self, machine_state, msg_type: str) -> None:
        self._drain(machine_state)
        # Window path: every SIM_ASSESS_EVERY of *sim* time (spec: anomaly check every 60 s).
        # The real-time timer alone isn't enough -- at time_scale 60, 10 real seconds is
        # 10 sim minutes, longer than the 5-minute window the models look at.
        now = machine_state.sim_now()
        last = self._last_sim_assess.get(machine_state.machine_id)
        if last is None or now - last >= SIM_ASSESS_EVERY:
            await self.publish(machine_state, routine=True)
        elif msg_type in FAST_PATH_TYPES and self.decision.has_new_alerts(machine_state):
            await self.publish(machine_state, routine=False)

    async def publish(self, machine_state, routine: bool = True) -> dict | None:
        if machine_state.latest_operation is None:
            return None
        machine_id = machine_state.machine_id
        operator_id = machine_state.latest_operation.operator_id
        now = machine_state.sim_now()
        events, history = writes.recent_history(operator_id)
        cutoff = writes.naive_utc(now - timedelta(days=7))
        recent = [e for e in events if writes.naive_utc(_parse(e["timestamp"])) >= cutoff]

        result = self.decision.assess(machine_state, baseline_for(operator_id), recent, history)
        payload = result.payload
        self._last_sim_assess[machine_id] = now
        self._persist(machine_state, operator_id, result, routine)
        self.latest[machine_id] = payload
        await self.broadcast(machine_id, payload)
        return payload

    # ------------------------------------------------------------------ persistence
    def _persist(self, machine_state, operator_id, result, routine: bool) -> None:
        now = machine_state.sim_now()
        machine_id = machine_state.machine_id
        try:
            for alert in result.new_alerts:
                writes.save_alert(machine_id, operator_id, alert, now)
                machine_state.note_alert()
                if alert["severity"] == "critical":
                    before = _snapshot(machine_state, now - timedelta(seconds=SNAPSHOT_S), now)
                    inc_id = writes.save_auto_incident(machine_id, operator_id, alert, now, before)
                    self._pending_after.append((inc_id, machine_id, now + timedelta(seconds=SNAPSHOT_S)))
            for a in result.new_anomalies:
                writes.save_anomaly(machine_id, operator_id, a, now)
            if routine:
                writes.save_readiness(machine_id, result.payload, now)
        except Exception:
            log.exception("persisting assessment for %s failed", machine_id)

    def _drain(self, machine_state) -> None:
        """Write what the state manager collected: finished tasks, manual reports, hourly summaries."""
        try:
            while machine_state.completed_tasks:
                rec = machine_state.completed_tasks.pop(0)
                pred = self.decision.last_prediction.get(rec["task_id"]) or {}
                writes.save_task(rec, pred.get("p50_min"))
            while machine_state.manual_reports:
                ev = machine_state.manual_reports.pop(0)
                writes.save_manual_incident(machine_state.machine_id, ev.operator_id, ev.event_type,
                                            (ev.details or {}).get("note"), ev.timestamp)
            while machine_state.finished_hours:
                writes.save_hour(machine_state.finished_hours.pop(0))
        except Exception:
            log.exception("draining state for %s failed", machine_state.machine_id)

    def _attach_pending_snapshots(self) -> None:
        keep = []
        for inc_id, machine_id, until in self._pending_after:
            ms = self.state.get(machine_id)
            if ms is None:
                continue
            if ms.sim_now() >= until:
                writes.attach_incident_after(inc_id, _snapshot(ms, until - timedelta(seconds=SNAPSHOT_S), until))
            else:
                keep.append((inc_id, machine_id, until))
        self._pending_after = keep

    # ------------------------------------------------------------------ loop
    async def run(self) -> None:
        while True:
            try:
                for machine_id in self.state.all_machine_ids():
                    ms = self.state.get(machine_id)
                    if ms is None:
                        continue
                    self._drain(ms)
                    if self._last_sim_assess.get(machine_id) == ms.sim_now():
                        # Sim time hasn't moved (sim paused / disconnected): keep screens fresh
                        # with the last assessment instead of writing duplicate history rows.
                        if machine_id in self.latest:
                            await self.broadcast(machine_id, self.latest[machine_id])
                        continue
                    await self.publish(ms, routine=True)
                self._attach_pending_snapshots()
            except Exception:
                log.exception("hub tick failed")
            await asyncio.sleep(PUSH_INTERVAL_S)

    def start(self) -> None:
        self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()


def _parse(ts):
    from datetime import datetime
    return ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def _snapshot(machine_state, start, end) -> list[dict]:
    """Raw messages in [start, end] from the machine's buffer, for the incident record."""
    return [m for ts, m in machine_state.buffer.messages if start <= ts <= end]
