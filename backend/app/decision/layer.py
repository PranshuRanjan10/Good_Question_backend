"""Decision layer: builds the `assessment` message pushed to the cab, exactly matching
frontend_handoff_spec_v1.md section 10.

Calls Pranshu's compute_readiness/predict_task directly - real, trained artifacts, not stubs.

Alert identity (spec: "alerts with the same alert_id are updates of the same alert"):
every safety finding has a key (type + object). The first time a key appears it gets a new
alert_id; while the condition lasts, every assessment repeats it with the SAME id (severity may
escalate); when the condition clears the key is dropped, and a later recurrence is a new alert.
Anomalies stay in the message while they last too - nothing is hidden by a cooldown - but only
their first appearance counts as "new" for logging and training.

This object is shared by every cab connection and is only called from the hub, so there is one
source of truth per machine instead of each socket keeping its own cooldowns.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from app.models.anomaly import detect_anomalies
from app.models.readiness import compute_readiness
from app.decision.recommender import recommend_training
from app.rules.safety import zone_radii, proximity_view, run_safety_checks

SEVERITY_RANK = {"info": 0, "warning": 1, "high": 2, "critical": 3}
ANOMALY_NEW_AFTER = timedelta(minutes=10)   # gap after which a recurring anomaly counts as new


@dataclass
class Assessment:
    payload: dict
    new_alerts: list[dict] = field(default_factory=list)      # first seen, or escalated
    new_anomalies: list[dict] = field(default_factory=list)


class DecisionLayer:
    def __init__(self, predict_task: Callable | None = None):
        self._predict_task = predict_task
        self._active_alerts: dict[str, dict[str, dict]] = {}     # machine -> key -> {alert_id, severity}
        self._anomaly_seen: dict[str, dict[str, datetime]] = {}  # machine -> type -> last seen
        self._alert_seq = 0
        self.last_prediction: dict[str, dict] = {}              # task_id -> latest task_prediction

    def reset(self, machine_id: str) -> None:
        """New session for this machine (shift_context): forget its active alerts and anomalies."""
        self._active_alerts.pop(machine_id, None)
        self._anomaly_seen.pop(machine_id, None)

    def _next_alert_id(self) -> str:
        self._alert_seq += 1
        return f"AL-{self._alert_seq:04d}"

    # ------------------------------------------------------------------ alerts
    def has_new_alerts(self, machine_state, row: dict | None = None) -> bool:
        """Cheap check for the fast path: would this state raise a new or escalated alert?
        Does not change any state."""
        active = self._active_alerts.get(machine_state.machine_id, {})
        for a in run_safety_checks(machine_state, row):
            cur = active.get(a["_key"])
            if cur is None or SEVERITY_RANK[a["severity"]] > SEVERITY_RANK[cur["severity"]]:
                return True
        return False

    def _resolve_alerts(self, machine_id: str, raw: list[dict]) -> tuple[list[dict], list[dict]]:
        active = self._active_alerts.setdefault(machine_id, {})
        alerts, new = [], []
        seen = set()
        for a in raw:
            key = a.pop("_key")
            seen.add(key)
            cur = active.get(key)
            if cur is None:
                cur = active[key] = {"alert_id": self._next_alert_id(), "severity": a["severity"]}
                new.append({"alert_id": cur["alert_id"], **a})
            elif SEVERITY_RANK[a["severity"]] > SEVERITY_RANK[cur["severity"]]:
                cur["severity"] = a["severity"]
                new.append({"alert_id": cur["alert_id"], **a})
            alerts.append({"alert_id": cur["alert_id"], **a})
        for key in list(active):
            if key not in seen:
                del active[key]                     # condition cleared
        return alerts, new

    def _resolve_anomalies(self, machine_id: str, anomalies: list[dict], now: datetime) -> list[dict]:
        seen = self._anomaly_seen.setdefault(machine_id, {})
        new = []
        for a in anomalies:
            last = seen.get(a["anomaly_type"])
            if last is None or now - last > ANOMALY_NEW_AFTER:
                new.append(a)
            seen[a["anomaly_type"]] = now
        return new

    # ------------------------------------------------------------------ task prediction
    def _task_prediction(self, machine_state, now: datetime) -> dict | None:
        if not self._predict_task:
            return None
        sc = machine_state.shift_context
        op = machine_state.latest_operation
        if sc is None or op is None or op.task_id is None:
            return None
        task_def = next((t for t in sc.daily_tasks if t.task_id == op.task_id), None)
        if task_def is None:
            return None

        env_msg = machine_state.latest_environment.environment if machine_state.latest_environment else None
        task = {
            "task_id": task_def.task_id, "task_type": task_def.task_type,
            "planned_estimate_min": task_def.planned_estimate_min,
            "target_volume_m3": task_def.target_volume_m3,
            "scheduled_start": task_def.scheduled_start,
        }
        env = {
            "weather": env_msg.weather if env_msg else "Sunny",
            "ambient_temp_c": env_msg.ambient_temp_c if env_msg else None,
            "rain_mm_h": env_msg.rain_mm_h if env_msg else None,
            "wind_speed_kmh": env_msg.wind_speed_kmh if env_msg else None,
            "visibility_m": env_msg.visibility_m if env_msg else None,
            "light": env_msg.light if env_msg else "day",
            "ground_condition": env_msg.ground_condition if env_msg else None,
        }
        operator = {"operator_id": sc.operator.operator_id, "skill_level": sc.operator.skill_level}
        machine = {"machine_id": sc.machine.machine_id, "machine_age_yrs": sc.machine.machine_age_yrs}

        status = machine_state.latest_status
        progress_pct = None
        if status and status.task and status.task.task_id == task_def.task_id:
            progress_pct = status.task.progress_pct
        elapsed_min = machine_state.task_elapsed_min(task_def.task_id, now)
        pred = self._predict_task(task, env, operator, machine, elapsed_min=elapsed_min, progress_pct=progress_pct)
        if pred:
            self.last_prediction[task_def.task_id] = pred
        return pred

    # ------------------------------------------------------------------ main
    def assess(self, machine_state, baseline: dict, incidents: list[dict] | None = None,
               history: list[dict] | None = None) -> Assessment:
        now = machine_state.sim_now()  # sim time, not wall clock
        machine_id = machine_state.machine_id
        row = machine_state.to_feature_row(now)

        anomalies = detect_anomalies(row, baseline)
        new_anomalies = self._resolve_anomalies(machine_id, anomalies, now)
        readiness = compute_readiness(row, anomalies)
        alerts, new_alerts = self._resolve_alerts(machine_id, run_safety_checks(machine_state, row))
        recommendations = recommend_training(anomalies, incidents or [], history or [])

        payload = {
            "msg_type": "assessment",
            "timestamp": now.isoformat(),
            "machine_id": machine_id,
            "readiness_score": readiness["readiness_score"],
            "readiness_breakdown": readiness["readiness_breakdown"],
            "zones": zone_radii(row),
            "proximity_view": proximity_view(machine_state, row),
            "alerts": alerts,
            "anomalies": anomalies,
            "task_prediction": self._task_prediction(machine_state, now),
            "training_recommendations": recommendations,
        }
        return Assessment(payload, new_alerts, new_anomalies)

    def build_assessment(self, machine_state, baseline: dict, incidents: list[dict] | None = None,
                         history: list[dict] | None = None) -> dict:
        """Backwards-compatible: just the message."""
        return self.assess(machine_state, baseline, incidents, history).payload
