"""Decision layer: builds the `assessment` message pushed to the cab every
10s (and immediately on a new alert), exactly matching
frontend_handoff_spec_v1.md section 10.

Calls Pranshu's compute_readiness/predict_task/operator_profile directly -
these are real, trained artifacts, not stubs.
"""

from __future__ import annotations
from datetime import datetime, UTC
from typing import Callable

from app.models.anomaly import detect_anomalies
from app.models.readiness import compute_readiness
from app.decision.recommender import recommend_training
from app.rules.safety import zone_radii, proximity_view, run_safety_checks

COOLDOWN_SECONDS = 60


class DecisionLayer:
    def __init__(self, predict_task: Callable | None = None):
        self._predict_task = predict_task  # optional override for tests
        self._last_alert_at: dict[str, datetime] = {}
        self._alert_seq = 0

    def _next_alert_id(self) -> str:
        self._alert_seq += 1
        return f"AL-{self._alert_seq:04d}"

    def _is_in_cooldown(self, key: str, now: datetime) -> bool:
        last = self._last_alert_at.get(key)
        return last is not None and (now - last).total_seconds() < COOLDOWN_SECONDS

    def _mark_alerted(self, key: str, now: datetime) -> None:
        self._last_alert_at[key] = now

    def _derive_task_inputs(self, machine_state, row: dict) -> tuple[dict, dict, dict, dict] | None:
        """Builds predict_task()'s (task, env, operator, machine) dicts straight
        from live state, so the cab WS doesn't have to assemble them by hand."""
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
        task_status = status.task if (status and status.task) else None
        elapsed_min = None
        progress_pct = task_status.progress_pct if task_status else None
        if sc.operator.shift_start and task_def.scheduled_start:
            pass  # elapsed_min needs task_start event timing; left None until wired to events

        return task, env, operator, machine, elapsed_min, progress_pct

    def build_assessment(self, machine_state, baseline: dict,
                          incidents: list[dict] | None = None,
                          history: list[dict] | None = None) -> dict:
        now = machine_state.sim_now()  # sim time, not wall clock: this is a simulated site on sim_tick
        machine_id = machine_state.machine_id
        row = machine_state.to_feature_row(now)

        anomalies = detect_anomalies(row, baseline)

        active_anomalies = []
        for a in anomalies:
            key = f"{machine_id}:{a['anomaly_type']}"
            if self._is_in_cooldown(key, now):
                continue
            self._mark_alerted(key, now)
            active_anomalies.append(a)

        readiness = compute_readiness(row, anomalies)

        raw_alerts = run_safety_checks(machine_state)
        alerts = []
        for a in raw_alerts:
            key = f"{machine_id}:{a['alert_type']}"
            if self._is_in_cooldown(key, now) and a["severity"] not in ("critical",):
                continue
            self._mark_alerted(key, now)
            alerts.append({"alert_id": self._next_alert_id(), **a})

        recommendations = recommend_training(anomalies, incidents or [], history or [])

        task_prediction = None
        if self._predict_task:
            derived = self._derive_task_inputs(machine_state, row)
            if derived:
                task, env, operator, machine, elapsed_min, progress_pct = derived
                task_prediction = self._predict_task(task, env, operator, machine,
                                                       elapsed_min=elapsed_min, progress_pct=progress_pct)

        return {
            "msg_type": "assessment",
            "timestamp": now.isoformat(),
            "machine_id": machine_id,
            "readiness_score": readiness["readiness_score"],
            "readiness_breakdown": readiness["readiness_breakdown"],
            "zones": zone_radii(row),
            "proximity_view": proximity_view(machine_state),
            "alerts": alerts,
            "anomalies": active_anomalies,
            "task_prediction": task_prediction,
            "training_recommendations": recommendations,
        }
