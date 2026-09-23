"""Training recommender: maps anomalies/incidents to training_modules.json entries.

No ML here — a lookup over `triggers`, plus a repeat-offense escalation rule.
"""

from __future__ import annotations
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

MODULES_PATH = Path(__file__).resolve().parents[3] / "data" / "datasets" / "training_modules.json"
REPEAT_WINDOW_DAYS = 7
REPEAT_THRESHOLD = 3
INSTRUCTOR_MODULE_ID = "TRN-INSTR-01"


def _load_modules() -> list[dict]:
    if not MODULES_PATH.exists():
        return []
    return json.loads(MODULES_PATH.read_text())


def _trigger_map(modules: list[dict]) -> dict[str, dict]:
    mapping = {}
    for module in modules:
        for trigger in module.get("triggers", []):
            mapping[trigger] = module
    return mapping


def recommend_training(anomalies: list[dict], incidents: list[dict], history: list[dict]) -> list[dict]:
    """Shared interface (handoff doc section 4).

    Returns: [{"module_id", "title", "reason"}]
    """
    modules = _load_modules()
    trigger_map = _trigger_map(modules)

    recs: list[dict] = []
    seen_module_ids = set()

    for anomaly in anomalies:
        anomaly_type = anomaly.get("anomaly_type")
        module = trigger_map.get(anomaly_type)
        if module and module["module_id"] not in seen_module_ids:
            recs.append({"module_id": module["module_id"], "title": module["title"], "reason": anomaly_type})
            seen_module_ids.add(module["module_id"])

    for incident in incidents:
        incident_type = incident.get("incident_type") or incident.get("type")
        module = trigger_map.get(incident_type)
        if module and module["module_id"] not in seen_module_ids:
            recs.append({"module_id": module["module_id"], "title": module["title"], "reason": incident_type})
            seen_module_ids.add(module["module_id"])

    repeat_module = _check_repeat_unsafe_types(incidents, history)
    if repeat_module and repeat_module["module_id"] not in seen_module_ids:
        recs.append(repeat_module)

    return recs


def _check_repeat_unsafe_types(incidents: list[dict], history: list[dict]) -> dict | None:
    cutoff = datetime.utcnow() - timedelta(days=REPEAT_WINDOW_DAYS)
    recent_types = []
    for incident in incidents:
        ts = incident.get("timestamp") or incident.get("occurred_at")
        incident_type = incident.get("incident_type") or incident.get("type")
        if not ts or not incident_type:
            continue
        try:
            ts_dt = datetime.fromisoformat(str(ts))
        except ValueError:
            continue
        if ts_dt >= cutoff:
            recent_types.append(incident_type)

    counts = Counter(recent_types)
    for incident_type, count in counts.items():
        if count >= REPEAT_THRESHOLD:
            return {
                "module_id": INSTRUCTOR_MODULE_ID,
                "title": "Instructor booking",
                "reason": f"{incident_type} repeated {count}x in {REPEAT_WINDOW_DAYS} days",
            }
    return None
