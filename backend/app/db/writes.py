"""Small write/read helpers for the live service, so the hub and REST layer don't hand-roll
sessions. All timestamps are stored as naive UTC (SQLite has no time zones)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.db.models import (Alert, AnomalyEvent, HourlySummary, Incident, InstructorBooking, ReadinessSnapshot,
                           TaskRecord, TrainingCompletion)
from app.db.session import SessionLocal

# Safety alert -> incident type used by incidents.csv / training_modules.json triggers, so an
# auto-logged incident maps to the right training module.
ALERT_TO_INCIDENT = {
    "proximity_person_blind_spot": "near_miss_person",
    "proximity_person_danger": "near_miss_person",
    "tilt_warning": "tip_over_risk",
    "seatbelt_off_moving": "seatbelt_off_while_moving",
    "operator_out_of_cab": "operator_out_of_seat",
    "refuel_engine_on": "unsafe_refuelling",
}


def naive_utc(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    return ts.astimezone(timezone.utc).replace(tzinfo=None) if ts.tzinfo else ts


def _commit(obj):
    db = SessionLocal()
    try:
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj
    finally:
        db.close()


def save_alert(machine_id: str, operator_id: str | None, alert: dict, ts: datetime) -> None:
    _commit(Alert(alert_id=alert["alert_id"], machine_id=machine_id, operator_id=operator_id,
                  alert_type=alert["alert_type"], severity=alert["severity"], message=alert["message"],
                  timestamp=naive_utc(ts)))


def acknowledge_alert(alert_id: str) -> None:
    db = SessionLocal()
    try:
        db.query(Alert).filter(Alert.alert_id == alert_id).update({"acknowledged": True})
        db.commit()
    finally:
        db.close()


def save_auto_incident(machine_id: str, operator_id: str | None, alert: dict, ts: datetime,
                       before: list[dict]) -> int:
    inc = _commit(Incident(
        machine_id=machine_id, operator_id=operator_id,
        incident_type=ALERT_TO_INCIDENT.get(alert["alert_type"], alert["alert_type"]),
        severity=alert["severity"], cause=alert["message"], source="auto", timestamp=naive_utc(ts),
        telemetry_window={"alert_id": alert["alert_id"], "before": before, "after": []}))
    return inc.id


def attach_incident_after(incident_id: int, after: list[dict]) -> None:
    db = SessionLocal()
    try:
        inc = db.get(Incident, incident_id)
        if inc is not None:
            inc.telemetry_window = {**(inc.telemetry_window or {}), "after": after}
            db.commit()
    finally:
        db.close()


def save_manual_incident(machine_id: str, operator_id: str | None, event_type: str, note: str | None,
                         ts: datetime) -> None:
    _commit(Incident(machine_id=machine_id, operator_id=operator_id, incident_type=event_type,
                     severity="high" if event_type == "manual_incident" else "warning",
                     cause=note, source="operator", timestamp=naive_utc(ts), telemetry_window={}))


def save_anomaly(machine_id: str, operator_id: str | None, a: dict, ts: datetime) -> None:
    _commit(AnomalyEvent(machine_id=machine_id, operator_id=operator_id, anomaly_type=a["anomaly_type"],
                         method=a.get("method", ""), score=float(a.get("score", 0)),
                         explanation=a.get("explanation", []), timestamp=naive_utc(ts)))


def save_readiness(machine_id: str, assessment: dict, ts: datetime) -> None:
    _commit(ReadinessSnapshot(machine_id=machine_id, timestamp=naive_utc(ts),
                              readiness_score=assessment["readiness_score"],
                              readiness_breakdown=assessment["readiness_breakdown"]))


def save_task(record: dict, predicted_p50: float | None) -> None:
    _commit(TaskRecord(task_id=record["task_id"], machine_id=record["machine_id"],
                       operator_id=record.get("operator_id"), task_type=record.get("task_type"),
                       planned_min=record.get("planned_min"), actual_min=record.get("actual_min"),
                       predicted_p50_min=predicted_p50, started_at=naive_utc(record.get("started_at")),
                       completed_at=naive_utc(record.get("completed_at"))))


def save_hour(summary: dict) -> None:
    _commit(HourlySummary(**{**summary, "timestamp": naive_utc(summary["timestamp"])}))


def recent_history(operator_id: str | None, limit: int = 200) -> tuple[list[dict], list[dict]]:
    """(incidents + anomaly events as incident-like dicts, training completions) for the recommender."""
    if not operator_id:
        return [], []
    db = SessionLocal()
    try:
        incs = (db.query(Incident).filter(Incident.operator_id == operator_id)
                .order_by(Incident.timestamp.desc()).limit(limit).all())
        anoms = (db.query(AnomalyEvent).filter(AnomalyEvent.operator_id == operator_id)
                 .order_by(AnomalyEvent.timestamp.desc()).limit(limit).all())
        done = db.query(TrainingCompletion).filter(TrainingCompletion.operator_id == operator_id).all()
        events = ([{"incident_type": i.incident_type, "timestamp": i.timestamp.isoformat()} for i in incs]
                  + [{"incident_type": a.anomaly_type, "timestamp": a.timestamp.isoformat()} for a in anoms])
        history = [{"module_id": t.module_id, "completed_at": t.completed_at.isoformat()} for t in done]
        return events, history
    finally:
        db.close()


def save_completion(operator_id: str, module_id: str, score: float | None) -> None:
    _commit(TrainingCompletion(operator_id=operator_id, module_id=module_id,
                               completed_at=naive_utc(datetime.now(timezone.utc)), quiz_score=score))


SLOT = timedelta(hours=1)


def book_instructor(operator_id: str, module_id: str, preferred: datetime) -> InstructorBooking:
    """Confirm the preferred hour, or the next free hour if the instructor already has a
    session then (one instructor, one session per slot)."""
    preferred = naive_utc(preferred)
    db = SessionLocal()
    try:
        taken = {b.confirmed_slot for b in db.query(InstructorBooking).filter(
            InstructorBooking.status == "confirmed", InstructorBooking.confirmed_slot >= preferred).all()}
        slot = preferred
        while slot in taken:
            slot += SLOT
        booking = InstructorBooking(operator_id=operator_id, module_id=module_id, preferred_slot=preferred,
                                    confirmed_slot=slot, status="confirmed",
                                    created_at=naive_utc(datetime.now(timezone.utc)))
        db.add(booking)
        db.commit()
        db.refresh(booking)
        return booking
    finally:
        db.close()


def bookings_for(operator_id: str) -> list[InstructorBooking]:
    db = SessionLocal()
    try:
        return (db.query(InstructorBooking).filter(InstructorBooking.operator_id == operator_id)
                .order_by(InstructorBooking.confirmed_slot).all())
    finally:
        db.close()
