"""REST endpoints, exactly the set in frontend_handoff_spec_v1.md section 2."""

from __future__ import annotations
import json
from datetime import datetime, UTC
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models import (Alert, AnomalyEvent, HourlySummary, Incident, ReadinessSnapshot,
                           TaskRecord, TrainingCompletion)
from app.db.writes import book_instructor, bookings_for, recent_history, save_completion
from app.paths import SEED_DIR
from app.models.anomaly import baseline_for
from app.decision.recommender import recommend_training, _load_modules

try:
    from app.models.profile import operator_profile
except ImportError:  # profile.py not present yet
    def operator_profile(operator_id: str) -> dict:
        return {"operator_id": operator_id, "skill_score": None, "level": "Intermediate"}

router = APIRouter(prefix="/api")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/tasks/today")
def tasks_today(operator_id: str, request: Request):
    state_manager = request.app.state.state_manager
    for machine_id in state_manager.all_machine_ids():
        ms = state_manager.get(machine_id)
        if ms and ms.shift_context and ms.shift_context.operator.operator_id == operator_id:
            return {"operator_id": operator_id, "tasks": [t.model_dump() for t in ms.shift_context.daily_tasks]}
    return {"operator_id": operator_id, "tasks": []}


@router.get("/incidents")
def list_incidents(operator_id: str | None = None, machine_id: str | None = None, db: Session = Depends(get_db)):
    q = db.query(Incident)
    if operator_id:
        q = q.filter(Incident.operator_id == operator_id)
    if machine_id:
        q = q.filter(Incident.machine_id == machine_id)
    return [
        {
            "id": i.id, "machine_id": i.machine_id, "operator_id": i.operator_id,
            "incident_type": i.incident_type, "severity": i.severity, "cause": i.cause,
            "source": i.source, "timestamp": i.timestamp.isoformat() if i.timestamp else None,
        }
        for i in q.order_by(Incident.timestamp.desc()).limit(200)
    ]


class ManualIncident(BaseModel):
    machine_id: str
    operator_id: str | None = None
    incident_type: str = "manual_near_miss"
    severity: str = "info"
    note: str | None = None


@router.post("/incidents")
def report_incident(payload: ManualIncident, db: Session = Depends(get_db)):
    incident = Incident(
        machine_id=payload.machine_id, operator_id=payload.operator_id,
        incident_type=payload.incident_type, severity=payload.severity,
        cause=payload.note, source="operator", timestamp=datetime.now(UTC), telemetry_window={},
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return {"id": incident.id, "status": "recorded"}


@router.get("/digest/{operator_id}")
def digest(operator_id: str, db: Session = Depends(get_db)):
    """End-of-shift debrief: tasks done vs predicted, safety events, anomalies, one thing that
    went well, one thing to improve, and a suggested lesson. Covers the operator's latest shift
    day (sim time), falling back to their history when nothing live has been recorded yet."""
    stamps = [db.query(func.max(col)).filter(op_col == operator_id).scalar()
              for col, op_col in ((HourlySummary.timestamp, HourlySummary.operator_id),
                                  (AnomalyEvent.timestamp, AnomalyEvent.operator_id),
                                  (Alert.timestamp, Alert.operator_id),
                                  (TaskRecord.completed_at, TaskRecord.operator_id))]
    stamps = [t for t in stamps if t]
    latest = (max(stamps),) if stamps else None
    anomalies_q = db.query(AnomalyEvent).filter(AnomalyEvent.operator_id == operator_id)
    alerts_q = db.query(Alert).filter(Alert.operator_id == operator_id)
    tasks_q = db.query(TaskRecord).filter(TaskRecord.operator_id == operator_id)
    hours_q = db.query(HourlySummary).filter(HourlySummary.operator_id == operator_id)
    if latest:
        day_start = latest[0].replace(hour=0, minute=0, second=0, microsecond=0)
        anomalies_q = anomalies_q.filter(AnomalyEvent.timestamp >= day_start)
        alerts_q = alerts_q.filter(Alert.timestamp >= day_start)
        tasks_q = tasks_q.filter(TaskRecord.completed_at >= day_start)
        hours_q = hours_q.filter(HourlySummary.timestamp >= day_start)
    tasks = tasks_q.all()
    anomalies = anomalies_q.all()
    alerts = alerts_q.all()
    hours = hours_q.all()

    counts: dict[str, int] = {}
    for a in anomalies:
        counts[a.anomaly_type] = counts.get(a.anomaly_type, 0) + 1
    top_issue = max(counts, key=counts.get) if counts else None
    on_time = [t for t in tasks if t.actual_min is not None and t.predicted_p50_min is not None
               and t.actual_min <= t.predicted_p50_min]
    compliance = [h.seatbelt_compliance_pct for h in hours]
    went_well = (f"{len(on_time)} of {len(tasks)} tasks finished within the predicted time" if tasks
                 else "No safety-critical alerts" if not any(a.severity == "critical" for a in alerts)
                 else "Shift completed")
    if compliance and min(compliance) >= 95:
        went_well = "Seatbelt on for the whole shift"
    events, history = recent_history(operator_id)
    recs = recommend_training([{"anomaly_type": top_issue}] if top_issue else [], [], history)

    incidents = db.query(Incident).filter(Incident.operator_id == operator_id).all()
    return {
        "operator_id": operator_id,
        "profile": operator_profile(operator_id),
        "baseline": baseline_for(operator_id),
        "tasks": [{"task_id": t.task_id, "task_type": t.task_type, "planned_min": t.planned_min,
                   "predicted_p50_min": t.predicted_p50_min, "actual_min": t.actual_min} for t in tasks],
        "alerts_by_severity": {sev: sum(1 for a in alerts if a.severity == sev)
                               for sev in ("warning", "high", "critical")},
        "anomaly_counts": counts,
        "idle_min": round(sum(h.idling_time_min for h in hours)),
        "fuel_used_l": round(sum(h.fuel_used_l for h in hours), 1),
        "seatbelt_compliance_pct": round(sum(compliance) / len(compliance), 1) if compliance else None,
        "went_well": went_well,
        "to_improve": top_issue.replace("_", " ") if top_issue else None,
        "suggested_training": recs[:1],
        "incident_count": len(incidents),
        "recent_incidents": [
            {"incident_type": i.incident_type, "severity": i.severity, "timestamp": i.timestamp.isoformat()}
            for i in sorted(incidents, key=lambda i: i.timestamp, reverse=True)[:5]
        ],
    }


@router.get("/training/recommendations/{operator_id}")
def training_recommendations(operator_id: str, db: Session = Depends(get_db)):
    events, history = recent_history(operator_id)   # incidents + live anomaly events
    recent_anoms = (db.query(AnomalyEvent).filter(AnomalyEvent.operator_id == operator_id)
                    .order_by(AnomalyEvent.timestamp.desc()).limit(20).all())
    anomalies = [{"anomaly_type": a.anomaly_type} for a in recent_anoms]
    recs = recommend_training(anomalies, events[:20], history)
    return {"operator_id": operator_id, "recommendations": recs, "modules": _load_modules()}


PASS_MARK_PCT = 60.0


@lru_cache(maxsize=1)
def _content() -> dict:
    """seed_data/training_content.json: video_url + quiz per module_id (kept apart from
    training_modules.json, which data/generate_data.py overwrites)."""
    path = SEED_DIR / "training_content.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _module(module_id: str) -> dict:
    module = next((m for m in _load_modules() if m["module_id"] == module_id), None)
    if module is None:
        raise HTTPException(status_code=404, detail=f"unknown module_id {module_id!r}")
    return module


@router.get("/training/modules/{module_id}")
def training_module(module_id: str):
    """The module plus its lesson content. Correct answers stay server-side (see /training/complete)."""
    module = _module(module_id)
    content = _content().get(module_id, {})
    return {**module, "video_url": content.get("video_url"),
            "quiz": [{"question": q["question"], "options": q["options"]} for q in content.get("quiz", [])]}


class TrainingComplete(BaseModel):
    operator_id: str
    module_id: str
    quiz_score: float | None = None
    answers: list[int] | None = None       # chosen option index per question, graded server-side


@router.post("/training/complete")
def training_complete(payload: TrainingComplete):
    """Mark a module done. With `answers` the server grades them against the module's quiz and
    stores that score; otherwise `quiz_score` (0-100) is stored as sent, as before."""
    _module(payload.module_id)
    response: dict = {"status": "recorded"}
    score = payload.quiz_score
    if payload.answers is not None:
        quiz = _content().get(payload.module_id, {}).get("quiz", [])
        if not quiz:
            raise HTTPException(status_code=422, detail="this module has no quiz to grade")
        if len(payload.answers) != len(quiz):
            raise HTTPException(status_code=422, detail=f"expected {len(quiz)} answers, got {len(payload.answers)}")
        results = [{"correct": a == q["correct"], "correct_index": q["correct"], "explanation": q["explanation"]}
                   for a, q in zip(payload.answers, quiz)]
        right = sum(r["correct"] for r in results)
        score = round(100.0 * right / len(quiz), 1)
        response.update(correct=right, total=len(quiz), passed=score >= PASS_MARK_PCT, results=results)
    save_completion(payload.operator_id, payload.module_id, score)
    response["score"] = score
    return response


class BookingRequest(BaseModel):
    operator_id: str
    module_id: str
    preferred_slot: datetime


def _booking_json(b) -> dict:
    return {"booking_id": f"BK-{b.id:04d}", "operator_id": b.operator_id, "module_id": b.module_id,
            "preferred_slot": b.preferred_slot.isoformat(), "confirmed_slot": b.confirmed_slot.isoformat(),
            "status": b.status}


@router.post("/training/book")
def training_book(payload: BookingRequest):
    """Instructor booking (TRN-INSTR-01 is the usual module). One session per hour slot: if the
    preferred hour is taken, the next free hour is confirmed instead."""
    _module(payload.module_id)
    return _booking_json(book_instructor(payload.operator_id, payload.module_id, payload.preferred_slot))


@router.get("/training/bookings/{operator_id}")
def training_bookings(operator_id: str):
    return {"operator_id": operator_id, "bookings": [_booking_json(b) for b in bookings_for(operator_id)]}


@router.get("/readiness/history")
def readiness_history(machine_id: str, db: Session = Depends(get_db)):
    rows = (
        db.query(ReadinessSnapshot)
        .filter(ReadinessSnapshot.machine_id == machine_id)
        .order_by(ReadinessSnapshot.timestamp.desc())
        .limit(500)
        .all()
    )
    return [
        {"timestamp": r.timestamp.isoformat(), "readiness_score": r.readiness_score,
         "readiness_breakdown": r.readiness_breakdown}
        for r in rows
    ]
