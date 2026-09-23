"""REST endpoints, exactly the set in frontend_handoff_spec_v1.md section 2."""

from __future__ import annotations
from datetime import datetime, UTC
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.db.models import Incident, ReadinessSnapshot, TrainingCompletion
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
    incidents = db.query(Incident).filter(Incident.operator_id == operator_id).all()
    profile = operator_profile(operator_id)
    baseline = baseline_for(operator_id)
    return {
        "operator_id": operator_id,
        "profile": profile,
        "baseline": baseline,
        "incident_count": len(incidents),
        "recent_incidents": [
            {"incident_type": i.incident_type, "severity": i.severity, "timestamp": i.timestamp.isoformat()}
            for i in sorted(incidents, key=lambda i: i.timestamp, reverse=True)[:5]
        ],
    }


@router.get("/training/recommendations/{operator_id}")
def training_recommendations(operator_id: str, db: Session = Depends(get_db)):
    incidents = [
        {"incident_type": i.incident_type, "timestamp": i.timestamp.isoformat()}
        for i in db.query(Incident).filter(Incident.operator_id == operator_id).all()
    ]
    completed = db.query(TrainingCompletion).filter(TrainingCompletion.operator_id == operator_id).all()
    history = [{"module_id": t.module_id, "completed_at": t.completed_at.isoformat()} for t in completed]
    recs = recommend_training([], incidents, history)
    return {"operator_id": operator_id, "recommendations": recs, "modules": _load_modules()}


class TrainingComplete(BaseModel):
    operator_id: str
    module_id: str
    quiz_score: float | None = None


@router.post("/training/complete")
def training_complete(payload: TrainingComplete, db: Session = Depends(get_db)):
    completion = TrainingCompletion(
        operator_id=payload.operator_id, module_id=payload.module_id,
        completed_at=datetime.now(UTC), quiz_score=payload.quiz_score,
    )
    db.add(completion)
    db.commit()
    return {"status": "recorded"}


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
