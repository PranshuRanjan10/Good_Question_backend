"""SQLite schema (SQLAlchemy) backing the REST endpoints in
frontend_handoff_spec_v1.md section 2. Seeded from datasets/incidents.csv,
datasets/training_history.csv at startup (see db/seed.py).
"""

from __future__ import annotations
import uuid
from datetime import datetime

from sqlalchemy import String, Float, Boolean, DateTime, JSON, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _uid() -> str:
    return uuid.uuid4().hex


class Synced:
    """Columns for the Supabase sync (app/sync). `uid` is the cloud primary key: local ids
    restart from 1 whenever Render wipes the SQLite file, uids never repeat. `synced_at` is
    NULL until the row is in Supabase; clearing it re-sends the row (upsert on uid)."""
    uid: Mapped[str | None] = mapped_column(String, default=_uid, index=True, nullable=True)
    synced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)


class InSession:
    """Which simulation run (shift_sessions.session_id) produced the row."""
    session_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)


class Incident(Synced, InSession, Base):
    __tablename__ = "incidents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(String, index=True)
    operator_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    incident_type: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    cause: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, default="sensor")  # auto | operator | supervisor
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    telemetry_window: Mapped[dict] = mapped_column(JSON, default=dict)  # +/- 30s buffer snapshot


class Alert(Synced, InSession, Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)  # AL-0001, as sent to the cab
    machine_id: Mapped[str] = mapped_column(String, index=True)
    operator_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    alert_type: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    message: Mapped[str] = mapped_column(String)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)


class ReadinessSnapshot(Synced, InSession, Base):
    __tablename__ = "readiness_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    readiness_score: Mapped[int] = mapped_column(Integer)
    readiness_breakdown: Mapped[dict] = mapped_column(JSON)


class TaskRecord(Synced, InSession, Base):
    __tablename__ = "task_records"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String, index=True)
    machine_id: Mapped[str] = mapped_column(String, index=True)
    operator_id: Mapped[str | None] = mapped_column(String, nullable=True)
    task_type: Mapped[str | None] = mapped_column(String, nullable=True)
    planned_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_p50_min: Mapped[float | None] = mapped_column(Float, nullable=True)  # last live prediction
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class TrainingCompletion(Synced, Base):
    __tablename__ = "training_completions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    operator_id: Mapped[str] = mapped_column(String, index=True)
    module_id: Mapped[str] = mapped_column(String)
    completed_at: Mapped[datetime] = mapped_column(DateTime)
    quiz_score: Mapped[float | None] = mapped_column(Float, nullable=True)


class AnomalyEvent(Synced, InSession, Base):
    """One row each time an anomaly type starts (not every 10 s while it lasts)."""
    __tablename__ = "anomaly_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(String, index=True)
    operator_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    anomaly_type: Mapped[str] = mapped_column(String, index=True)
    method: Mapped[str] = mapped_column(String)
    score: Mapped[float] = mapped_column(Float)
    explanation: Mapped[list] = mapped_column(JSON, default=list)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)


class HourlySummary(Synced, InSession, Base):
    """The backend-built interval_summary (organizers' telemetry columns)."""
    __tablename__ = "hourly_summaries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(String, index=True)
    operator_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    engine_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    fuel_used_l: Mapped[float] = mapped_column(Float)
    load_cycles: Mapped[int] = mapped_column(Integer)
    idling_time_min: Mapped[float] = mapped_column(Float)
    engine_on_min: Mapped[float] = mapped_column(Float)
    seatbelt_status: Mapped[str] = mapped_column(String)
    seatbelt_compliance_pct: Mapped[float] = mapped_column(Float)
    safety_alert_triggered: Mapped[str] = mapped_column(String)
    safety_alerts: Mapped[int] = mapped_column(Integer)
    fuel_per_cycle_l: Mapped[float] = mapped_column(Float)


class InstructorBooking(Synced, Base):
    """A booked instructor session (TRN-INSTR-01 or any module an operator wants coaching on)."""
    __tablename__ = "instructor_bookings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    operator_id: Mapped[str] = mapped_column(String, index=True)
    module_id: Mapped[str] = mapped_column(String)
    preferred_slot: Mapped[datetime] = mapped_column(DateTime)
    confirmed_slot: Mapped[datetime] = mapped_column(DateTime, index=True)
    status: Mapped[str] = mapped_column(String, default="confirmed")
    created_at: Mapped[datetime] = mapped_column(DateTime)


class ShiftSession(Synced, Base):
    """One simulation run: opened by a shift_context, kept up to date while telemetry flows."""
    __tablename__ = "shift_sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    machine_id: Mapped[str] = mapped_column(String, index=True)
    operator_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    scenario_id: Mapped[str | None] = mapped_column(String, nullable=True)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    time_scale: Mapped[float | None] = mapped_column(Float, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime)          # sim time
    last_seen_at: Mapped[datetime] = mapped_column(DateTime)        # sim time
    wall_started_at: Mapped[datetime] = mapped_column(DateTime)     # real UTC time
    wall_last_seen_at: Mapped[datetime] = mapped_column(DateTime)
