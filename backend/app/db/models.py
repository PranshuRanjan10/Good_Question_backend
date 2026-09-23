"""SQLite schema (SQLAlchemy) backing the REST endpoints in
frontend_handoff_spec_v1.md section 2. Seeded from datasets/incidents.csv,
datasets/training_history.csv at startup (see db/seed.py).
"""

from __future__ import annotations
from datetime import datetime

from sqlalchemy import String, Float, Boolean, DateTime, JSON, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Incident(Base):
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


class Alert(Base):
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


class ReadinessSnapshot(Base):
    __tablename__ = "readiness_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    readiness_score: Mapped[int] = mapped_column(Integer)
    readiness_breakdown: Mapped[dict] = mapped_column(JSON)


class TaskRecord(Base):
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


class TrainingCompletion(Base):
    __tablename__ = "training_completions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    operator_id: Mapped[str] = mapped_column(String, index=True)
    module_id: Mapped[str] = mapped_column(String)
    completed_at: Mapped[datetime] = mapped_column(DateTime)
    quiz_score: Mapped[float | None] = mapped_column(Float, nullable=True)


class AnomalyEvent(Base):
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


class HourlySummary(Base):
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


class InstructorBooking(Base):
    """A booked instructor session (TRN-INSTR-01 or any module an operator wants coaching on)."""
    __tablename__ = "instructor_bookings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    operator_id: Mapped[str] = mapped_column(String, index=True)
    module_id: Mapped[str] = mapped_column(String)
    preferred_slot: Mapped[datetime] = mapped_column(DateTime)
    confirmed_slot: Mapped[datetime] = mapped_column(DateTime, index=True)
    status: Mapped[str] = mapped_column(String, default="confirmed")
    created_at: Mapped[datetime] = mapped_column(DateTime)
