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
    machine_id: Mapped[str] = mapped_column(String, index=True)
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
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class TrainingCompletion(Base):
    __tablename__ = "training_completions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    operator_id: Mapped[str] = mapped_column(String, index=True)
    module_id: Mapped[str] = mapped_column(String)
    completed_at: Mapped[datetime] = mapped_column(DateTime)
    quiz_score: Mapped[float | None] = mapped_column(Float, nullable=True)
