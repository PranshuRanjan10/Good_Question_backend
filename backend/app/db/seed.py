"""Seeds incidents.csv and training_history.csv into the DB on first run.
Columns per datasets/README.md / the real CSVs in datasets/.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path

from app.paths import SEED_DIR

import pandas as pd

from app.db.models import Incident, TrainingCompletion
from app.db.session import SessionLocal

DATASETS_DIR = SEED_DIR
# Seeded rows are the synthetic history, reloaded on every start. Stamping them "already
# synced" keeps the Supabase sync to live data only (otherwise every restart re-uploads them).
SEED_MARK = datetime(1970, 1, 1)


def seed_incidents():
    path = DATASETS_DIR / "incidents.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, parse_dates=["timestamp"])
    session = SessionLocal()
    try:
        if session.query(Incident).count() > 0:
            return
        for _, row in df.iterrows():
            session.add(Incident(
                machine_id=str(row.get("machine_id", "")),
                operator_id=str(row["operator_id"]) if pd.notna(row.get("operator_id")) else None,
                incident_type=str(row.get("incident_type", "unknown")),
                severity=str(row.get("severity", "warning")),
                cause=str(row["cause"]) if pd.notna(row.get("cause")) else None,
                source=str(row.get("source", "auto")),
                timestamp=row["timestamp"],
                telemetry_window={},
                synced_at=SEED_MARK,
            ))
        session.commit()
    finally:
        session.close()


def seed_training_history():
    path = DATASETS_DIR / "training_history.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, parse_dates=["recommended_at", "completed_at"])
    session = SessionLocal()
    try:
        if session.query(TrainingCompletion).count() > 0:
            return
        for _, row in df.iterrows():
            if pd.isna(row.get("completed_at")):
                continue
            session.add(TrainingCompletion(
                operator_id=str(row["operator_id"]),
                module_id=str(row["module_id"]),
                completed_at=row["completed_at"],
                quiz_score=float(row["quiz_score_pct"]) if pd.notna(row.get("quiz_score_pct")) else None,
                synced_at=SEED_MARK,
            ))
        session.commit()
    finally:
        session.close()


def run_all_seeds():
    seed_incidents()
    seed_training_history()
