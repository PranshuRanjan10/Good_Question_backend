from __future__ import annotations
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_PATH = Path(__file__).resolve().parents[2] / "ironsense.db"
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    from app.db.models import Base
    Base.metadata.create_all(bind=engine)
