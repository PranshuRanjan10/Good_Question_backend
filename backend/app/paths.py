"""Filesystem locations, overridable by environment variables for deployment.

The backend must run from backend/ alone (a container or an EC2 box won't have the 60 MB
data/ folder), so the four small reference files it needs live in backend/seed_data/.
data/generate_data.py refreshes them whenever the datasets are regenerated.
"""
from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
SEED_DIR = Path(os.getenv("IRONSENSE_SEED_DIR", BACKEND_DIR / "seed_data"))
ARTIFACT_DIR = Path(os.getenv("IRONSENSE_ARTIFACT_DIR", BACKEND_DIR / "artifacts"))
DB_PATH = Path(os.getenv("IRONSENSE_DB_PATH", BACKEND_DIR / "ironsense.db"))
LOG_DIR = Path(os.getenv("IRONSENSE_LOG_DIR", BACKEND_DIR / "logs" / "raw"))
