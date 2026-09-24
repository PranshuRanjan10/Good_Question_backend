"""Loads backend/.env into the environment for local runs. On Render the same variables are
set in the dashboard (Environment), and a real environment variable always wins over .env."""
from __future__ import annotations

import os

from app.paths import BACKEND_DIR


def load_env(path=None) -> None:
    path = path or BACKEND_DIR / ".env"
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
