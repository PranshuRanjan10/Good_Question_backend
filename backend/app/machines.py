"""Machine registry: size_factor per machine, from data/datasets/machines.csv.

The anomaly model divides size-dependent features (rpm_max, bucket payload) by size_factor, and
the overload rule scales its limit by it, so it must be the same number training used.
Falls back to the model name (from shift_context), then to 1.0 (a CAT 320).
"""
from __future__ import annotations

import csv
from functools import lru_cache
from app.paths import SEED_DIR

MACHINES_CSV = SEED_DIR / "machines.csv"
SIZE_BY_MODEL = {"CAT 320": 1.0, "CAT 323": 1.1, "CAT 330": 1.4, "CAT 336": 1.55,
                 "CAT 950": 1.0, "CAT 432": 0.55}


@lru_cache(maxsize=1)
def _by_id() -> dict[str, float]:
    try:
        with open(MACHINES_CSV, newline="") as f:
            return {r["machine_id"]: float(r["size_factor"]) for r in csv.DictReader(f)}
    except (OSError, KeyError, ValueError):
        return {}


def size_factor(machine_id: str | None, model: str | None = None) -> float:
    if machine_id and machine_id in _by_id():
        return _by_id()[machine_id]
    return SIZE_BY_MODEL.get(model or "", 1.0)
