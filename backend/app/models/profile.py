"""
IronSense - operator skill profile (inference).

    from app.models.profile import operator_profile
    operator_profile("OP1001")

Returns the living replacement for the static Beginner / Intermediate / Expert label:

    {"operator_id", "skill_score", "level", "trend", "strengths", "focus_areas",
     "components": {"efficiency", "compliance", "safety", "smoothness"}, "as_of"}

This closes the loop in the pitch: telemetry measures how someone actually operates, the
profile feeds personalised training, and the improved profile feeds back into the task time
estimator. Nothing is trained here -- the profile is rolling statistics over the operator's
own history, which is why it updates the moment behaviour changes.

Profiles are built offline by training/build_profiles.py into artifacts/operator_profiles.json.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[2] / "artifacts" / "operator_profiles.json"

# The skill score is a weighted average of percentile ranks, so it centres near 50 by
# construction: an absolute 0-100 scale (80 = Expert) would label almost the whole fleet
# Beginner. These cuts are set against the real score distribution instead.
LEVELS = ((62, "Expert"), (45, "Intermediate"), (0, "Beginner"))
COMPONENTS = ("efficiency", "compliance", "safety", "smoothness")
COMPONENT_WEIGHTS = {"efficiency": 0.30, "compliance": 0.25, "safety": 0.30, "smoothness": 0.15}

_PROFILES = None
_LOCK = threading.Lock()

_LABELS = {
    "efficiency": ("Finishes tasks ahead of plan", "Tasks running over the planned time"),
    "compliance": ("Consistent seatbelt use", "Seatbelt and cab compliance"),
    "safety": ("Few alerts or incidents", "Safety alerts and near misses"),
    "smoothness": ("Smooth, controlled operation", "Harsh braking and fast swings"),
}


def load(path: Path | str = ARTIFACT) -> dict:
    global _PROFILES
    with _LOCK:
        if _PROFILES is None:
            try:
                _PROFILES = json.loads(Path(path).read_text())
            except (OSError, ValueError):
                _PROFILES = {"operators": {}}
    return _PROFILES


def level_for(score: float) -> str:
    for cut, name in LEVELS:
        if score >= cut:
            return name
    return "Beginner"


def operator_profile(operator_id: str, profiles: dict | None = None) -> dict:
    """
    The operator's current profile. Falls back to a neutral Intermediate profile for an
    operator with no history yet (a new hire on day one), so callers never have to branch.
    """
    p = (profiles or load()).get("operators", {}).get(operator_id)
    if not p:
        return {"operator_id": operator_id, "skill_score": 65.0, "level": "Intermediate",
                "trend": "new", "strengths": [], "focus_areas": [],
                "components": {c: 65.0 for c in COMPONENTS}, "as_of": None, "weeks_of_history": 0}

    comp = p["components"]
    ranked = sorted(COMPONENTS, key=lambda c: comp.get(c, 0.0))
    return {
        "operator_id": operator_id,
        "skill_score": p["skill_score"],
        "level": p["level"],
        "trend": p["trend"],                       # improving | steady | declining | new
        "strengths": [_LABELS[c][0] for c in reversed(ranked) if comp.get(c, 0) >= 75][:2],
        "focus_areas": [_LABELS[c][1] for c in ranked if comp.get(c, 0) < 70][:2],
        "components": comp,
        "as_of": p.get("as_of"),
        "weeks_of_history": p.get("weeks_of_history", 0),
    }


def skill_label(operator_id: str, static_label: str | None = None) -> str:
    """
    What to hand the task time estimator. Uses the live profile once there is enough history,
    otherwise the static label from the roster.
    """
    p = operator_profile(operator_id)
    return p["level"] if p["weeks_of_history"] >= 2 else (static_label or p["level"])


def history(operator_id: str, profiles: dict | None = None) -> list[dict]:
    """Weekly skill_score points, for the trend chart on the operator's own screen."""
    return (profiles or load()).get("weekly", {}).get(operator_id, [])
