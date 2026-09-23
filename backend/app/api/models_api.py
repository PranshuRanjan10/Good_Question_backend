"""REST endpoints owned by Backend 1 (Pranshu): model outputs that aren't part of the live
assessment. Kept out of api/rest.py (Backend 2's) so the two never edit the same file."""

from __future__ import annotations

from fastapi import APIRouter

from app.models.profile import operator_profile

router = APIRouter(prefix="/api")


@router.get("/operators/{operator_id}/profile")
def get_operator_profile(operator_id: str):
    return operator_profile(operator_id)
