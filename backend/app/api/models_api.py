"""REST endpoints owned by Backend 1 (Pranshu): model outputs that aren't part of the live
assessment. Kept out of api/rest.py (Backend 2's) so the two never edit the same file."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.profile import operator_profile

router = APIRouter(prefix="/api")


@router.get("/operators/{operator_id}/profile")
def get_operator_profile(operator_id: str):
    return operator_profile(operator_id)


@router.get("/sync/status")
def sync_status(request: Request):
    """Is the Supabase sync on, and how did the last run go (rows pushed per table, or the error)."""
    return request.app.state.sync.status()


@router.post("/sync/now")
async def sync_now(request: Request):
    """Push everything pending right away (e.g. just before the demo ends)."""
    sync = request.app.state.sync
    if not sync.enabled:
        return {"enabled": False, "detail": "sync is off"}
    return await sync.sync_once()
