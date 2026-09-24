"""Two bookkeeping rules applied to every database write, in one place:

1. New rows from a machine are tagged with that machine's current simulation session
   (shift_sessions.session_id), so everything a run produced can be pulled out together.
2. A row edited after it reached Supabase (an incident getting its "after" snapshot, an alert
   being acknowledged, a session's last_seen moving on) is marked unsynced again, so the next
   sync pushes the new version.

Doing it with a flush hook means none of the individual write functions need to remember.
"""
from __future__ import annotations

from sqlalchemy import event, inspect
from sqlalchemy.orm import Session

from app.db.models import InSession, Synced

_current_session: dict[str, str] = {}      # machine_id -> session_id


def set_current_session(machine_id: str, session_id: str) -> None:
    _current_session[machine_id] = session_id


def current_session(machine_id: str | None) -> str | None:
    return _current_session.get(machine_id) if machine_id else None


@event.listens_for(Session, "before_flush")
def _tag_and_mark(session, flush_context, instances) -> None:
    for obj in session.new:
        if isinstance(obj, InSession) and obj.session_id is None:
            obj.session_id = current_session(getattr(obj, "machine_id", None))
    for obj in session.dirty:
        if not isinstance(obj, Synced) or not session.is_modified(obj):
            continue
        state = inspect(obj)
        changed = {a.key for a in state.attrs if a.history.has_changes()}
        if changed - {"synced_at"}:
            obj.synced_at = None
