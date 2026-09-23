"""ws /ws/telemetry - validates each msg_type with Pydantic (spec section 5),
logs raw messages to logs/raw/YYYY-MM-DD.jsonl, and forwards valid ones into
the state manager.
"""

from __future__ import annotations
import json
from datetime import datetime, UTC
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.ingest.schemas import parse_message
from app.state.manager import StateManager

router = APIRouter()

LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "raw"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log_path() -> Path:
    return LOG_DIR / f"{datetime.now(UTC).date().isoformat()}.jsonl"


def _append_raw(payload: dict) -> None:
    with open(_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _error_reply(ref_msg_type: str | None, sim_tick, detail: str) -> dict:
    # Exact shape from frontend_handoff_spec_v1.md section 2.
    return {"msg_type": "error", "ref_msg_type": ref_msg_type, "sim_tick": sim_tick, "detail": detail}


@router.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket):
    await websocket.accept()
    state: StateManager = websocket.app.state.state_manager
    try:
        while True:
            raw = await websocket.receive_json()
            _append_raw(raw)

            ref_type = raw.get("msg_type")
            sim_tick = raw.get("sim_tick")
            try:
                msg = parse_message(raw)
            except KeyError:
                await websocket.send_json(_error_reply(ref_type, sim_tick, f"unknown msg_type: {ref_type!r}"))
                continue
            except ValidationError as e:
                first = e.errors()[0]
                field = ".".join(str(p) for p in first["loc"])
                await websocket.send_json(_error_reply(ref_type, sim_tick, f"{field}: {first['msg']}"))
                continue

            await state.ingest(msg)
    except WebSocketDisconnect:
        pass
