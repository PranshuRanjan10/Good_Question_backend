"""ws /ws/cab/{machine_id} - registers the screen with the CabHub, which pushes `assessment`
every 10 s and immediately on a new alert (frontend_handoff_spec_v1.md sections 4 and 10).

Language: `ws://<host>/ws/cab/EXC001?lang=hi` (en | hi; anything else falls back to en). Alerts
and anomaly explanations are translated per connection, and every alert gains `voice_text` (one
short sentence for text-to-speech); the assessment gains `lang`. Nothing else changes.

The screen may send `{"msg_type": "ack", "alert_id": "AL-0107"}` when the operator
acknowledges an alert; it is recorded on the alert row.
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.db.writes import acknowledge_alert
from app.i18n import resolve_lang

router = APIRouter()


@router.websocket("/ws/cab/{machine_id}")
async def cab_ws(websocket: WebSocket, machine_id: str, lang: str = "en"):
    await websocket.accept()
    hub = websocket.app.state.hub
    await hub.connect(machine_id, websocket, resolve_lang(lang))
    try:
        while True:
            try:
                msg = await websocket.receive_json()
            except ValueError:
                continue
            if isinstance(msg, dict) and msg.get("msg_type") == "ack" and msg.get("alert_id"):
                acknowledge_alert(str(msg["alert_id"]))
    except WebSocketDisconnect:
        pass
    finally:
        hub.disconnect(machine_id, websocket)
