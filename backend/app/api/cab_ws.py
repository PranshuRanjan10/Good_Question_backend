"""ws /ws/cab/{machine_id} - pushes `assessment` every 10s, plus immediately
on a new alert (per frontend_handoff_spec_v1.md section 4 & 10).
"""

from __future__ import annotations
import asyncio

from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.anomaly import baseline_for
from app.db.session import SessionLocal
from app.db.models import ReadinessSnapshot

PUSH_INTERVAL_SECONDS = 10

router = APIRouter()


def _persist_readiness(machine_id: str, assessment: dict) -> None:
    db = SessionLocal()
    try:
        db.add(ReadinessSnapshot(
            machine_id=machine_id,
            timestamp=datetime.fromisoformat(assessment["timestamp"]),
            readiness_score=assessment["readiness_score"],
            readiness_breakdown=assessment["readiness_breakdown"],
        ))
        db.commit()
    finally:
        db.close()


@router.websocket("/ws/cab/{machine_id}")
async def cab_ws(websocket: WebSocket, machine_id: str):
    await websocket.accept()
    state_manager = websocket.app.state.state_manager
    decision_layer = websocket.app.state.decision_layer

    try:
        while True:
            machine_state = state_manager.get(machine_id)
            if machine_state is not None and machine_state.latest_operation is not None:
                operator_id = machine_state.latest_operation.operator_id
                baseline = baseline_for(operator_id)
                assessment = decision_layer.build_assessment(machine_state, baseline)
                await websocket.send_json(assessment)
                _persist_readiness(machine_id, assessment)
            await asyncio.sleep(PUSH_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        pass
