"""FastAPI entrypoint. Run with:

    uvicorn app.main:app --reload --port 8000

from the backend/ directory.
"""

from __future__ import annotations
from fastapi import FastAPI

from app.db.session import init_db
from app.db.seed import run_all_seeds
from app.state.manager import StateManager
from app.decision.layer import DecisionLayer
from app.ingest.ws import router as telemetry_router
from app.api.cab_ws import router as cab_router
from app.api.rest import router as rest_router

app = FastAPI(title="IronSense Backend 2 - Anomaly + Live Service")


@app.on_event("startup")
def startup():
    init_db()
    run_all_seeds()
    app.state.state_manager = StateManager()

    predict_task = None
    try:
        from app.models.task_time import predict_task as _predict_task, load as _load_task_time
        _load_task_time()  # fail fast if artifacts/task_time_model.joblib is missing/corrupt
        predict_task = _predict_task
    except (ImportError, FileNotFoundError):
        pass  # task_time_model.joblib not trained/present yet - task_prediction stays null

    app.state.decision_layer = DecisionLayer(predict_task=predict_task)


app.include_router(telemetry_router)
app.include_router(cab_router)
app.include_router(rest_router)
