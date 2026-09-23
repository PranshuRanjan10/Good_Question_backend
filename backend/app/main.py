"""FastAPI entrypoint. Run from the backend/ directory:

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.session import init_db
from app.db.seed import run_all_seeds
from app.state.manager import StateManager
from app.decision.layer import DecisionLayer
from app.decision.hub import CabHub
from app.ingest.ws import router as telemetry_router
from app.api.cab_ws import router as cab_router
from app.api.rest import router as rest_router
from app.api.models_api import router as models_router   # Backend 1 (Pranshu)

log = logging.getLogger("ironsense")


def _load_predict_task():
    try:
        from app.models.task_time import predict_task, load
        load()  # fail fast if artifacts/task_time_model.joblib is missing/corrupt
        return predict_task
    except Exception:
        log.exception("task time model not loaded - task_prediction will be null")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    run_all_seeds()
    app.state.state_manager = StateManager()
    app.state.decision_layer = DecisionLayer(predict_task=_load_predict_task())
    app.state.hub = CabHub(app.state.state_manager, app.state.decision_layer)
    app.state.hub.start()
    yield
    await app.state.hub.stop()


app = FastAPI(title="IronSense backend", lifespan=lifespan)

# The React cab UI is served from another origin (Vite dev server, S3/CloudFront, ...).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("IRONSENSE_CORS_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(telemetry_router)
app.include_router(cab_router)
app.include_router(rest_router)
app.include_router(models_router)
