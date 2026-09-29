"""
FastAPI application for credit default risk scoring.

"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    DECLINE_THRESHOLD,
    MODEL_VERSION,
    REVIEW_THRESHOLD,
)
from app.model import CreditRiskModel
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CreditApplication,
    HealthResponse,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model: CreditRiskModel | None = None


# =============================================================================
# Lifespan: load the model once, at startup
# =============================================================================
# Loading a model takes time. Doing it per-request would add that cost to every
# call; doing it at import time would break the test client. The lifespan hook
# is the right place — it runs once when the process starts.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the process starts, release it when it stops."""
    global model
    try:
        model = CreditRiskModel()
        logger.info("Model loaded at startup")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model: %s", exc)
        model = None
    yield
    model = None


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health (PROVIDED — do not modify)
# =============================================================================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Report whether the service can serve predictions right now."""
    ready = model is not None and model.is_loaded()
    return HealthResponse(
        status="healthy" if ready else "unhealthy",
        model_loaded=ready,
        model_version=MODEL_VERSION,
    )


# =============================================================================
# Prediction endpoints
# =============================================================================
def _require_model() -> CreditRiskModel:
    """503, not 500: the request was fine, the service is not ready."""
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return model


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(application: CreditApplication):
    """Score one applicant and return the underwriting decision."""
    scorer = _require_model()
    try:
        return PredictionResponse(**scorer.score(application.model_dump()))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Score up to 500 applicants in one call, preserving request order."""
    scorer = _require_model()
    try:
        results = scorer.score_batch([a.model_dump() for a in request.applications])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Batch prediction failed") from exc
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**r) for r in results],
        total_count=len(results),
    )


# =============================================================================
# Info endpoints
# =============================================================================
@app.get("/", tags=["Info"])
async def root():
    """API metadata and where to find the docs."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/model/info", tags=["Info"])
async def model_info():
    """Model version, training metrics and the thresholds in force."""
    return {
        "model_version": MODEL_VERSION,
        "model_type": (model.metadata.get("model_type") if model else None),
        "trained_at": (model.metadata.get("trained_at") if model else None),
        "metrics": (model.metadata.get("metrics") if model else None),
        "review_threshold": REVIEW_THRESHOLD,
        "decline_threshold": DECLINE_THRESHOLD,
        "is_loaded": model is not None and model.is_loaded(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
