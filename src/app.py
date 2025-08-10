import os
import time
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Prometheus
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# -------------------------
# NOTE: If you run multiple worker processes (Gunicorn uvicorn workers),
# you must use prometheus_client's multiprocess mode and expose metrics differently.
# This file uses single-process mode which works for simple deployments.
# -------------------------

# -------------------------
# Configure logging
# -------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )

# -------------------------
# Prometheus metrics
# -------------------------
PREDICTION_COUNTER = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["model_version", "method"]
)
PREDICTION_FAILURES = Counter(
    "prediction_failures_total",
    "Total number of failed prediction requests",
    ["model_version"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model_version"]
)
MODEL_LOADED = Gauge(
    "model_loaded_info",
    "Info about currently loaded model (label useful for queries)",
    ["model_version"]
)
IN_PROGRESS = Gauge("in_progress_requests", "In-progress request count")
REQUEST_COUNT = Counter("api_requests_total", "Total API Requests", ["endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latency of API requests", ["endpoint"])


# Helper to compute model_version label from CURRENT_MODEL_URI
def model_version_label() -> str:
    uri = globals().get("CURRENT_MODEL_URI", "")
    if not uri:
        return "unknown"
    # prefer the last meaningful segment (e.g. models:/Name/3 -> '3', runs:/<run_id>/model -> '<run_id>')
    parts = [p for p in uri.split("/") if p]
    return parts[-1] if parts else "unknown"

# -------------------------
# Load MLflow model helper
# -------------------------
def load_model_from_registry(
    model_name: str,
    stage: str = "Production",
    run_id: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    artifact_path: str = "model",
):
    """
    Loads model either from a specific run (runs:/) if run_id provided,
    otherwise loads the latest version in the specified registry stage (models:/).
    Returns (model, model_uri).
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    else:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    client = MlflowClient()

    if run_id:
        model_uri = f"runs:/{run_id}/{artifact_path}"
    else:
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise RuntimeError(f"No versions of '{model_name}' found in stage '{stage}'")
        mv = versions[0]
        model_uri = f"models:/{model_name}/{mv.version}"

    logging.info(f"Attempting to load model from URI: {model_uri}")
    try:
        # Using sklearn flavor; if your models are logged as pyfunc, use mlflow.pyfunc.load_model instead.
        model = mlflow.sklearn.load_model(model_uri)
        logging.info(f"Loaded model class: {model.__class__.__name__} from {model_uri}")
        return model, model_uri
    except Exception as e:
        logging.error(f"Failed to load model from {model_uri}: {e}", exc_info=True)
        raise

# -------------------------
# Schemas
# -------------------------
class HousingFeatures(BaseModel):
    MedInc: float = Field(..., ge=0, le=20, description="Median income in block group (tens of thousands USD)")
    HouseAge: float = Field(..., ge=0, le=100, description="Median house age in years")
    AveRooms: float = Field(..., ge=0, le=50, description="Average number of rooms per household")
    AveBedrms: float = Field(..., ge=0, le=10, description="Average number of bedrooms per household")
    Population: float = Field(..., ge=0, le=50000, description="Block group population")
    AveOccup: float = Field(..., ge=0.1, le=20, description="Average household occupancy")
    Latitude: float = Field(..., ge=32, le=42, description="Latitude (California range)")
    Longitude: float = Field(..., ge=-125, le=-113, description="Longitude (California range)")

    @validator("AveBedrms")
    def bedrooms_not_more_than_rooms(cls, v, values):
        if "AveRooms" in values and v > values["AveRooms"]:
            raise ValueError("Average bedrooms cannot exceed average rooms")
        return v

class PredictionResponse(BaseModel):
    predicted_value: float

# -------------------------
# App init
# -------------------------
setup_logging()
app = FastAPI(title="California Housing Price Predictor")

MODEL_NAME = os.getenv("MODEL_NAME", "CaliforniaHousingRegressor")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MODEL_RUN_ID = os.getenv("MODEL_RUN_ID", None)           # set this to switch to a run (restart required)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None)
ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT_PATH", "model")  # artifact path used when logging model

# ensure the global CURRENT_MODEL_URI exists before model load (middleware can reference)
CURRENT_MODEL_URI = ""

# Load the model once at startup (Option A: change env and restart to switch)
try:
    model, CURRENT_MODEL_URI = load_model_from_registry(
        model_name=MODEL_NAME,
        stage=MODEL_STAGE,
        run_id=MODEL_RUN_ID,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        artifact_path=ARTIFACT_PATH,
    )
    # set model_loaded gauge (label value -> 1)
    MODEL_LOADED.labels(model_version=model_version_label()).set(1)
except Exception as e:
    logging.exception("Model loading failed at startup.")
    # Fail fast so deployment shows error
    raise

# -------------------------
# Prometheus middleware
# -------------------------
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    path = request.url.path
    start = time.time()
    IN_PROGRESS.inc()
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Count failures for prediction endpoint
        if path.startswith("/predict"):
            try:
                PREDICTION_FAILURES.labels(model_version=model_version_label()).inc()
            except Exception:
                logging.exception("Failed incrementing failure counter")
        raise
    finally:
        elapsed = time.time() - start
        IN_PROGRESS.dec()
        if path.startswith("/predict"):
            try:
                PREDICTION_COUNTER.labels(model_version=model_version_label(), method=request.method).inc()
                PREDICTION_LATENCY.labels(model_version=model_version_label()).observe(elapsed)
                process_time = time.time() - start
                REQUEST_COUNT.labels(endpoint=request.url.path).inc()
                REQUEST_LATENCY.labels(endpoint=request.url.path).observe(process_time)
            except Exception:
                logging.exception("Failed updating metrics")

# Expose metrics for Prometheus to scrape
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# -------------------------
# Endpoints
# -------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures):
    try:
        feat_list = [list(features.dict().values())]
        logging.info(f"INPUT: {features.dict()}")
        pred = model.predict(feat_list)[0]
        logging.info(f"INPUT: {features.dict()} -> PREDICTION: {pred:.3f}")
        return PredictionResponse(predicted_value=float(pred))
    except Exception as e:
        logging.exception("Prediction failed")
        # increment failure counter explicitly (in addition to middleware)
        try:
            PREDICTION_FAILURES.labels(model_version=model_version_label()).inc()
        except Exception:
            logging.exception("Failed incrementing failure counter in handler")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {
        "message": "Send POST to /predict with housing features.",
        "model_uri": CURRENT_MODEL_URI
    }
