import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# -------------------------
# Configure logging
# -------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("app.log"),      # Log to file
            logging.StreamHandler()              # Also log to console
        ]
    )

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

# Load the model once at startup (Option A: change env and restart to switch)
try:
    model, CURRENT_MODEL_URI = load_model_from_registry(
        model_name=MODEL_NAME,
        stage=MODEL_STAGE,
        run_id=MODEL_RUN_ID,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        artifact_path=ARTIFACT_PATH,
    )
except Exception as e:
    logging.exception("Model loading failed at startup.")
    # Optionally exit if you want the container to fail fast
    raise

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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {
        "message": "Send POST to /predict with housing features.",
        "model_uri": CURRENT_MODEL_URI
    }
