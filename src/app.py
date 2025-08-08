from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os
from mlflow.tracking import MlflowClient
import mlflow.sklearn

# Configure logging
def setup_logging():
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),      # Log to file
        logging.StreamHandler()              # Also log to console
    ]
)

# Load MLflow model from registry
def load_model_from_registry(model_name: str, stage: str = "Production"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise RuntimeError(f"No versions of '{model_name}' found in stage '{stage}'")
    mv = versions[0]
    model_uri = f"models:/{model_name}/{mv.version}"
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logging.error(f"Failed to load model from {model_uri}: {e}")
        raise

# Define request and response schemas
class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

class PredictionResponse(BaseModel):
    predicted_value: float

# Initialize logging and FastAPI app
setup_logging()
app = FastAPI(title="California Housing Price Predictor")

# Load the model once at startup
model = load_model_from_registry(
    model_name="CaliforniaHousingRegressor", stage="Production"
)

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures):
    try:
        # Convert incoming features to a 2D list for prediction
        feat_list = [list(features.dict().values())]
        logging.info(f"INPUT: {features.dict()}")
        # Predict
        pred = model.predict(feat_list)[0]
        logging.info(f"INPUT: {features.dict()} -> PREDICTION: {pred:.3f}")
        return PredictionResponse(predicted_value=float(pred))
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Send POST to /predict with housing features."}
