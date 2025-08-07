from fastapi import FastAPI, HTTPException
from src.schemas import HousingFeatures, PredictionResponse
from src.model_loader import load_model
import logging

# Configure logging
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = FastAPI(title="California Housing Price Predictor")

# Load model once at startup
model = load_model(model_name="CaliforniaHousingRegressor")

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HousingFeatures):
    """
    Accepts housing features as JSON, returns the median house value prediction.
    """
    try:
        # Convert to 2D array for sklearn
        data = [features.dict().values()]
        pred = model.predict(data)[0]
        
        # Logging request and response
        logging.info(f"INPUT: {features.dict()} -> PREDICTION: {pred:.3f}")
        
        return {"predicted_value": float(pred)}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/")
def root():
    return {"message": "Send POST to /predict with housing features."}
