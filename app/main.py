"""
app/main.py
-----------
The FastAPI web service. This is what runs on the EC2 server.

Endpoints:
  GET  /          -> simple welcome / info
  GET  /health    -> health check (the pipeline & monitors hit this)
  POST /predict   -> predict a house price from its features
  POST /drift     -> check a batch of recent data for drift, and LOG the result

When drift is found we do exactly what you asked: report it in the API response
AND write it to a log file. Nothing more (no auto-retraining).
"""

import os
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    HouseFeatures,
    PredictionResponse,
    DriftRequest,
    DriftResponse,
)
from app.drift import detect_drift

# --- Where our saved files live -------------------------------------------
REFERENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference")
MODEL_PATH = os.path.join(REFERENCE_DIR, "model.pkl")
REFERENCE_DATA_PATH = os.path.join(REFERENCE_DIR, "reference_data.csv")

FEATURE_NAMES = ["size_sqft", "num_rooms", "age_years"]

# --- Logging setup --------------------------------------------------------
# Drift events get written here so you have a record over time.
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "drift.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("drift-service")

# --- Load the model and reference data ONCE at import time ----------------
# We load these as the module is imported (not inside a startup event) so they
# are ready the moment the app object exists. Loading once -- not per request --
# keeps responses fast.
app = FastAPI(
    title="Model Drift Detection Service",
    description="A FastAPI service that serves a linear regression model and "
                "detects data drift using KS test + PSI.",
    version="1.0.0",
)


def _load_artifacts():
    """Load the trained model and reference snapshot from disk."""
    loaded_model = None
    loaded_reference = None
    if os.path.exists(MODEL_PATH):
        loaded_model = joblib.load(MODEL_PATH)
    if os.path.exists(REFERENCE_DATA_PATH):
        loaded_reference = pd.read_csv(REFERENCE_DATA_PATH)
    return loaded_model, loaded_reference


model, reference_df = _load_artifacts()
logger.info("Service started. Model loaded: %s", model is not None)


@app.get("/")
def home():
    return {
        "service": "Model Drift Detection",
        "version": "2.0.0",
        "status": "running",
        "deployed_via": "GitHub Actions CI/CD",
        "endpoints": ["/health", "/predict", "/drift", "/docs"],
    }


@app.get("/health")
def health():
    """Returns 200 if the app is alive and the model is loaded."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    """Predict a single house price."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    row = pd.DataFrame([{
        "size_sqft": features.size_sqft,
        "num_rooms": features.num_rooms,
        "age_years": features.age_years,
    }])
    prediction = model.predict(row)[0]
    return PredictionResponse(predicted_price=round(float(prediction), 2))


@app.post("/drift", response_model=DriftResponse)
def drift(request: DriftRequest):
    """Check a batch of recent data against the reference data for drift."""
    if reference_df is None:
        raise HTTPException(status_code=503, detail="Reference data not loaded.")

    # Make sure the three lists are the same length.
    lengths = {len(request.size_sqft), len(request.num_rooms), len(request.age_years)}
    if len(lengths) != 1:
        raise HTTPException(
            status_code=400,
            detail="size_sqft, num_rooms and age_years must all be the same length.",
        )

    current_df = pd.DataFrame({
        "size_sqft": request.size_sqft,
        "num_rooms": request.num_rooms,
        "age_years": request.age_years,
    })

    result = detect_drift(reference_df, current_df, FEATURE_NAMES)

    # --- This is the "what to do on drift" part: report + log ----------
    if result["drift_detected"]:
        message = f"DRIFT DETECTED in features: {result['drifted_features']}"
        logger.warning(message + f" | details={result['feature_results']}")
    else:
        message = "No drift detected. Data looks consistent with the reference."
        logger.info(message)

    return DriftResponse(message=message, **result)
