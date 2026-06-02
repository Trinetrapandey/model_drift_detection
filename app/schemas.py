from pydantic import BaseModel
from typing import List, Dict, Any


class HouseFeatures(BaseModel):
    size_sqft: float
    num_rooms: float
    age_years: float


class PredictionResponse(BaseModel):
    predicted_price: float


class DriftRequest(BaseModel):
    size_sqft: List[float]
    num_rooms: List[float]
    age_years: List[float]


class DriftResponse(BaseModel):
    message: str
    drift_detected: bool
    drifted_features: List[str]
    feature_results: Dict[str, Any]
