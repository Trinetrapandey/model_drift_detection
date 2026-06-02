"""
tests/test_app.py
-----------------
Automated tests. The CI stage of the pipeline runs these. If any fail,
the broken code is NEVER deployed to EC2.

We test three things:
  1. The basic endpoints respond correctly.
  2. The model makes a sensible prediction.
  3. The drift detector correctly says "no drift" for similar data
     and "drift!" for clearly shifted data.
"""

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.drift import detect_drift, calculate_psi

client = TestClient(app)

FEATURE_NAMES = ["size_sqft", "num_rooms", "age_years"]


def _make_reference():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "size_sqft": rng.normal(1500, 400, 500),
        "num_rooms": rng.normal(3, 1, 500),
        "age_years": rng.normal(20, 10, 500),
    })


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_predict_endpoint():
    response = client.post("/predict", json={
        "size_sqft": 1500, "num_rooms": 3, "age_years": 20,
    })
    assert response.status_code == 200
    # A normal house should have a positive predicted price.
    assert response.json()["predicted_price"] > 0


def test_no_drift_for_similar_data():
    """Data drawn from the SAME distribution should NOT trigger drift."""
    rng = np.random.default_rng(1)
    reference = _make_reference()
    current = pd.DataFrame({
        "size_sqft": rng.normal(1500, 400, 300),
        "num_rooms": rng.normal(3, 1, 300),
        "age_years": rng.normal(20, 10, 300),
    })
    result = detect_drift(reference, current, FEATURE_NAMES)
    assert result["drift_detected"] is False


def test_drift_for_shifted_data():
    """Data with a clearly shifted distribution SHOULD trigger drift."""
    rng = np.random.default_rng(2)
    reference = _make_reference()
    current = pd.DataFrame({
        # size shifted way up: 1500 -> 3000 sqft on average
        "size_sqft": rng.normal(3000, 400, 300),
        "num_rooms": rng.normal(3, 1, 300),
        "age_years": rng.normal(20, 10, 300),
    })
    result = detect_drift(reference, current, FEATURE_NAMES)
    assert result["drift_detected"] is True
    assert "size_sqft" in result["drifted_features"]


def test_psi_zero_for_identical_data():
    """PSI of a sample against itself should be ~0 (no change)."""
    values = np.random.default_rng(3).normal(0, 1, 1000)
    assert calculate_psi(values, values) < 0.01


def test_drift_endpoint_reports_shift():
    """End-to-end: the /drift endpoint flags clearly shifted data."""
    rng = np.random.default_rng(4)
    payload = {
        "size_sqft": list(rng.normal(5000, 400, 200)),  # huge shift up
        "num_rooms": list(rng.normal(3, 1, 200)),
        "age_years": list(rng.normal(20, 10, 200)),
    }
    response = client.post("/drift", json=payload)
    assert response.status_code == 200
    assert response.json()["drift_detected"] is True


# ----- Retraining / promotion logic ----------------------------------

def test_should_promote_when_no_champion():
    """With no existing model, any challenger should be promoted."""
    from retrain import should_promote
    assert should_promote(None, 5000.0) is True


def test_should_promote_when_challenger_better():
    """A challenger with lower error replaces the champion."""
    from retrain import should_promote
    assert should_promote(10000.0, 4000.0) is True


def test_should_keep_champion_when_challenger_worse():
    """A challenger that is worse must NOT be promoted."""
    from retrain import should_promote
    assert should_promote(4000.0, 9000.0) is False


def test_challenger_beats_stale_champion_on_new_world():
    """The freshly trained challenger should beat a stale champion when the
    underlying relationship has changed (concept drift)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from retrain import make_fresh_data, compute_rmse
    from train import make_dataset, FEATURE_NAMES

    # Champion trained on the ORIGINAL world.
    original = make_dataset()
    champion = LinearRegression().fit(original[FEATURE_NAMES], original["price"])

    # Challenger trained on the NEW world.
    fresh = make_fresh_data()
    X = fresh[FEATURE_NAMES]
    y = fresh["price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    challenger = LinearRegression().fit(X_tr, y_tr)

    champ_rmse = compute_rmse(champion, X_te, y_te)
    chall_rmse = compute_rmse(challenger, X_te, y_te)
    assert chall_rmse < champ_rmse
