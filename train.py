"""
train.py
--------
This script does two jobs ONCE, before the service ever runs:

  1. Creates a simple dataset and trains a Linear Regression model on it.
  2. Saves TWO things to the reference/ folder:
       - model.pkl          -> the trained model
       - reference_data.csv -> the data the model was trained on

Why save the reference data?
  Drift detection works by COMPARING new incoming data against the original
  "reference" data the model learned from. So we must keep a snapshot of it.

We use a fixed random seed so the dataset is identical every time. That makes
the whole demo reproducible.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Folder where we keep the model and reference snapshot
REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference")
MODEL_PATH = os.path.join(REFERENCE_DIR, "model.pkl")
REFERENCE_DATA_PATH = os.path.join(REFERENCE_DIR, "reference_data.csv")

# The three input features our model uses. Keeping these names in one place
# means the API and the drift checker can both import them.
FEATURE_NAMES = ["size_sqft", "num_rooms", "age_years"]


def make_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create a simple, relatable 'house price' dataset.

    price ~= a linear combination of size, rooms, and age, plus a little noise.
    Because price really is a linear function of the inputs here, a linear
    regression model fits it well -- perfect for a simple demo.
    """
    rng = np.random.default_rng(seed)

    size_sqft = rng.normal(loc=1500, scale=400, size=n_samples)   # ~1500 sqft homes
    num_rooms = rng.normal(loc=3, scale=1, size=n_samples)        # ~3 rooms
    age_years = rng.normal(loc=20, scale=10, size=n_samples)      # ~20 years old

    noise = rng.normal(loc=0, scale=10000, size=n_samples)
    price = (
        150 * size_sqft        # each sqft adds ~$150
        + 20000 * num_rooms    # each room adds ~$20k
        - 1000 * age_years     # each year of age removes ~$1k
        + 50000                # base price
        + noise
    )

    return pd.DataFrame(
        {
            "size_sqft": size_sqft,
            "num_rooms": num_rooms,
            "age_years": age_years,
            "price": price,
        }
    )


def main():
    os.makedirs(REFERENCE_DIR, exist_ok=True)

    # 1. Build the dataset
    data = make_dataset()

    # 2. Train the linear regression model
    X = data[FEATURE_NAMES]
    y = data["price"]
    model = LinearRegression()
    model.fit(X, y)

    # 3. Save the model and the reference snapshot (features only -- that's what
    #    we compare incoming data against).
    joblib.dump(model, MODEL_PATH)
    data[FEATURE_NAMES].to_csv(REFERENCE_DATA_PATH, index=False)

    print(f"Model trained. R^2 on training data: {model.score(X, y):.4f}")
    print(f"Saved model to:          {MODEL_PATH}")
    print(f"Saved reference data to: {REFERENCE_DATA_PATH}")


if __name__ == "__main__":
    main()
