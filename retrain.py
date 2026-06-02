"""
retrain.py
----------
This is the script that runs AFTER drift has been detected. Its job is to
produce a NEW model and decide whether it actually deserves to be deployed.

It uses the classic "champion vs challenger" pattern:

  - CHAMPION   = the model currently live in production (reference/model.pkl).
  - CHALLENGER = a brand-new model trained on fresh, recent data.

We then test BOTH on the same fresh hold-out data and only promote the
challenger if it is genuinely better than the champion. This is the
"validate before you deploy" safety gate -- we never blindly trust a new model.

If the challenger wins:
  - the old champion is archived (so we can roll back),
  - the challenger becomes the new model.pkl,
  - the reference snapshot is updated to the fresh data (the new "normal"),
  - we signal the pipeline to deploy.

If the challenger does NOT win:
  - nothing changes, nothing is deployed.

Note on the demo data:
  train.py creates the ORIGINAL world. Here, make_fresh_data() simulates a
  CHANGED world (a hotter market: bigger houses AND a higher price per sqft).
  Because the underlying relationship changed, the old champion makes worse
  predictions on the new data, so the freshly trained challenger clearly wins.
"""

import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference")
ARCHIVE_DIR = os.path.join(REFERENCE_DIR, "archive")
MODEL_PATH = os.path.join(REFERENCE_DIR, "model.pkl")
REFERENCE_DATA_PATH = os.path.join(REFERENCE_DIR, "reference_data.csv")

FEATURE_NAMES = ["size_sqft", "num_rooms", "age_years"]


def make_fresh_data(n_samples: int = 1000, seed: int = 99) -> pd.DataFrame:
    """Simulate RECENT real-world data after the market has shifted.

    Compared to the original training data (in train.py):
      - houses are bigger on average,
      - and crucially the *relationship* changed: price per sqft is higher.
    This is the kind of change that makes the old model stale.
    """
    rng = np.random.default_rng(seed)

    size_sqft = rng.normal(loc=2200, scale=500, size=n_samples)
    num_rooms = rng.normal(loc=3.5, scale=1, size=n_samples)
    age_years = rng.normal(loc=18, scale=9, size=n_samples)

    noise = rng.normal(loc=0, scale=12000, size=n_samples)
    price = (
        230 * size_sqft        # was 150 -> price per sqft went UP
        + 25000 * num_rooms    # was 20000
        - 900 * age_years      # was -1000
        + 70000                # base price went up (was 50000)
        + noise
    )

    return pd.DataFrame({
        "size_sqft": size_sqft,
        "num_rooms": num_rooms,
        "age_years": age_years,
        "price": price,
    })


def compute_rmse(model, X, y) -> float:
    """Root Mean Squared Error: average prediction error, in dollars.

    Lower is better. We use it to compare champion vs challenger fairly.
    """
    predictions = model.predict(X)
    return float(np.sqrt(np.mean((y - predictions) ** 2)))


def should_promote(champion_rmse, challenger_rmse) -> bool:
    """Decide whether the challenger is good enough to replace the champion.

    Promote if there's no champion yet, or if the challenger has a strictly
    lower error on the fresh hold-out data.
    """
    if champion_rmse is None:
        return True
    return challenger_rmse < champion_rmse


def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # 1. Get fresh data and split it into train / test parts.
    fresh = make_fresh_data()
    X = fresh[FEATURE_NAMES]
    y = fresh["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # 2. Train the CHALLENGER on the fresh training data.
    challenger = LinearRegression()
    challenger.fit(X_train, y_train)
    challenger_rmse = compute_rmse(challenger, X_test, y_test)

    # 3. Load the CHAMPION (current live model), if there is one.
    champion = None
    champion_rmse = None
    if os.path.exists(MODEL_PATH):
        champion = joblib.load(MODEL_PATH)
        champion_rmse = compute_rmse(champion, X_test, y_test)

    # 4. Print a clear comparison.
    print("--- Validation on fresh hold-out data ---")
    print(f"Champion RMSE:   {champion_rmse if champion_rmse is None else round(champion_rmse, 2)}")
    print(f"Challenger RMSE: {round(challenger_rmse, 2)}")

    promoted = should_promote(champion_rmse, challenger_rmse)

    if promoted:
        # 5a. Archive the old champion so we can roll back if needed.
        if champion is not None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(ARCHIVE_DIR, f"model_{stamp}.pkl")
            shutil.copy(MODEL_PATH, archive_path)
            print(f"Archived old champion to: {archive_path}")

        # 5b. Save the challenger as the new live model.
        joblib.dump(challenger, MODEL_PATH)

        # 5c. Move the reference baseline FORWARD to the new normal, so future
        #     drift checks compare against today's reality, not the old one.
        fresh[FEATURE_NAMES].to_csv(REFERENCE_DATA_PATH, index=False)

        print("RESULT: PROMOTED. New model + updated reference saved.")
    else:
        print("RESULT: KEPT CHAMPION. Challenger was not better; nothing changes.")

    # 6. If running inside GitHub Actions, tell the workflow what happened.
    #    The workflow uses this to decide whether to commit + deploy.
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as f:
            f.write(f"promoted={'true' if promoted else 'false'}\n")

    return promoted


if __name__ == "__main__":
    main()
