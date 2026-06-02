"""
monitor.py
----------
This is the bridge between "drift was detected" and "the pipeline runs".

It does two things:
  1. Asks the running service's /drift endpoint to check a batch of recent data.
  2. If drift is found, it tells GitHub to start the retrain pipeline by sending
     a "repository_dispatch" event of type "drift-detected".

You run this wherever your monitoring lives (a cron job, a small server, your
laptop) -- NOT inside the API itself, so no secret tokens live in the web app.

It needs two environment variables (never hard-code these, never commit them):
  GITHUB_TOKEN -> a GitHub Personal Access Token with "repo" / "workflow" scope
  GITHUB_REPO  -> "your-username/drift-detection-fastapi"

Example:
  export GITHUB_TOKEN=ghp_xxx
  export GITHUB_REPO=alice/drift-detection-fastapi
  python monitor.py --url http://YOUR_EC2_IP:5000
"""

import os
import argparse
import requests


def check_drift(service_url: str, batch: dict) -> dict:
    """Call the service's /drift endpoint with a batch of recent data."""
    response = requests.post(f"{service_url}/drift", json=batch, timeout=30)
    response.raise_for_status()
    return response.json()


def trigger_retrain_pipeline() -> None:
    """Tell GitHub to start the retrain.yml workflow via repository_dispatch."""
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPO"]

    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    # "event_type" must match the type listed under repository_dispatch in retrain.yml
    payload = {"event_type": "drift-detected"}

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    print("Retrain pipeline triggered on GitHub (event: drift-detected).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:5000",
                        help="Base URL of the running drift service.")
    args = parser.parse_args()

    # In a real system this batch would be your actual recent production data.
    # Here we send clearly-shifted data so the demo detects drift.
    recent_batch = {
        "size_sqft": [3000 + i * 5 for i in range(100)],   # much bigger than the ~1500 baseline
        "num_rooms": [3 for _ in range(100)],
        "age_years": [20 for _ in range(100)],
    }

    result = check_drift(args.url, recent_batch)
    print("Drift check result:", result["message"])

    if result["drift_detected"]:
        print("Drift detected -> triggering the retrain pipeline...")
        trigger_retrain_pipeline()
    else:
        print("No drift. Nothing to do.")


if __name__ == "__main__":
    main()
