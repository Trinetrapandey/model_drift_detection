"""
app/drift.py
------------
The heart of the project: the drift detection logic.

We use TWO classic, lightweight statistical methods (only scipy + numpy needed):

  1. KS test (Kolmogorov-Smirnov)
       Asks: "Do these two samples come from the same distribution?"
       It returns a p-value. A SMALL p-value (e.g. < 0.05) means
       "very unlikely they're the same" => DRIFT for that feature.

  2. PSI (Population Stability Index)
       A number that grows as the new data's distribution moves away
       from the reference. Common rule of thumb:
         PSI < 0.10            -> no real change
         0.10 <= PSI < 0.25    -> moderate change (worth watching)
         PSI >= 0.25           -> significant change (drift!)

For each feature we compute both, then flag the feature as "drifted" if
either method says so. We do this per-feature so you can see WHICH input
changed, not just that *something* changed.
"""

import numpy as np
from scipy import stats

# Thresholds -- the dials you can tune for how sensitive detection is.
#
# Why is the KS threshold so small (0.001, not the usual 0.05)?
# The KS test is VERY sensitive on large samples -- it will often shout "drift!"
# for tiny, meaningless wiggles, giving false alarms. By only trusting it when
# it is extremely confident (p < 0.001), we avoid those false alarms. PSI, which
# is based on the SIZE of the change (not sample count), carries the main signal.
KS_PVALUE_THRESHOLD = 0.001  # below this p-value => KS says drift
PSI_THRESHOLD = 0.25         # at/above this PSI => PSI says drift


def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample.

    Steps in plain words:
      1. Slice the reference data into `bins` buckets (by its own deciles).
      2. See what fraction of reference values fall in each bucket.
      3. See what fraction of current values fall in those SAME buckets.
      4. PSI sums up how different those fractions are.
    """
    # Build bucket edges from the reference data's spread.
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(reference, quantiles)
    # Make sure edges are strictly increasing (guard against duplicate values).
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0
    # Open up the outer edges so values outside the reference range still count.
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Convert counts to fractions. Replace 0 with a tiny number to avoid
    # dividing by zero or taking log(0).
    ref_frac = np.where(ref_counts == 0, 1e-6, ref_counts / len(reference))
    cur_frac = np.where(cur_counts == 0, 1e-6, cur_counts / len(current))

    psi = np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac))
    return float(psi)


def check_feature_drift(reference: np.ndarray, current: np.ndarray) -> dict:
    """Run both tests on a single feature and decide if it drifted."""
    # KS test: compares the two samples directly.
    ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
    psi_value = calculate_psi(reference, current)

    ks_drift = ks_pvalue < KS_PVALUE_THRESHOLD
    psi_drift = psi_value >= PSI_THRESHOLD

    return {
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pvalue), 4),
        "psi": round(psi_value, 4),
        "drift_detected": bool(ks_drift or psi_drift),
    }


def detect_drift(reference_df, current_df, feature_names) -> dict:
    """Check every feature and summarise the result.

    Returns a dictionary that's easy to turn into an API response:
      - per-feature results
      - a list of which features drifted
      - one overall true/false flag
    """
    per_feature = {}
    drifted_features = []

    for feature in feature_names:
        ref_values = reference_df[feature].to_numpy()
        cur_values = current_df[feature].to_numpy()
        result = check_feature_drift(ref_values, cur_values)
        per_feature[feature] = result
        if result["drift_detected"]:
            drifted_features.append(feature)

    return {
        "drift_detected": len(drifted_features) > 0,
        "drifted_features": drifted_features,
        "feature_results": per_feature,
    }
