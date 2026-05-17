"""
Injects synthetic anomalies into hourly_sales.csv.
Each anomaly multiplies the point's value by a random coefficient in [COEF_MIN, COEF_MAX],
excluding the range [NORMAL_LOW, NORMAL_HIGH] (so the multiplier is always "anomalous").

Anomalies are injected as isolated points (not consecutive) separated by at least
MIN_GAP_HOURS hours to avoid clustering.

Config:
    ANOMALY_RATE   - fraction of points to mark as anomalies
    COEF_MIN/MAX   - outer bounds of the multiplier range
    NORMAL_LOW/HIGH - inner bounds excluded from multiplier (the "normal" zone)
    MIN_GAP_HOURS  - minimum hours between two anomalies
    SEED           - random seed for reproducibility
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --- config ---
ANOMALY_RATE = 0.01        # 1% of points
COEF_MIN = 0.5
COEF_MAX = 2.0
NORMAL_LOW = 0.85          # multipliers in [0.85, 1.15] are considered normal
NORMAL_HIGH = 1.15
MIN_GAP_HOURS = 6
SEED = 42

IN_PATH = Path("data/1c/hourly_sales.csv")
OUT_PATH = Path("data/1c/hourly_sales_with_anomalies.csv")


def sample_anomaly_coef(rng: np.random.Generator) -> float:
    """Sample from [COEF_MIN, NORMAL_LOW) ∪ (NORMAL_HIGH, COEF_MAX]."""
    low_range = NORMAL_LOW - COEF_MIN
    high_range = COEF_MAX - NORMAL_HIGH
    total = low_range + high_range
    if rng.random() < low_range / total:
        return rng.uniform(COEF_MIN, NORMAL_LOW)
    else:
        return rng.uniform(NORMAL_HIGH, COEF_MAX)


def main():
    df = pd.read_csv(IN_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_anomaly"] = 0

    rng = np.random.default_rng(SEED)
    n_anomalies = int(len(df) * ANOMALY_RATE)

    # Pick anomaly indices with minimum gap constraint:
    # repeatedly sample a candidate, accept if far enough from all chosen.
    chosen = set()
    attempts = 0
    while len(chosen) < n_anomalies and attempts < n_anomalies * 100:
        attempts += 1
        idx = int(rng.integers(0, len(df)))
        if all(abs(idx - c) >= MIN_GAP_HOURS for c in chosen):
            chosen.add(idx)
    chosen = sorted(chosen)

    for idx in chosen:
        coef = sample_anomaly_coef(rng)
        df.at[idx, "total_sells"] = round(df.at[idx, "total_sells"] * coef, 2)
        df.at[idx, "is_anomaly"] = 1

    df.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(f"Injected {len(chosen)} anomalies ({len(chosen)/len(df)*100:.2f}%)")
    print(f"Anomalies on holiday-marked hours: {df.loc[df['is_anomaly']==1, 'holiday_mark'].ne('').sum()}")


if __name__ == "__main__":
    main()
