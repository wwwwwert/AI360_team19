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
ANOMALY_RATE = 0.01        # 1% of non-holiday points
ANOMALY_RATE_HOLIDAY = 0.10  # 10% of holiday points
COEF_MIN = 0.1
COEF_MAX = 5.0
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


def pick_indices(candidate_indices: np.ndarray, n: int, rng: np.random.Generator) -> list:
    """Sample up to n indices from candidate_indices with MIN_GAP_HOURS constraint."""
    chosen = []
    shuffled = rng.permutation(candidate_indices)
    for idx in shuffled:
        if len(chosen) >= n:
            break
        if all(abs(idx - c) >= MIN_GAP_HOURS for c in chosen):
            chosen.append(int(idx))
    return sorted(chosen)


def main():
    df = pd.read_csv(IN_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_anomaly"] = 0

    rng = np.random.default_rng(SEED)

    holiday_idx = np.where(df["holiday_mark"].notna() & (df["holiday_mark"] != ""))[0]
    normal_idx  = np.where(df["holiday_mark"].isna()  | (df["holiday_mark"] == ""))[0]

    n_holiday = int(len(holiday_idx) * ANOMALY_RATE_HOLIDAY)
    n_normal  = int(len(normal_idx)  * ANOMALY_RATE)

    chosen_holiday = pick_indices(holiday_idx, n_holiday, rng)
    chosen_normal  = pick_indices(normal_idx,  n_normal,  rng)
    chosen = sorted(chosen_holiday + chosen_normal)

    for idx in chosen:
        coef = sample_anomaly_coef(rng)
        df.at[idx, "total_sells"] = round(df.at[idx, "total_sells"] * coef, 2)
        df.at[idx, "is_anomaly"] = 1

    df.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(f"Injected {len(chosen)} anomalies ({len(chosen)/len(df)*100:.2f}%)")
    print(f"  Holiday anomalies: {len(chosen_holiday)} / {len(holiday_idx)} ({100*len(chosen_holiday)/len(holiday_idx):.1f}%)")
    print(f"  Normal  anomalies: {len(chosen_normal)}  / {len(normal_idx)}  ({100*len(chosen_normal)/len(normal_idx):.1f}%)")


if __name__ == "__main__":
    main()
