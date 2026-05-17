"""
Expands daily_sales.csv to hourly resolution.
- Each day's value is anchored at midnight (00:00).
- Values between anchors are interpolated with cubic spline (smooth).
- holiday_mark is forward-filled from midnight of each day.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

IN_PATH = Path("data/1c/daily_sales.csv")
OUT_PATH = Path("data/1c/hourly_sales.csv")


def main():
    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Anchor points: midnight of each day in hours-since-start
    t0 = df["date"].iloc[0]
    anchor_hours = ((df["date"] - t0).dt.total_seconds() / 3600).to_numpy()
    anchor_values = df["total_sells"].to_numpy(dtype=float)

    cs = CubicSpline(anchor_hours, anchor_values)

    last_hour = int(anchor_hours[-1])
    all_hours = np.arange(0, last_hour + 1, dtype=float)
    interp_values = cs(all_hours)
    # Clamp negatives that spline may produce in low-value regions
    interp_values = np.maximum(interp_values, 0.0)

    timestamps = pd.date_range(start=t0, periods=len(all_hours), freq="h")

    # Build holiday_mark: assign each day's mark to all 24 hours of that day
    date_to_mark = dict(zip(df["date"], df["holiday_mark"].fillna("")))
    holiday_marks = [
        date_to_mark.get(pd.Timestamp(ts.date()), "")
        for ts in timestamps
    ]

    out = pd.DataFrame({
        "timestamp": timestamps,
        "total_sells": np.round(interp_values, 2),
        "holiday_mark": holiday_marks,
    })

    out.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(out)} rows to {OUT_PATH}")
    print(f"Range: {out['timestamp'].iloc[0]} — {out['timestamp'].iloc[-1]}")
    print(f"Holiday-marked hours: {(out['holiday_mark'] != '').sum()}")


if __name__ == "__main__":
    main()
