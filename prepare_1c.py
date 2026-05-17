"""
Aggregates 1C sales_train.csv into a daily time series:
    date, total_sells, holiday_mark

holiday_mark: code of the nearest Russian holiday (empty string if none).
Window: N_days * WINDOW_FACTOR on each side of the holiday.
"""

from pathlib import Path
from datetime import timedelta

import holidays
import pandas as pd

# --- config ---
WINDOW_FACTOR = 0.5
IN_PATH = Path("data/1c/sales_train.csv")
OUT_PATH = Path("data/1c/daily_sales.csv")

HOLIDAY_CODES = {
    "Новогодние каникулы":               "NEW_YEAR",
    "Рождество Христово":                "CHRISTMAS_ORTHODOX",
    "День защитника Отечества":          "DEFENDER_DAY",
    "Международный женский день":        "WOMENS_DAY",
    "Международный женский день (выходной)": "WOMENS_DAY",
    "Праздник Весны и Труда":            "LABOUR_DAY",
    "День Победы":                       "VICTORY_DAY",
    "День Победы (выходной)":            "VICTORY_DAY",
    "День России":                       "RUSSIA_DAY",
    "День народного единства":           "UNITY_DAY",
}
# Перенесённые выходные — группируем в TRANSFERRED
TRANSFERRED_PREFIX = "Выходной (перенесено"


def get_holiday_windows(year_start: int, year_end: int, window_factor: float) -> list[tuple]:
    hols = holidays.country_holidays("RU", years=range(year_start, year_end + 1))

    by_code: dict[str, list] = {}
    for date, name in sorted(hols.items()):
        if name.startswith(TRANSFERRED_PREFIX):
            code = "TRANSFERRED"
        else:
            code = HOLIDAY_CODES.get(name, name.upper().replace(" ", "_")[:20])
        by_code.setdefault(code, []).append(date)

    windows = []
    for code, dates in by_code.items():
        dates = sorted(dates)
        runs = []
        run = [dates[0]]
        for d in dates[1:]:
            if (d - run[-1]).days == 1:
                run.append(d)
            else:
                runs.append(run)
                run = [d]
        runs.append(run)

        for run in runs:
            n_days = len(run)
            delta = timedelta(days=max(1, int(n_days * window_factor)))
            win_start = pd.Timestamp(run[0]) - delta
            win_end = pd.Timestamp(run[-1]) + delta
            windows.append((win_start, win_end, code))

    return windows


def mark_holidays(dates: pd.Series, window_factor: float) -> pd.Series:
    windows = get_holiday_windows(dates.min().year, dates.max().year, window_factor)
    marks = pd.Series("", index=dates.index)
    for win_start, win_end, code in windows:
        mask = (dates >= win_start) & (dates <= win_end)
        marks[mask] = code
    return marks


def main():
    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    daily = (
        df.groupby("date")["item_cnt_day"]
        .sum()
        .reset_index()
        .rename(columns={"item_cnt_day": "total_sells"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily["holiday_mark"] = mark_holidays(daily["date"], WINDOW_FACTOR)

    daily.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(daily)} rows to {OUT_PATH}")
    print(f"Date range: {daily['date'].min().date()} — {daily['date'].max().date()}")
    print(f"Holiday-marked days: {(daily['holiday_mark'] != '').sum()}")
    codes = daily.loc[daily['holiday_mark'] != '', 'holiday_mark'].unique()
    print(f"Holiday codes: {', '.join(sorted(codes))}")


if __name__ == "__main__":
    main()
