import numpy as np
import pandas as pd
from pathlib import Path

from test_detection import sweep_beta

DATA_DIR = Path("data/yandex")
ALL_SERIES = sorted(DATA_DIR.glob("series_*/data.csv"))


def load_yandex_dataset(data_path: Path):
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    time_series = df[["value"]].copy()
    time_series.columns = ["value_0"]
    labels = df["ground_truth"].values.astype(int)
    return time_series, labels


def load_all(paths: list[Path]):
    return [load_yandex_dataset(p) for p in paths]


def prompt(text, default, cast):
    raw = input(f"{text} [{default}]: ").strip()
    return cast(raw) if raw else default


print("=== Yandex anomaly detection — настройка ===")
print(f"Доступные датасеты: 0–{len(ALL_SERIES)-1} или 'all'")

series_raw = input("Датасет (номер 0-13 или all) [all]: ").strip()
if series_raw == "" or series_raw.lower() == "all":
    paths = ALL_SERIES
    series_name = "all (combined)"
    plot_path = "training_curve_yandex_all.png"
    title = "Training curve — Yandex all 14 datasets combined"
else:
    idx = int(series_raw)
    assert 0 <= idx <= 13, "Индекс датасета должен быть от 0 до 13"
    paths = [ALL_SERIES[idx]]
    series_name = f"series_{idx:03d}"
    plot_path = f"training_curve_yandex_{idx:03d}.png"
    title = f"Training curve — Yandex {series_name}"

n_points  = prompt("Количество точек sweep", 250,  int)
threshold = prompt("Threshold",              3.49, float)
beta_min  = prompt("Beta min",               1.0,  float)
beta_max  = prompt("Beta max",               5.0,  float)

datasets = load_all(paths)
total_points = sum(len(ts) for ts, _ in datasets)
total_anomalies = sum(lbl.sum() for _, lbl in datasets)

print(f"\nДатасет:   {series_name}")
print(f"Точек:     {total_points}, аномалий: {total_anomalies}")
print(f"Sweep:     β от {beta_min} до {beta_max}, {n_points} точек")
print(f"Threshold: {threshold}")

sweep_beta(
    datasets=datasets,
    threshold=threshold,
    beta_min=beta_min,
    beta_max=beta_max,
    n_points=n_points,
    title=title,
    plot_path=plot_path,
)
