import pandas as pd
from pathlib import Path

from test_detection import sweep_beta

DATA_PATH = Path("data/1c/hourly_sales_with_anomalies.csv")


def load_1c_dataset(data_path: Path):
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)
    time_series = df[["total_sells"]].copy()
    time_series.columns = ["value_0"]
    labels = df["is_anomaly"].values
    return time_series, labels


time_series, true_labels = load_1c_dataset(DATA_PATH)

print(f"Загружено {len(time_series)} точек, {true_labels.sum()} аномалий")

sweep_beta(
    datasets=[(time_series, true_labels)],
    threshold=3.49,
    beta_min=1.0,
    beta_max=5.0,
    n_points=100,
    title="F1 vs Holiday β — 1C dataset",
    plot_path="beta_sweep_1c.png",
)
