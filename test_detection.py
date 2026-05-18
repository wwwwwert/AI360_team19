import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
from anomaly_detection import DEFAULT_CONFIGURATION, AnomalyDetectionSystem


def load_csv_dataset(data_path: Path):
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    time_series = df[['value']].copy()
    time_series.columns = ['value_0']
    labels = df['ground_truth'].values.astype(int)
    return time_series, labels


def calculate_metrics(true_labels, predictions):
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1, tp, fp, fn


# Загружаем один датасет
csv_path = Path("data/yandex/series_000/data.csv")
time_series, true_labels = load_csv_dataset(csv_path)
dates = time_series.index

print(f"Загружено {len(time_series)} точек, {true_labels.sum()} аномалий")

# Конфиг с праздниками
CONFIG = DEFAULT_CONFIGURATION.copy()
CONFIG["detection_model_params"]["threshold"] = 3.49
CONFIG["detection_model_params"]["apply_holidays"] = True

system = AnomalyDetectionSystem(**CONFIG)

N_EPOCHS = 100

print(f"\nОбучение на {N_EPOCHS} эпохах без сброса параметров...")
print("=" * 70)
print(f"{'Epoch':<8} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Holiday β':<10}")
print("=" * 70)

best_f1 = 0
best_beta = 2.0

for epoch in range(N_EPOCHS):
    # Детекция
    detection_result = system.detect(time_series=time_series, dates=dates)
    predictions = detection_result.is_anomaly.astype(int)
    anomaly_scores = detection_result.anomaly_scores

    # backward — обучение без сброса
    holiday_param = system.calculate_std_backward(
        time_series=time_series,
        dates=dates,
        ground_truth=true_labels,
        predicted=anomaly_scores
    )

    # Метрики
    precision, recall, f1, tp, fp, fn = calculate_metrics(true_labels, predictions)

    if f1 > best_f1:
        best_f1 = f1
        best_beta = holiday_param

    if epoch % 10 == 0 or epoch < 10:
        print(f"{epoch} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {holiday_param:<10.4f}")

print("=" * 70)
print(f"\nЛучший F1: {best_f1:.4f} при β = {best_beta:.4f}")
print(f"Финальный β после {N_EPOCHS} эпох: {system.holiday_param:.4f}")