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


def main():
    datasets = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith('series')])

    if not datasets:
        print(f"Датасеты не найдены в {DATA_DIR}")
        return

    print(f"Найдено {len(datasets)} датасетов\n")

    holiday_param = np.float64(2.0)

    CONFIG = DEFAULT_CONFIGURATION.copy()
    CONFIG["detection_model_params"]["threshold"] = 3.49
    CONFIG["detection_model_params"]["apply_holidays"] = True  # Включаем праздники
    CONFIG["detection_model_params"]["holiday_param"] = holiday_param

    system = AnomalyDetectionSystem(**CONFIG)

    all_metrics = []
    total_tp = total_fp = total_fn = 0

    print("=" * 90)
    print(f"{'Датасет':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Holiday β':<10}")
    print("=" * 90)

    for dataset_path in datasets:
        csv_path = dataset_path / "data.csv"

        if not csv_path.exists():
            print(f"{dataset_path.name:<15} Файл не найден")
            continue

        time_series, true_labels = load_csv_dataset(csv_path)
        dates = time_series.index

        # Первый проход — детекция
        detection_result = system.detect(time_series=time_series, dates=dates)
        predictions = detection_result.is_anomaly.astype(int)
        anomaly_scores = detection_result.anomaly_scores

        # backward — обучаем holiday_param
        residual = detection_result.anomaly_scores  # или detection_result.expected_value


        holiday_param = system.calculate_std_backward(time_series=time_series, dates=dates, ground_truth=true_labels, predicted=predictions)

        # Второй проход с обновлённым holiday_param
        detection_result = system.detect(time_series=time_series)
        predictions = detection_result.is_anomaly.astype(int)

        precision, recall, f1, tp, fp, fn = calculate_metrics(true_labels, predictions)

        all_metrics.append({
            'dataset': dataset_path.name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp, 'fp': fp, 'fn': fn,
            'holiday_beta': holiday_param
        })

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"{dataset_path.name:<15} "
              f"{precision:<10.4f} "
              f"{recall:<10.4f} "
              f"{f1:<10.4f} "
              f"{holiday_param:<10.4f}")

    print("=" * 90)

    if all_metrics:
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = (2 * overall_precision * overall_recall /
                      (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0)

        print(f"\n{'ОБЩИЕ РЕЗУЛЬТАТЫ':<20}")
        print(f"{'Всего датасетов:':<20} {len(all_metrics)}")
        print(f"{'Overall Precision:':<20} {overall_precision:.4f}")
        print(f"{'Overall Recall:':<20} {overall_recall:.4f}")
        print(f"{'Overall F1-score:':<20} {overall_f1:.4f}")

        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv('csv_all_metrics_holidays.csv', index=False)
        print(f"\nМетрики сохранены в csv_all_metrics_holidays.csv")


if __name__ == "__main__":
    main()