import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Путь к данным
DATA_DIR = Path("data/yahoo")

# Детекция
from anomaly_detection import DEFAULT_CONFIGURATION, AnomalyDetectionSystem


def load_yahoo_dataset(data_path: Path):
    """Загружает один датасет Yahoo"""
    test_values = np.load(data_path / "test.npy")
    test_labels = np.load(data_path / "test_label.npy")
    test_timestamps = np.load(data_path / "test_timestamp.npy")

    # Конвертируем timestamp в datetime
    if test_timestamps.max() > 1e12:  # миллисекунды
        test_timestamps = test_timestamps / 1000

    dates = [datetime.fromtimestamp(ts) for ts in test_timestamps]

    time_series = pd.DataFrame({
        'value_0': test_values
    }, index=pd.DatetimeIndex(dates))

    return time_series, test_labels.astype(int)


def calculate_metrics(true_labels, predictions):
    """Рассчитывает метрики для одного датасета"""
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'total_points': len(true_labels),
        'true_anomalies': int(np.sum(true_labels)),
        'predicted_anomalies': int(np.sum(predictions)),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def main():
    # Находим все датасеты
    datasets = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

    if not datasets:
        print(f"Датасеты не найдены в {DATA_DIR}")
        return

    print(f"Найдено {len(datasets)} датасетов\n")

    # Инициализируем систему детекции
    system = AnomalyDetectionSystem(**DEFAULT_CONFIGURATION)

    all_metrics = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    print("=" * 80)
    print(f"{'Датасет':<30} {'Точек':<8} {'Аномалий':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("=" * 80)

    for dataset_path in datasets:
        try:
            # Загружаем данные
            time_series, true_labels = load_yahoo_dataset(dataset_path)

            # Запускаем детекцию
            detection_result = system.detect(time_series=time_series)
            predictions = detection_result.is_anomaly.astype(int)

            # Считаем метрики
            metrics = calculate_metrics(true_labels, predictions)
            metrics['dataset'] = dataset_path.name
            all_metrics.append(metrics)

            # Суммируем для общих метрик
            total_tp += metrics['tp']
            total_fp += metrics['fp']
            total_fn += metrics['fn']

            # Выводим результат для этого датасета
            print(f"{dataset_path.name:<30} "
                  f"{metrics['total_points']:<8} "
                  f"{metrics['true_anomalies']:<10} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1_score']:<10.4f}")

        except Exception as e:
            print(f"{dataset_path.name:<30} ОШИБКА: {str(e)[:50]}")

    print("=" * 80)

    # Общие метрики по всем датасетам
    if total_tp + total_fp > 0:
        overall_precision = total_tp / (total_tp + total_fp)
    else:
        overall_precision = 0

    if total_tp + total_fn > 0:
        overall_recall = total_tp / (total_tp + total_fn)
    else:
        overall_recall = 0

    overall_f1 = (2 * overall_precision * overall_recall /
                  (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0)

    print(f"\n{'ОБЩИЕ РЕЗУЛЬТАТЫ':<30}")
    print(f"{'Всего датасетов:':<30} {len(all_metrics)}")
    print(f"{'Total TP:':<30} {total_tp}")
    print(f"{'Total FP:':<30} {total_fp}")
    print(f"{'Total FN:':<30} {total_fn}")
    print(f"{'Overall Precision:':<30} {overall_precision:.4f}")
    print(f"{'Overall Recall:':<30} {overall_recall:.4f}")
    print(f"{'Overall F1-score:':<30} {overall_f1:.4f}")

    # Сохраняем детальные результаты в CSV
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv('yahoo_all_metrics.csv', index=False)
    print(f"\nДетальные метрики сохранены в yahoo_all_metrics.csv")

    # # Визуализация (закомментирована)
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    #
    # metrics_names = ['precision', 'recall', 'f1_score']
    # titles = ['Precision', 'Recall', 'F1-score']
    #
    # for ax, metric, title in zip(axes, metrics_names, titles):
    #     values = [m[metric] for m in all_metrics]
    #     datasets_names = [m['dataset'] for m in all_metrics]
    #     ax.bar(range(len(values)), values)
    #     ax.set_xticks(range(len(values)))
    #     ax.set_xticklabels(datasets_names, rotation=45, ha='right')
    #     ax.set_title(title)
    #     ax.set_ylim(0, 1)
    #     ax.axhline(y=np.mean(values), color='r', linestyle='--', alpha=0.5)
    #
    # plt.tight_layout()
    # plt.savefig('yahoo_all_metrics.png', dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    main()