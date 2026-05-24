import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from anomaly_detection import DEFAULT_CONFIGURATION, AnomalyDetectionSystem


def calculate_metrics(true_labels, predictions):
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1, tp, fp, fn


def sweep_beta(
    datasets: list[tuple[pd.DataFrame, np.ndarray]],
    threshold: float = 3.49,
    beta_min: float = 1.0,
    beta_max: float = 5.0,
    n_points: int = 250,
    title: str = "F1 vs Holiday β",
    plot_path: str = "beta_sweep.png",
):
    CONFIG = DEFAULT_CONFIGURATION.copy()
    CONFIG["detection_model_params"]["threshold"] = threshold
    CONFIG["detection_model_params"]["apply_holidays"] = True
    system = AnomalyDetectionSystem(**CONFIG)

    betas = np.linspace(beta_min, beta_max, n_points)
    precisions, recalls, f1s = [], [], []

    for i, beta in enumerate(betas):
        system.holiday_param = np.float64(beta)
        all_true, all_pred = [], []
        for time_series, true_labels in datasets:
            result = system.detect(time_series=time_series, dates=time_series.index)
            all_pred.append(result.is_anomaly.astype(int))
            all_true.append(true_labels)
        precision, recall, f1, *_ = calculate_metrics(
            np.concatenate(all_true), np.concatenate(all_pred)
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        print(f"  {i + 1:>3}/{n_points}  β={beta:.3f}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

    best_idx = int(np.argmax(np.round(f1s, 3)))
    best_beta = betas[best_idx]
    best_f1 = f1s[best_idx]
    print(f"\nЛучший F1={best_f1:.4f} при β={best_beta:.4f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(betas, precisions, label="Precision")
    ax1.plot(betas, recalls, label="Recall")
    ax1.plot(betas, f1s, label="F1", linewidth=2)
    ax1.axvline(best_beta, color="tab:red", linestyle="--", linewidth=1, label=f"Best β={best_beta:.3f}")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    ax2.plot(betas, f1s, color="tab:green", linewidth=2)
    ax2.axvline(best_beta, color="tab:red", linestyle="--", linewidth=1)
    ax2.scatter([best_beta], [best_f1], color="tab:red", zorder=5)
    ax2.set_ylabel("F1")
    ax2.set_xlabel("Holiday β")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"График сохранён: {plot_path}")

    return betas, f1s, best_beta, best_f1
