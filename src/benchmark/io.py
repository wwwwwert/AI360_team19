from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.benchmark.schemas import BenchmarkResult


def write_benchmark_outputs(
    results: dict[str, BenchmarkResult],
    output_dir: str | Path,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    summary_rows = []
    for key, result in results.items():
        stem = key.replace("@", "_k")
        prediction_path = output / f"{stem}_predictions.csv"
        neighbors_path = output / f"{stem}_neighbors.csv"
        metrics_path = output / f"{stem}_metrics.csv"
        confusion_path = output / f"{stem}_confusion_matrix.csv"

        result.predictions.to_csv(prediction_path, index=False)
        result.neighbors.to_csv(neighbors_path, index=False)
        result.metrics.to_csv(metrics_path, index=False)
        result.confusion_matrix.to_csv(confusion_path)

        row = {
            "run": key,
            "method": result.method,
            "resample": result.resample,
            "k": result.k,
        }
        overall = result.metrics[result.metrics["label"] == "overall"]
        for metric_row in overall.itertuples(index=False):
            column_name = f"{metric_row.metric_name}_{metric_row.average}"
            row[column_name] = float(metric_row.value)
        summary_rows.append(row)

        written[f"{key}:predictions"] = str(prediction_path)
        written[f"{key}:neighbors"] = str(neighbors_path)
        written[f"{key}:metrics"] = str(metrics_path)
        written[f"{key}:confusion"] = str(confusion_path)

    summary_path = output / "benchmark_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    written["summary"] = str(summary_path)
    return written
