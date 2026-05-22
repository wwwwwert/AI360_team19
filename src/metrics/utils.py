from __future__ import annotations

import numpy as np
import pandas as pd


MISSING_LABEL = "__missing_label__"


def vote_label(labels: list[str], distances: list[float]) -> str:
    frame = pd.DataFrame({"label": labels, "distance": distances})
    votes = frame.groupby("label").agg(count=("label", "size"), mean_distance=("distance", "mean"))
    votes = votes.sort_values(["count", "mean_distance"], ascending=[False, True])
    return str(votes.index[0])


def prepare_classification_inputs(
    *,
    target,
    predicted,
    k: int,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    target_arr = np.asarray(target, dtype=str).reshape(-1)
    predicted_arr = np.asarray(predicted, dtype=str)

    if predicted_arr.ndim == 1:
        predicted_arr = predicted_arr.reshape(-1, 1)
    if predicted_arr.ndim != 2:
        raise ValueError("predicted must be a 1D or 2D array-like object")
    if len(target_arr) != predicted_arr.shape[0]:
        raise ValueError("target and predicted must contain the same number of rows")
    if predicted_arr.shape[1] < k:
        raise ValueError(f"predicted must contain at least k={k} columns")

    predicted_arr = predicted_arr[:, :k]
    if labels is None:
        labels = collect_labels(target_arr)
    else:
        labels = [str(label) for label in labels]
    return target_arr, predicted_arr, labels


def prepare_relevant_counts(relevant_counts, n_rows: int) -> np.ndarray:
    if relevant_counts is None:
        return np.ones(n_rows, dtype=float)

    counts = np.asarray(relevant_counts, dtype=float).reshape(-1)
    if len(counts) != n_rows:
        raise ValueError("relevant_counts must contain the same number of rows as target")
    if not np.all(np.isfinite(counts)):
        raise ValueError("relevant_counts must contain only finite values")
    if np.any(counts < 0):
        raise ValueError("relevant_counts must be non-negative")
    return counts


def collect_labels(target: np.ndarray, predicted: np.ndarray | None = None) -> list[str]:
    values = set(np.asarray(target, dtype=str).reshape(-1).tolist())
    if predicted is not None:
        for value in np.asarray(predicted, dtype=str).reshape(-1).tolist():
            if value != MISSING_LABEL:
                values.add(value)
    return sorted(values, key=label_sort_key)


def label_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def topk_contains_target(target: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.array(
        [target_value in set(row[row != MISSING_LABEL]) for target_value, row in zip(target, predicted)],
        dtype=bool,
    )


def topk_relevant_counts(target: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.array(
        [int(np.sum(row[row != MISSING_LABEL] == target_value)) for target_value, row in zip(target, predicted)],
        dtype=float,
    )


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = collect_labels(np.asarray(y_true, dtype=str), np.asarray(y_pred, dtype=str))
    matrix = pd.crosstab(
        pd.Series(np.asarray(y_true, dtype=str), name="true_label"),
        pd.Series(np.asarray(y_pred, dtype=str), name="predicted_label"),
        dropna=False,
    )
    return matrix.reindex(index=labels, columns=labels, fill_value=0)
