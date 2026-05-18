from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from src.metrics.utils import prepare_classification_inputs, prepare_relevant_counts


Average = Literal["micro", "macro"]


class BaseMetric(ABC):
    """Base class for offline top-k retrieval/classification metrics."""

    metric_name = "Metric"

    def __init__(
        self,
        k: int = 1,
        average: Average = "macro",
        name: str | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        if average not in {"micro", "macro"}:
            raise ValueError("average must be either 'micro' or 'macro'")

        self.k = int(k)
        self.average = average
        self.name = name if name is not None else f"{self.metric_name}@{self.k}"

    def __call__(
        self,
        *,
        target,
        predicted,
        labels: list[str] | None = None,
        relevant_counts=None,
    ) -> float:
        return self.compute(
            target=target,
            predicted=predicted,
            labels=labels,
            relevant_counts=relevant_counts,
        )

    def compute(
        self,
        *,
        target,
        predicted,
        labels: list[str] | None = None,
        relevant_counts=None,
    ) -> float:
        target_arr, predicted_arr, labels = prepare_classification_inputs(
            target=target,
            predicted=predicted,
            k=self.k,
            labels=labels,
        )
        relevant_counts_arr = prepare_relevant_counts(relevant_counts, len(target_arr))
        return self._compute(target_arr, predicted_arr, labels, relevant_counts_arr)

    def per_class(
        self,
        *,
        target,
        predicted,
        labels: list[str] | None = None,
        relevant_counts=None,
    ) -> dict[str, float]:
        target_arr, predicted_arr, labels = prepare_classification_inputs(
            target=target,
            predicted=predicted,
            k=self.k,
            labels=labels,
        )
        relevant_counts_arr = prepare_relevant_counts(relevant_counts, len(target_arr))
        return self._per_class(target_arr, predicted_arr, labels, relevant_counts_arr)

    def to_record(
        self,
        *,
        target,
        predicted,
        labels: list[str] | None = None,
        relevant_counts=None,
    ) -> dict[str, object]:
        target_arr, predicted_arr, labels = prepare_classification_inputs(
            target=target,
            predicted=predicted,
            k=self.k,
            labels=labels,
        )
        relevant_counts_arr = prepare_relevant_counts(relevant_counts, len(target_arr))
        return {
            "metric": self.name,
            "metric_name": self.metric_name,
            "k": self.k,
            "average": self.average,
            "label": "overall",
            "value": self._compute(target_arr, predicted_arr, labels, relevant_counts_arr),
            "support": int(len(target_arr)),
        }

    def per_class_records(
        self,
        *,
        target,
        predicted,
        labels: list[str] | None = None,
        relevant_counts=None,
    ) -> list[dict[str, object]]:
        target_arr, predicted_arr, labels = prepare_classification_inputs(
            target=target,
            predicted=predicted,
            k=self.k,
            labels=labels,
        )
        relevant_counts_arr = prepare_relevant_counts(relevant_counts, len(target_arr))
        values = self._per_class(target_arr, predicted_arr, labels, relevant_counts_arr)
        return [
            {
                "metric": self.name,
                "metric_name": self.metric_name,
                "k": self.k,
                "average": "per_class",
                "label": label,
                "value": value,
                "support": int(np.sum(target_arr == label)),
            }
            for label, value in values.items()
        ]

    @abstractmethod
    def _compute(
        self,
        target: np.ndarray,
        predicted: np.ndarray,
        labels: list[str],
        relevant_counts: np.ndarray,
    ) -> float:
        pass

    @abstractmethod
    def _per_class(
        self,
        target: np.ndarray,
        predicted: np.ndarray,
        labels: list[str],
        relevant_counts: np.ndarray,
    ) -> dict[str, float]:
        pass
