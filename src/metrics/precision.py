from __future__ import annotations

import numpy as np

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import MISSING_LABEL, safe_divide, topk_relevant_counts


class Precision(BaseMetric):
    metric_name = "Precision"

    def _compute(
        self,
        target: np.ndarray,
        predicted: np.ndarray,
        labels: list[str],
        relevant_counts: np.ndarray,
    ) -> float:
        per_class = self._per_class(target, predicted, labels, relevant_counts)
        if self.average == "macro":
            return float(np.mean(list(per_class.values()))) if per_class else 0.0

        true_positive = float(np.sum(topk_relevant_counts(target, predicted)))
        predicted_count = float(np.sum(predicted != MISSING_LABEL))
        return safe_divide(true_positive, predicted_count)

    def _per_class(
        self,
        target: np.ndarray,
        predicted: np.ndarray,
        labels: list[str],
        relevant_counts: np.ndarray,
    ) -> dict[str, float]:
        relevant_in_topk = topk_relevant_counts(target, predicted)
        retrieved = np.sum(predicted != MISSING_LABEL, axis=1).astype(float)
        per_query_precision = np.array(
            [safe_divide(tp, count) for tp, count in zip(relevant_in_topk, retrieved)],
            dtype=float,
        )
        return {
            label: float(np.mean(per_query_precision[target == label]))
            if np.any(target == label)
            else 0.0
            for label in labels
        }
