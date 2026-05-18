from __future__ import annotations

import numpy as np

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import safe_divide, topk_relevant_counts


class Recall(BaseMetric):
    metric_name = "Recall"

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
        relevant_in_topk = np.minimum(topk_relevant_counts(target, predicted), relevant_counts)
        true_positive = float(np.sum(relevant_in_topk))
        total_relevant = float(np.sum(relevant_counts))
        return safe_divide(true_positive, total_relevant)

    def _per_class(
        self,
        target: np.ndarray,
        predicted: np.ndarray,
        labels: list[str],
        relevant_counts: np.ndarray,
    ) -> dict[str, float]:
        relevant_in_topk = np.minimum(topk_relevant_counts(target, predicted), relevant_counts)
        return {
            label: safe_divide(
                float(np.sum(relevant_in_topk[target == label])),
                float(np.sum(relevant_counts[target == label])),
            )
            for label in labels
        }
