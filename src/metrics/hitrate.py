from __future__ import annotations

import numpy as np

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import safe_divide, topk_contains_target


class Hitrate(BaseMetric):
    """Share of samples where the true label appears in the top-k predictions."""

    metric_name = "Hitrate"

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
        return float(topk_contains_target(target, predicted).mean()) if len(target) else 0.0

    def _per_class(
        self,
        target: np.ndarray,
        predicted: np.ndarray,
        labels: list[str],
        relevant_counts: np.ndarray,
    ) -> dict[str, float]:
        hits = topk_contains_target(target, predicted)
        return {
            label: safe_divide(float(np.sum(hits[target == label])), float(np.sum(target == label)))
            for label in labels
        }
