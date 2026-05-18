from __future__ import annotations

import numpy as np

from src.metrics.base_metric import BaseMetric
from src.metrics.precision import Precision
from src.metrics.recall import Recall
from src.metrics.utils import safe_divide


class F1(BaseMetric):
    metric_name = "F1"

    def _compute(
        self,
        target: np.ndarray,
        predicted: np.ndarray,
        labels: list[str],
        relevant_counts: np.ndarray,
    ) -> float:
        if self.average == "micro":
            precision = Precision(k=self.k, average="micro")._compute(
                target,
                predicted,
                labels,
                relevant_counts,
            )
            recall = Recall(k=self.k, average="micro")._compute(
                target,
                predicted,
                labels,
                relevant_counts,
            )
            return safe_divide(2 * precision * recall, precision + recall)

        per_class = self._per_class(target, predicted, labels, relevant_counts)
        return float(np.mean(list(per_class.values()))) if per_class else 0.0

    def _per_class(
        self,
        target: np.ndarray,
        predicted: np.ndarray,
        labels: list[str],
        relevant_counts: np.ndarray,
    ) -> dict[str, float]:
        precision = Precision(k=self.k, average="macro")._per_class(
            target,
            predicted,
            labels,
            relevant_counts,
        )
        recall = Recall(k=self.k, average="macro")._per_class(
            target,
            predicted,
            labels,
            relevant_counts,
        )
        return {
            label: safe_divide(2 * precision[label] * recall[label], precision[label] + recall[label])
            for label in labels
        }
