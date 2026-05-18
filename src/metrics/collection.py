from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.metrics.base_metric import BaseMetric
from src.metrics.f1 import F1
from src.metrics.hitrate import Hitrate
from src.metrics.precision import Precision
from src.metrics.recall import Recall
from src.metrics.utils import collect_labels


METRIC_REGISTRY: dict[str, type[BaseMetric]] = {
    "hitrate": Hitrate,
    "precision": Precision,
    "recall": Recall,
    "f1": F1,
}
DEFAULT_METRIC_NAMES = ("hitrate", "precision", "recall", "f1")
DEFAULT_AVERAGES = ("micro", "macro")


class ClassificationMetricCollection:
    """Compute a set of stateless offline classification metrics."""

    def __init__(
        self,
        metrics: list[BaseMetric] | None = None,
        include_per_class: bool = True,
    ) -> None:
        self.metrics = metrics if metrics is not None else default_metrics()
        self.include_per_class = include_per_class

    def compute(
        self,
        *,
        target,
        predicted,
        labels: list[str] | None = None,
        relevant_counts=None,
    ) -> pd.DataFrame:
        if labels is None:
            labels = collect_labels(target)

        rows = []
        seen_per_class_metrics = set()
        for metric in self.metrics:
            rows.append(
                metric.to_record(
                    target=target,
                    predicted=predicted,
                    labels=labels,
                    relevant_counts=relevant_counts,
                )
            )
            if self.include_per_class and metric.name not in seen_per_class_metrics:
                rows.extend(
                    metric.per_class_records(
                        target=target,
                        predicted=predicted,
                        labels=labels,
                        relevant_counts=relevant_counts,
                    )
                )
                seen_per_class_metrics.add(metric.name)
        return pd.DataFrame(rows)


def build_metric(metric_name: str, k: int = 1, average: str = "macro") -> BaseMetric:
    key = metric_name.lower()
    if key not in METRIC_REGISTRY:
        known = ", ".join(sorted(METRIC_REGISTRY))
        raise ValueError(f"Unknown metric '{metric_name}'. Known metrics: {known}")
    return METRIC_REGISTRY[key](k=k, average=average)


def build_metrics(
    metric_names: Iterable[str] = DEFAULT_METRIC_NAMES,
    k: int = 1,
    averages: Iterable[str] = DEFAULT_AVERAGES,
) -> list[BaseMetric]:
    return [
        build_metric(metric_name=metric_name, k=k, average=average)
        for metric_name in metric_names
        for average in averages
    ]


def default_metrics(k: int = 1) -> list[BaseMetric]:
    return build_metrics(k=k)


def classification_metrics(
    target,
    predicted,
    k: int = 1,
    labels: list[str] | None = None,
    include_per_class: bool = True,
    metric_names: Iterable[str] = DEFAULT_METRIC_NAMES,
    averages: Iterable[str] = DEFAULT_AVERAGES,
    relevant_counts=None,
) -> pd.DataFrame:
    return ClassificationMetricCollection(
        metrics=build_metrics(metric_names=metric_names, k=k, averages=averages),
        include_per_class=include_per_class,
    ).compute(
        target=target,
        predicted=predicted,
        labels=labels,
        relevant_counts=relevant_counts,
    )
