from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd

from src.benchmark.metrics import classification_metrics, confusion_matrix
from src.benchmark.predictor import Predictor
from src.benchmark.schemas import BenchmarkResult, MethodName
from src.datasets.gesture_pebble_z1 import GesturePebbleZ1Dataset
from src.metrics import DEFAULT_AVERAGES, DEFAULT_METRIC_NAMES
from src.metrics.utils import MISSING_LABEL


class GesturePebbleZ1Benchmark:
    """Evaluate a Predictor on the GesturePebbleZ1 train/test split."""

    def __init__(
        self,
        dataset: GesturePebbleZ1Dataset,
        normalize: bool = True,
        resample: bool = False,
        distance_scale: Literal["sqrt_m", "none"] = "sqrt_m",
        resample_length: int | Literal["max", "median"] = "max",
        metric_names: Iterable[str] = DEFAULT_METRIC_NAMES,
        metric_averages: Iterable[str] = DEFAULT_AVERAGES,
    ) -> None:
        self.dataset = dataset
        self.normalize = normalize
        self.resample = resample
        self.distance_scale = distance_scale
        self.resample_length = resample_length
        self.metric_names = tuple(metric_names)
        self.metric_averages = tuple(metric_averages)

    def run(
        self,
        methods: Iterable[MethodName] = ("mass_subsequence",),
        k_values: Iterable[int] = (1, 5, 10),
        max_test: int | None = None,
    ) -> dict[str, BenchmarkResult]:
        results = {}
        for method in methods:
            for k in k_values:
                result = self.evaluate(method=method, k=k, max_test=max_test)
                results[f"{self._run_name(method)}@{k}"] = result
        return results

    def evaluate(
        self,
        method: MethodName = "mass_subsequence",
        k: int = 1,
        max_test: int | None = None,
    ) -> BenchmarkResult:
        if k <= 0:
            raise ValueError("k must be positive")

        test_records = self.dataset.test[:max_test] if max_test else self.dataset.test
        predictor = self._build_predictor(method)
        predictor.fit(
            reference_records=self.dataset.train,
            resample_context_records=self.dataset.records,
        )

        predictions_df, neighbors_df = predictor.predict_many(test_records, k=k)
        predictions_df = self._to_legacy_prediction_columns(predictions_df)
        neighbors_df = self._to_legacy_neighbor_columns(neighbors_df, k=k)
        metric_predicted = self._make_topk_label_matrix(
            predictions_df=predictions_df,
            neighbors_df=neighbors_df,
            k=k,
        )
        relevant_counts = self._make_relevant_counts(predictions_df)
        metrics_df = classification_metrics(
            target=predictions_df["true_label"].to_numpy(dtype=str),
            predicted=metric_predicted,
            k=k,
            metric_names=self.metric_names,
            averages=self.metric_averages,
            relevant_counts=relevant_counts,
        )
        confusion = confusion_matrix(
            predictions_df["true_label"].to_numpy(dtype=str),
            predictions_df["predicted_label"].to_numpy(dtype=str),
        )

        return BenchmarkResult(
            method=method,
            resample=self.resample,
            k=k,
            predictions=predictions_df,
            neighbors=neighbors_df,
            metrics=metrics_df,
            confusion_matrix=confusion,
        )

    def build_predictor(self, method: MethodName = "mass_subsequence") -> Predictor:
        """Return a fitted predictor that ranks train records for a query record."""
        predictor = self._build_predictor(method)
        return predictor.fit(
            reference_records=self.dataset.train,
            resample_context_records=self.dataset.records,
        )

    def _build_predictor(self, method: MethodName) -> Predictor:
        return Predictor(
            method=method,
            resample=self.resample,
            normalize=self.normalize,
            distance_scale=self.distance_scale,
            resample_length=self.resample_length,
        )

    def _run_name(self, method: MethodName) -> str:
        return f"{method}_resample" if self.resample else method

    @staticmethod
    def _to_legacy_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
        return predictions.rename(
            columns={
                "query_id": "test_id",
                "nearest_reference_id": "nearest_train_id",
                "nearest_reference_label": "nearest_train_label",
            }
        )

    @staticmethod
    def _to_legacy_neighbor_columns(neighbors: pd.DataFrame, k: int) -> pd.DataFrame:
        neighbors = neighbors.rename(
            columns={
                "query_id": "test_id",
                "query_label": "true_label",
                "query_label_name": "true_label_name",
                "reference_id": "train_id",
                "reference_label": "train_label",
                "reference_label_name": "train_label_name",
            }
        )
        neighbors.insert(1, "k", k)
        return neighbors

    @staticmethod
    def _make_topk_label_matrix(
        predictions_df: pd.DataFrame,
        neighbors_df: pd.DataFrame,
        k: int,
    ) -> np.ndarray:
        labels_by_test_id = {}
        for test_id, group in neighbors_df.sort_values("rank").groupby("test_id", sort=False):
            labels = group.head(k)["train_label"].astype(str).tolist()
            labels = labels + [MISSING_LABEL] * (k - len(labels))
            labels_by_test_id[str(test_id)] = labels

        return np.array(
            [
                labels_by_test_id.get(str(test_id), [MISSING_LABEL] * k)
                for test_id in predictions_df["test_id"].astype(str)
            ],
            dtype=str,
        )

    def _make_relevant_counts(self, predictions_df: pd.DataFrame) -> np.ndarray:
        train_label_counts = pd.Series(
            [record.label for record in self.dataset.train],
            dtype=str,
        ).value_counts()
        return (
            predictions_df["true_label"]
            .astype(str)
            .map(train_label_counts)
            .fillna(0)
            .to_numpy(dtype=float)
        )
