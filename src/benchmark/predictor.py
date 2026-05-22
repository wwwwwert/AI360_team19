from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd

from src.benchmark.distances import (
    mass_subsequence_distance,
    resample_values,
)
from src.benchmark.metrics import vote_label
from src.benchmark.schemas import MethodName
from src.datasets.gesture_pebble_z1 import CLASS_NAMES, GestureRecord


class Predictor:
    """Nearest-neighbor predictor that ranks reference time series for a query.

    The predictor follows the familiar fit/predict shape:

    - ``fit(reference_records)`` stores the searchable collection, usually train.
    - ``rank(query_record)`` returns references sorted by similarity to one query.
    - ``predict(query_record, k)`` predicts a label by top-k neighbor voting.
    """

    def __init__(
        self,
        method: MethodName = "mass_subsequence",
        resample: bool = False,
        normalize: bool = True,
        distance_scale: Literal["sqrt_m", "none"] = "sqrt_m",
        resample_length: int | Literal["max", "median"] = "max",
    ) -> None:
        self.method = method
        self.resample = resample
        self.normalize = normalize
        self.distance_scale = distance_scale
        self.resample_length = resample_length

        self.reference_records: list[GestureRecord] = []
        self._fixed_length: int | None = None
        self._reference_cache: dict[str, np.ndarray] = {}

    def fit(
        self,
        reference_records: Iterable[GestureRecord],
        resample_context_records: Iterable[GestureRecord] | None = None,
    ) -> "Predictor":
        self.reference_records = list(reference_records)
        if not self.reference_records:
            raise ValueError("reference_records must contain at least one record")

        self._reference_cache = {}
        self._fixed_length = None
        if self.resample:
            context_records = (
                list(resample_context_records)
                if resample_context_records is not None
                else self.reference_records
            )
            self._fixed_length = self._resolve_resample_length(context_records)
            self._reference_cache = {
                record.series_id: resample_values(record.values, self._fixed_length)
                for record in self.reference_records
            }
        return self

    def rank(
        self,
        query_record: GestureRecord,
        top_k: int | None = None,
    ) -> pd.DataFrame:
        self._require_fitted()
        rows = [
            self._make_neighbor_row(query_record=query_record, reference_record=reference_record)
            for reference_record in self.reference_records
        ]
        rows.sort(key=lambda row: (row["distance"], row["reference_id"]))
        if top_k is not None:
            rows = rows[:top_k]

        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return pd.DataFrame(rows)

    def rank_many(
        self,
        query_records: Iterable[GestureRecord],
        top_k: int | None = None,
    ) -> pd.DataFrame:
        frames = [self.rank(query_record, top_k=top_k) for query_record in query_records]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def predict(
        self,
        query_record: GestureRecord,
        k: int = 1,
        keep_neighbors: bool = True,
    ) -> dict[str, object]:
        if k <= 0:
            raise ValueError("k must be positive")

        neighbors = self.rank(query_record)
        top_neighbors = neighbors.head(k)
        predicted_label = vote_label(
            top_neighbors["reference_label"].astype(str).tolist(),
            top_neighbors["distance"].astype(float).tolist(),
        )
        result: dict[str, object] = {
            "method": self.method,
            "resample": self.resample,
            "resample_length": self._fixed_length if self.resample else None,
            "k": k,
            "query_id": query_record.series_id,
            "true_label": query_record.label,
            "true_label_name": query_record.label_name,
            "predicted_label": predicted_label,
            "predicted_label_name": CLASS_NAMES.get(predicted_label, predicted_label),
            "correct": predicted_label == query_record.label,
            "nearest_reference_id": str(top_neighbors.iloc[0]["reference_id"]),
            "nearest_reference_label": str(top_neighbors.iloc[0]["reference_label"]),
            "nearest_distance": float(top_neighbors.iloc[0]["distance"]),
            "top_k_contains_true_label": query_record.label
            in set(top_neighbors["reference_label"].astype(str)),
        }
        if keep_neighbors:
            result["neighbors"] = neighbors
        return result

    def predict_many(
        self,
        query_records: Iterable[GestureRecord],
        k: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        prediction_rows = []
        neighbor_frames = []

        for query_record in query_records:
            result = self.predict(query_record, k=k, keep_neighbors=True)
            neighbors = result.pop("neighbors")
            prediction_rows.append(result)
            neighbor_frames.append(neighbors)

        predictions = pd.DataFrame(prediction_rows)
        neighbors = pd.concat(neighbor_frames, ignore_index=True) if neighbor_frames else pd.DataFrame()
        return predictions, neighbors

    def _make_neighbor_row(
        self,
        query_record: GestureRecord,
        reference_record: GestureRecord,
    ) -> dict[str, object]:
        distance_info = self._distance(query_record, reference_record)
        return {
            "method": self.method,
            "resample": self.resample,
            "resample_length": self._fixed_length if self.resample else None,
            "query_id": query_record.series_id,
            "query_label": query_record.label,
            "query_label_name": query_record.label_name,
            "reference_id": reference_record.series_id,
            "reference_label": reference_record.label,
            "reference_label_name": reference_record.label_name,
            **distance_info,
        }

    def _distance(
        self,
        query_record: GestureRecord,
        reference_record: GestureRecord,
    ) -> dict[str, object]:
        if self.method == "mass_subsequence":
            return mass_subsequence_distance(
                self._query_values(query_record),
                self._reference_values(reference_record),
                normalize=self.normalize,
                distance_scale=self.distance_scale,
            )

        raise ValueError(f"Unknown method: {self.method}")

    def _query_values(self, query_record: GestureRecord) -> np.ndarray:
        if not self.resample:
            return query_record.values
        if self._fixed_length is None:
            raise RuntimeError("Resample length is not initialized. Call fit() first.")
        return resample_values(query_record.values, self._fixed_length)

    def _reference_values(self, reference_record: GestureRecord) -> np.ndarray:
        if not self.resample:
            return reference_record.values
        return self._reference_cache[reference_record.series_id]

    def _resolve_resample_length(self, records: list[GestureRecord]) -> int:
        lengths = np.array([record.length for record in records], dtype=int)
        if self.resample_length == "max":
            return int(lengths.max())
        if self.resample_length == "median":
            return int(round(float(np.median(lengths))))
        length = int(self.resample_length)
        if length <= 1:
            raise ValueError("resample_length must be greater than 1")
        return length

    def _require_fitted(self) -> None:
        if not self.reference_records:
            raise RuntimeError("Call fit(reference_records) before ranking or predicting")
