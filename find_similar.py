from __future__ import annotations

from functools import reduce
from math import ceil, gcd
from typing import Any, Iterable

import numpy as np
import pandas as pd
import stumpy


RESULT_COLUMNS = [
    "series_id",
    "start_idx",
    "end_idx",
    "start_time",
    "end_time",
    "distance",
]


class TimeSeriesSubsequenceSearcher:
    """Find similar fixed-duration subsequences across resampled time series.

    The class prepares every input series on one regular timestamp grid and then
    uses STUMPY's MASS distance profile to compare a query with all possible
    windows of the same length in every target series.
    """

    def __init__(
        self,
        freq: str | pd.Timedelta | None = "5min",
        agg: str = "mean",
        interpolate_limit: int | None = 3,
        normalize: bool = True,
        exclusion_fraction: float = 0.5,
    ) -> None:
        if interpolate_limit is not None and interpolate_limit < 0:
            raise ValueError("interpolate_limit must be None or a non-negative integer")
        if exclusion_fraction < 0:
            raise ValueError("exclusion_fraction must be non-negative")

        self.freq = freq
        self.agg = agg
        self.interpolate_limit = interpolate_limit
        self.normalize = normalize
        self.exclusion_fraction = float(exclusion_fraction)

        self.freq_: str | pd.Timedelta | None = None
        self.series_: dict[str, pd.Series] = {}
        self.prepared_series_: dict[str, pd.Series] = self.series_

    def fit(self, series_dict: dict[str, pd.DataFrame]) -> "TimeSeriesSubsequenceSearcher":
        """Prepare and resample all input series.

        Parameters
        ----------
        series_dict:
            Mapping from a user-defined series id to a dataframe with a
            ``timestamp`` column and either ``value`` or ``value_0``.
        """
        if not series_dict:
            raise ValueError("series_dict must contain at least one time series")

        raw_series = {
            str(series_id): self._coerce_input_frame(df, str(series_id))
            for series_id, df in series_dict.items()
        }
        self.freq_ = self.freq if self.freq is not None else self._infer_common_freq(raw_series)
        self.series_ = {
            series_id: self._resample_series(series)
            for series_id, series in raw_series.items()
        }
        self.prepared_series_ = self.series_
        return self

    def search_by_query_array(
        self,
        query: np.ndarray | pd.Series | list[float],
        top_k: int = 20,
        exclude_series_id: str | Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Search for the global top-k matches for an already prepared query array."""
        query_values = self._coerce_query_array(query)
        exclude_series_ids = self._normalize_excluded_ids(exclude_series_id)
        return self._search(query_values, top_k=top_k, exclude_series_ids=exclude_series_ids)

    def search_by_time_range(
        self,
        source_series_id: str,
        start_time: Any,
        end_time: Any,
        top_k: int = 20,
        search_in_source: bool = False,
    ) -> pd.DataFrame:
        """Cut a query from one prepared series by time and search in other series."""
        self._require_fitted()
        source_series_id = str(source_series_id)
        if source_series_id not in self.series_:
            raise KeyError(f"Unknown source_series_id: {source_series_id!r}")

        start = self._parse_timestamp(start_time)
        end = self._parse_timestamp(end_time)
        if end < start:
            raise ValueError("end_time must be greater than or equal to start_time")

        query = self.series_[source_series_id].loc[start:end]
        if query.empty:
            raise ValueError(
                f"No samples found in {source_series_id!r} between {start} and {end}"
            )

        exclude = None if search_in_source else source_series_id
        return self.search_by_query_array(query.to_numpy(), top_k=top_k, exclude_series_id=exclude)

    def get_segment(self, series_id: str, start_idx: int, end_idx: int) -> pd.Series:
        """Return a previously found segment as a timestamp-indexed pandas Series."""
        self._require_fitted()
        series_id = str(series_id)
        if series_id not in self.series_:
            raise KeyError(f"Unknown series_id: {series_id!r}")
        if start_idx < 0 or end_idx < start_idx or end_idx >= len(self.series_[series_id]):
            raise IndexError("Invalid start_idx/end_idx for the requested series")
        return self.series_[series_id].iloc[start_idx : end_idx + 1].copy()

    def match_one_series(
        self,
        query: np.ndarray | pd.Series | list[float],
        series_id: str,
        max_matches: int | None = None,
    ) -> pd.DataFrame:
        """Convenience wrapper around ``stumpy.match`` for one target series.

        The main global search uses ``stumpy.mass`` because it needs the full
        distance profile before applying cross-series ranking and suppression.
        This method is handy when you want STUMPY's own per-series matching
        behavior for inspection.
        """
        self._require_fitted()
        query_values = self._coerce_query_array(query)
        series_id = str(series_id)
        if series_id not in self.series_:
            raise KeyError(f"Unknown series_id: {series_id!r}")

        target_values = self.series_[series_id].to_numpy(dtype=float)
        if len(target_values) < len(query_values):
            return self._empty_result()

        matches = stumpy.match(
            query_values,
            target_values,
            max_matches=max_matches,
            normalize=self.normalize,
        )
        rows = []
        for distance, start_idx in np.asarray(matches):
            start_idx = int(start_idx)
            end_idx = start_idx + len(query_values) - 1
            rows.append(self._make_result_row(series_id, start_idx, end_idx, float(distance)))
        return pd.DataFrame(rows, columns=RESULT_COLUMNS)

    def _search(
        self,
        query_values: np.ndarray,
        top_k: int,
        exclude_series_ids: set[str],
    ) -> pd.DataFrame:
        self._require_fitted()
        top_k = int(top_k)
        if top_k <= 0:
            return self._empty_result()

        m = len(query_values)
        candidates: list[tuple[float, str, int]] = []

        for series_id, target in self.series_.items():
            if series_id in exclude_series_ids or len(target) < m:
                continue

            target_values = target.to_numpy(dtype=float)
            if np.isfinite(target_values).sum() < m:
                continue

            distance_profile = stumpy.mass(query_values, target_values, normalize=self.normalize)
            distance_profile = np.asarray(distance_profile, dtype=float)
            finite_idx = np.flatnonzero(np.isfinite(distance_profile))
            candidates.extend(
                (float(distance_profile[idx]), series_id, int(idx))
                for idx in finite_idx
            )

        if not candidates:
            return self._empty_result()

        candidates.sort(key=lambda item: item[0])
        return self._select_top_k_with_suppression(candidates, m=m, top_k=top_k)

    def _select_top_k_with_suppression(
        self,
        candidates: list[tuple[float, str, int]],
        m: int,
        top_k: int,
    ) -> pd.DataFrame:
        radius = int(ceil(m * self.exclusion_fraction))
        suppressed = {
            series_id: np.zeros(max(len(series) - m + 1, 0), dtype=bool)
            for series_id, series in self.series_.items()
        }
        rows = []

        for distance, series_id, start_idx in candidates:
            if suppressed[series_id][start_idx]:
                continue

            end_idx = start_idx + m - 1
            rows.append(self._make_result_row(series_id, start_idx, end_idx, distance))

            if radius > 0:
                left = max(0, start_idx - radius)
                right = min(len(suppressed[series_id]), start_idx + radius + 1)
                suppressed[series_id][left:right] = True

            if len(rows) >= top_k:
                break

        return pd.DataFrame(rows, columns=RESULT_COLUMNS)

    def _make_result_row(
        self,
        series_id: str,
        start_idx: int,
        end_idx: int,
        distance: float,
    ) -> dict[str, Any]:
        target = self.series_[series_id]
        return {
            "series_id": series_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time": target.index[start_idx],
            "end_time": target.index[end_idx],
            "distance": distance,
        }

    def _resample_series(self, series: pd.Series) -> pd.Series:
        if self.freq_ is None:
            raise RuntimeError("Frequency is not initialized. Call fit() first.")

        prepared = series.resample(self.freq_).agg(self.agg).astype(float)

        if self.interpolate_limit == 0:
            return prepared

        interpolate_kwargs: dict[str, Any] = {
            "method": "time",
            "limit_direction": "both",
            "limit_area": "inside",
        }
        if self.interpolate_limit is not None:
            interpolate_kwargs["limit"] = self.interpolate_limit

        return prepared.interpolate(**interpolate_kwargs)

    def _coerce_input_frame(self, df: pd.DataFrame, series_id: str) -> pd.Series:
        if "timestamp" not in df.columns:
            raise ValueError(f"{series_id!r} must contain a 'timestamp' column")

        value_col = self._find_value_column(df)
        index = self._to_datetime_index(df["timestamp"])
        values = pd.to_numeric(df[value_col], errors="coerce")

        series = pd.Series(values.to_numpy(dtype=float), index=index, name=series_id)
        series = series[~series.index.isna()].sort_index()
        if series.empty:
            raise ValueError(f"{series_id!r} does not contain valid timestamps")

        return series.groupby(level=0).agg(self.agg).sort_index()

    @staticmethod
    def _find_value_column(df: pd.DataFrame) -> str:
        if "value" in df.columns:
            return "value"
        if "value_0" in df.columns:
            return "value_0"
        raise ValueError("Each dataframe must contain either 'value' or 'value_0'")

    @staticmethod
    def _to_datetime_index(values: pd.Series) -> pd.DatetimeIndex:
        values = pd.Series(values)
        if pd.api.types.is_numeric_dtype(values):
            unit = TimeSeriesSubsequenceSearcher._infer_epoch_unit(values)
            parsed = pd.to_datetime(values, unit=unit, errors="coerce", utc=True)
        else:
            parsed = pd.to_datetime(values, errors="coerce", utc=True)

        if parsed.isna().any():
            bad_count = int(parsed.isna().sum())
            raise ValueError(f"Could not parse {bad_count} timestamp values")

        return pd.DatetimeIndex(parsed.dt.tz_convert(None))

    @staticmethod
    def _parse_timestamp(value: Any) -> pd.Timestamp:
        if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
            unit = TimeSeriesSubsequenceSearcher._infer_epoch_unit(pd.Series([value]))
            parsed = pd.to_datetime(value, unit=unit, utc=True)
        else:
            parsed = pd.to_datetime(value, utc=True)

        if pd.isna(parsed):
            raise ValueError(f"Could not parse timestamp: {value!r}")

        return pd.Timestamp(parsed).tz_convert(None)

    @staticmethod
    def _infer_epoch_unit(values: pd.Series) -> str:
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return "s"

        max_abs = float(numeric.abs().max())
        if max_abs >= 1e17:
            return "ns"
        if max_abs >= 1e14:
            return "us"
        if max_abs >= 1e11:
            return "ms"
        return "s"

    @staticmethod
    def _infer_common_freq(raw_series: dict[str, pd.Series]) -> pd.Timedelta:
        median_deltas_ns = []
        for series in raw_series.values():
            index_ns = series.index.drop_duplicates().sort_values().asi8
            deltas = np.diff(index_ns)
            positive_deltas = deltas[deltas > 0]
            if len(positive_deltas):
                median_deltas_ns.append(int(np.median(positive_deltas)))

        if not median_deltas_ns:
            raise ValueError("Could not infer frequency from the input series")

        freq_ns = reduce(gcd, median_deltas_ns)
        if freq_ns <= 0:
            raise ValueError("Inferred frequency must be positive")
        return pd.Timedelta(freq_ns, unit="ns")

    @staticmethod
    def _coerce_query_array(query: np.ndarray | pd.Series | list[float]) -> np.ndarray:
        query_values = np.asarray(query, dtype=float)
        query_values = np.squeeze(query_values)
        if query_values.ndim != 1:
            raise ValueError("query must be a one-dimensional array or pandas Series")
        if len(query_values) == 0:
            raise ValueError("query must contain at least one value")
        if not np.all(np.isfinite(query_values)):
            raise ValueError("query contains NaN or infinite values after preparation")
        return query_values

    @staticmethod
    def _normalize_excluded_ids(exclude_series_id: str | Iterable[str] | None) -> set[str]:
        if exclude_series_id is None:
            return set()
        if isinstance(exclude_series_id, str):
            return {exclude_series_id}
        return {str(series_id) for series_id in exclude_series_id}

    def _require_fitted(self) -> None:
        if not self.series_:
            raise RuntimeError("Call fit(series_dict) before searching")

    @staticmethod
    def _empty_result() -> pd.DataFrame:
        return pd.DataFrame(columns=RESULT_COLUMNS)
