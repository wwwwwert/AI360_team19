from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .system import AnomalyDetectionSystem, DetectionResult
from .time_series import TimeSeriesWrapper


TimeLike = Union[str, datetime, pd.Timestamp]
GapLike = Union[int, str, pd.Timedelta, None]

SUPPORTED_OVERLAP_METRICS = {"recall", "precision", "iou", "f1"}


@dataclass(frozen=True)
class AnomalyInterval:
    """Inclusive interval over a single time-series index."""

    start: pd.Timestamp
    end: pd.Timestamp
    start_idx: int
    end_idx: int
    n_anomaly_points: int

    @property
    def n_points(self) -> int:
        return self.end_idx - self.start_idx + 1

    def to_tuple(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.start, self.end


@dataclass(frozen=True)
class IntervalOverlapMetrics:
    recall: float
    precision: float
    iou: float
    f1: float
    overlap_points: int
    query_points: int
    candidate_points: int
    union_points: int

    def as_dict(self) -> dict[str, Union[float, int]]:
        return {
            "recall": self.recall,
            "precision": self.precision,
            "iou": self.iou,
            "f1": self.f1,
            "overlap_points": self.overlap_points,
            "query_points": self.query_points,
            "candidate_points": self.candidate_points,
            "union_points": self.union_points,
        }


@dataclass(frozen=True)
class SeriesIntervalMatch:
    series_name: str
    intervals: list[AnomalyInterval]
    metrics: IntervalOverlapMetrics
    score: float
    matched: bool
    is_query: bool = False


@dataclass
class AnomalyIntervalSearchResult:
    query_series: str
    query_start: pd.Timestamp
    query_end: pd.Timestamp
    query_intervals: list[AnomalyInterval]
    series_matches: dict[str, SeriesIntervalMatch]
    detections: dict[str, DetectionResult] = field(repr=False)
    window_index: pd.DatetimeIndex = field(repr=False)
    score_metric: str = "f1"
    overlap_threshold: float = 0.5

    @property
    def matching_series(self) -> list[str]:
        return [
            name
            for name, match in self.series_matches.items()
            if match.matched and not match.is_query
        ]

    @property
    def matches(self) -> dict[str, SeriesIntervalMatch]:
        return {
            name: match
            for name, match in self.series_matches.items()
            if match.matched and not match.is_query
        }

    def to_frame(self, matched_only: bool = False, include_query: bool = False) -> pd.DataFrame:
        rows = []
        for name, match in self.series_matches.items():
            if match.is_query and not include_query:
                continue
            if matched_only and not match.matched:
                continue

            metrics = match.metrics.as_dict()
            rows.append(
                {
                    "series_name": name,
                    "is_query": match.is_query,
                    "matched": match.matched,
                    "score": match.score,
                    **metrics,
                    "n_intervals": len(match.intervals),
                    "intervals": format_intervals(match.intervals),
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "series_name",
                    "is_query",
                    "matched",
                    "score",
                    "recall",
                    "precision",
                    "iou",
                    "f1",
                    "overlap_points",
                    "query_points",
                    "candidate_points",
                    "union_points",
                    "n_intervals",
                    "intervals",
                ]
            )

        return pd.DataFrame(rows).sort_values(
            ["matched", "score", "iou", "f1"],
            ascending=[False, False, False, False],
            ignore_index=True,
        )


def extract_anomaly_intervals(
    index: Sequence[pd.Timestamp],
    is_anomaly: Sequence[bool],
    *,
    max_gap: GapLike = 0,
    min_points: int = 1,
) -> list[AnomalyInterval]:
    """
    Convert point-wise anomaly labels into inclusive intervals.

    Consecutive anomalous points are grouped first, groups with a small gap are
    merged next, and short intervals are discarded last.
    """
    index = pd.DatetimeIndex(index)
    mask = np.asarray(is_anomaly, dtype=bool).reshape(-1)

    if len(index) != len(mask):
        raise ValueError(f"index length ({len(index)}) and is_anomaly length ({len(mask)}) must match")
    if min_points < 1:
        raise ValueError("min_points must be >= 1")
    if len(mask) == 0 or not mask.any():
        return []

    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True])

    intervals = [
        AnomalyInterval(
            start=index[start],
            end=index[end],
            start_idx=int(start),
            end_idx=int(end),
            n_anomaly_points=int(mask[start : end + 1].sum()),
        )
        for start, end in zip(starts, ends)
    ]

    intervals = _merge_close_intervals(intervals, index=index, max_gap=max_gap)
    return [interval for interval in intervals if interval.n_anomaly_points >= min_points]


def intervals_to_mask(intervals: Sequence[AnomalyInterval], length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for interval in intervals:
        if interval.start_idx < 0 or interval.end_idx >= length:
            raise ValueError("interval indices are outside the requested mask length")
        mask[interval.start_idx : interval.end_idx + 1] = True
    return mask


def compute_interval_overlap_metrics(
    query_mask: Sequence[bool],
    candidate_mask: Sequence[bool],
) -> IntervalOverlapMetrics:
    query_mask = np.asarray(query_mask, dtype=bool).reshape(-1)
    candidate_mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)

    if len(query_mask) != len(candidate_mask):
        raise ValueError("query_mask and candidate_mask must have the same length")

    overlap = query_mask & candidate_mask
    union = query_mask | candidate_mask

    overlap_points = int(overlap.sum())
    query_points = int(query_mask.sum())
    candidate_points = int(candidate_mask.sum())
    union_points = int(union.sum())

    recall = overlap_points / query_points if query_points else 0.0
    precision = overlap_points / candidate_points if candidate_points else 0.0
    iou = overlap_points / union_points if union_points else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return IntervalOverlapMetrics(
        recall=recall,
        precision=precision,
        iou=iou,
        f1=f1,
        overlap_points=overlap_points,
        query_points=query_points,
        candidate_points=candidate_points,
        union_points=union_points,
    )


def format_intervals(intervals: Sequence[AnomalyInterval]) -> str:
    if not intervals:
        return ""
    return "; ".join(f"{interval.start} - {interval.end}" for interval in intervals)


class AnomalyIntervalOverlapSearch:
    """
    Search for series whose anomalous intervals overlap a query series interval.

    The wrapped AnomalyDetectionSystem is applied independently to every column,
    because the existing multivariate detector returns a shared anomaly score for
    each timestamp rather than per-series anomaly labels.
    """

    def __init__(
        self,
        detector: AnomalyDetectionSystem,
        *,
        max_gap: GapLike = 0,
        min_points: int = 1,
        metric: str = "f1",
        metric_weights: Optional[Mapping[str, float]] = None,
        overlap_threshold: float = 0.5,
        required_metrics: Optional[Mapping[str, float]] = None,
    ):
        if not isinstance(detector, AnomalyDetectionSystem):
            raise TypeError("detector must be an AnomalyDetectionSystem instance")
        if metric not in SUPPORTED_OVERLAP_METRICS:
            raise ValueError(f"metric must be one of {sorted(SUPPORTED_OVERLAP_METRICS)}")
        if overlap_threshold < 0:
            raise ValueError("overlap_threshold must be >= 0")

        self.detector = detector
        self.max_gap = max_gap
        self.min_points = min_points
        self.metric = metric
        self.metric_weights = self._validate_metric_weights(metric_weights)
        self.overlap_threshold = overlap_threshold
        self.required_metrics = self._validate_required_metrics(required_metrics)

    def search(
        self,
        time_series: Union[
            TimeSeriesWrapper,
            pd.DataFrame,
            tuple[list[datetime], list[float]],
            tuple[list[datetime], list[list[float]]],
            list[tuple[list[datetime], list[float]]],
        ],
        *,
        query_series: Union[str, int],
        start_time: TimeLike,
        end_time: TimeLike,
        candidate_series: Optional[Sequence[Union[str, int]]] = None,
    ) -> AnomalyIntervalSearchResult:
        if isinstance(time_series, TimeSeriesWrapper):
            df = time_series.original_time_series
        else:
            df = TimeSeriesWrapper(time_series).original_time_series

        df = df.copy()
        df.columns = [str(column) for column in df.columns]

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        query_name = self._resolve_series_name(df, query_series)
        candidate_names = self._resolve_candidate_names(df, query_name, candidate_series)

        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)
        if start_ts > end_ts:
            raise ValueError("start_time must be <= end_time")

        window_mask = (df.index >= start_ts) & (df.index <= end_ts)
        window_index = pd.DatetimeIndex(df.index[window_mask])
        if len(window_index) == 0:
            raise ValueError("The requested query time range does not contain any points")

        detections = self._detect_each_series(df, [query_name, *candidate_names])
        window_intervals = {
            name: extract_anomaly_intervals(
                window_index,
                detections[name].is_anomaly[window_mask],
                max_gap=self.max_gap,
                min_points=self.min_points,
            )
            for name in [query_name, *candidate_names]
        }

        query_intervals = window_intervals[query_name]
        query_interval_mask = intervals_to_mask(query_intervals, len(window_index))

        series_matches = {}
        for name in [query_name, *candidate_names]:
            intervals = window_intervals[name]
            candidate_interval_mask = intervals_to_mask(intervals, len(window_index))
            metrics = compute_interval_overlap_metrics(query_interval_mask, candidate_interval_mask)
            score = self._score(metrics)
            is_query = name == query_name
            matched = self._is_matched(metrics, score)
            series_matches[name] = SeriesIntervalMatch(
                series_name=name,
                intervals=intervals,
                metrics=metrics,
                score=score,
                matched=matched,
                is_query=is_query,
            )

        return AnomalyIntervalSearchResult(
            query_series=query_name,
            query_start=start_ts,
            query_end=end_ts,
            query_intervals=query_intervals,
            series_matches=series_matches,
            detections=detections,
            window_index=window_index,
            score_metric="weighted" if self.metric_weights else self.metric,
            overlap_threshold=self.overlap_threshold,
        )

    def search_series_dict(
        self,
        series_dict: Mapping[str, pd.Series],
        *,
        query_series: str,
        start_time: TimeLike,
        end_time: TimeLike,
        candidate_series: Optional[Sequence[str]] = None,
        alignment_tolerance: GapLike = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> AnomalyIntervalSearchResult:
        """
        Search independent time-series whose anomaly intervals overlap in time.

        Unlike search(), this method does not require all series to share one
        DataFrame index. Each series is detected on its own timestamp index, and
        candidate anomaly masks are aligned to the query window index only for
        overlap scoring.
        """
        if not series_dict:
            raise ValueError("series_dict must not be empty")

        prepared = {
            str(name): self._prepare_independent_series(name, series)
            for name, series in series_dict.items()
        }
        if query_series not in prepared:
            raise ValueError(f"Unknown query series {query_series!r}. Available series: {list(prepared)}")

        query_name = str(query_series)
        if candidate_series is None:
            candidate_names = [name for name in prepared if name != query_name]
        else:
            candidate_names = [str(name) for name in candidate_series if str(name) != query_name]
            unknown = [name for name in candidate_names if name not in prepared]
            if unknown:
                raise ValueError(f"Unknown candidate series: {unknown}")

        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)
        if start_ts > end_ts:
            raise ValueError("start_time must be <= end_time")

        query_window_index = pd.DatetimeIndex(prepared[query_name].loc[start_ts:end_ts].index)
        if len(query_window_index) == 0:
            raise ValueError("The requested query time range does not contain any query points")

        series_names = [query_name, *candidate_names]
        detections = self._detect_independent_series(
            prepared,
            series_names,
            progress_callback=progress_callback,
        )
        window_intervals = {
            name: self._extract_window_intervals(
                prepared[name],
                detections[name],
                start_ts,
                end_ts,
            )
            for name in series_names
        }

        query_intervals = window_intervals[query_name]
        query_interval_mask = self._intervals_to_reference_mask(
            query_intervals,
            query_window_index,
            alignment_tolerance=None,
        )

        series_matches = {}
        for name in series_names:
            if name == query_name:
                candidate_interval_mask = query_interval_mask
            else:
                candidate_interval_mask = self._intervals_to_reference_mask(
                    window_intervals[name],
                    query_window_index,
                    alignment_tolerance=alignment_tolerance,
                )

            metrics = compute_interval_overlap_metrics(query_interval_mask, candidate_interval_mask)
            score = self._score(metrics)
            is_query = name == query_name
            matched = self._is_matched(metrics, score)
            series_matches[name] = SeriesIntervalMatch(
                series_name=name,
                intervals=window_intervals[name],
                metrics=metrics,
                score=score,
                matched=matched,
                is_query=is_query,
            )

        return AnomalyIntervalSearchResult(
            query_series=query_name,
            query_start=start_ts,
            query_end=end_ts,
            query_intervals=query_intervals,
            series_matches=series_matches,
            detections=detections,
            window_index=query_window_index,
            score_metric="weighted" if self.metric_weights else self.metric,
            overlap_threshold=self.overlap_threshold,
        )

    def _detect_each_series(self, df: pd.DataFrame, series_names: Sequence[str]) -> dict[str, DetectionResult]:
        detections: dict[str, DetectionResult] = {}
        seen = set()

        for name in series_names:
            if name in seen:
                continue
            seen.add(name)
            single_series = pd.DataFrame({"value_0": df[name].astype(float)}, index=df.index)
            detections[name] = self.detector.detect(single_series)

        return detections

    def _detect_independent_series(
        self,
        series_dict: Mapping[str, pd.Series],
        series_names: Sequence[str],
        *,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict[str, DetectionResult]:
        detections: dict[str, DetectionResult] = {}
        unique_names = list(dict.fromkeys(series_names))
        total = len(unique_names)

        for i, name in enumerate(unique_names, start=1):
            single_series = pd.DataFrame({"value_0": series_dict[name].astype(float)})
            detections[name] = self.detector.detect(single_series)
            if progress_callback is not None:
                progress_callback(i, total, name)

        return detections

    def _extract_window_intervals(
        self,
        series: pd.Series,
        detection: DetectionResult,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> list[AnomalyInterval]:
        window_mask = (series.index >= start_ts) & (series.index <= end_ts)
        window_index = pd.DatetimeIndex(series.index[window_mask])
        if len(window_index) == 0:
            return []

        return extract_anomaly_intervals(
            window_index,
            detection.is_anomaly[window_mask],
            max_gap=self.max_gap,
            min_points=self.min_points,
        )

    @staticmethod
    def _prepare_independent_series(name: str, series: pd.Series) -> pd.Series:
        if not isinstance(series, pd.Series):
            raise TypeError(f"{name!r} must be a pandas Series")

        prepared = pd.to_numeric(series, errors="coerce").dropna()
        if prepared.empty:
            raise ValueError(f"{name!r} is empty")
        if not isinstance(prepared.index, pd.DatetimeIndex):
            prepared.index = pd.to_datetime(prepared.index)

        prepared = prepared[~prepared.index.isna()]
        prepared = prepared.sort_index()
        prepared = prepared[~prepared.index.duplicated(keep="last")]
        if prepared.empty:
            raise ValueError(f"{name!r} does not contain valid timestamps")
        return prepared

    @staticmethod
    def _intervals_to_reference_mask(
        intervals: Sequence[AnomalyInterval],
        reference_index: pd.DatetimeIndex,
        *,
        alignment_tolerance: GapLike,
    ) -> np.ndarray:
        reference_index = pd.DatetimeIndex(reference_index)
        mask = np.zeros(len(reference_index), dtype=bool)
        if len(reference_index) == 0:
            return mask

        tolerance = pd.Timedelta(0) if alignment_tolerance is None else pd.Timedelta(alignment_tolerance)
        for interval in intervals:
            mask |= (reference_index >= interval.start - tolerance) & (reference_index <= interval.end + tolerance)
        return mask

    def _score(self, metrics: IntervalOverlapMetrics) -> float:
        metric_values = metrics.as_dict()
        if not self.metric_weights:
            return float(metric_values[self.metric])

        weighted_sum = sum(metric_values[name] * weight for name, weight in self.metric_weights.items())
        return float(weighted_sum / sum(self.metric_weights.values()))

    def _is_matched(self, metrics: IntervalOverlapMetrics, score: float) -> bool:
        if score < self.overlap_threshold:
            return False

        metric_values = metrics.as_dict()
        return all(metric_values[name] >= threshold for name, threshold in self.required_metrics.items())

    @staticmethod
    def _resolve_series_name(df: pd.DataFrame, series: Union[str, int]) -> str:
        if isinstance(series, int):
            try:
                return str(df.columns[series])
            except IndexError as exc:
                raise ValueError(f"Series index {series} is outside dataframe columns") from exc

        if series not in df.columns:
            raise ValueError(f"Unknown series {series!r}. Available series: {list(df.columns)}")
        return str(series)

    @classmethod
    def _resolve_candidate_names(
        cls,
        df: pd.DataFrame,
        query_name: str,
        candidate_series: Optional[Sequence[Union[str, int]]],
    ) -> list[str]:
        if candidate_series is None:
            return [str(column) for column in df.columns if str(column) != query_name]

        names = [cls._resolve_series_name(df, series) for series in candidate_series]
        return [name for name in dict.fromkeys(names) if name != query_name]

    @staticmethod
    def _validate_metric_weights(metric_weights: Optional[Mapping[str, float]]) -> Optional[dict[str, float]]:
        if metric_weights is None:
            return None

        weights = {str(name): float(weight) for name, weight in metric_weights.items()}
        unknown = set(weights) - SUPPORTED_OVERLAP_METRICS
        if unknown:
            raise ValueError(f"Unknown metric weights: {sorted(unknown)}")
        if not weights:
            raise ValueError("metric_weights must not be empty")
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("metric weights must be >= 0")
        if sum(weights.values()) <= 0:
            raise ValueError("At least one metric weight must be positive")
        return weights

    @staticmethod
    def _validate_required_metrics(required_metrics: Optional[Mapping[str, float]]) -> dict[str, float]:
        if required_metrics is None:
            return {}

        thresholds = {str(name): float(threshold) for name, threshold in required_metrics.items()}
        unknown = set(thresholds) - SUPPORTED_OVERLAP_METRICS
        if unknown:
            raise ValueError(f"Unknown required metrics: {sorted(unknown)}")
        if any(threshold < 0 for threshold in thresholds.values()):
            raise ValueError("required metric thresholds must be >= 0")
        return thresholds


def _merge_close_intervals(
    intervals: Sequence[AnomalyInterval],
    *,
    index: pd.DatetimeIndex,
    max_gap: GapLike,
) -> list[AnomalyInterval]:
    if not intervals:
        return []

    merged = [intervals[0]]
    for interval in intervals[1:]:
        previous = merged[-1]
        if _should_merge(previous, interval, index=index, max_gap=max_gap):
            merged[-1] = AnomalyInterval(
                start=previous.start,
                end=interval.end,
                start_idx=previous.start_idx,
                end_idx=interval.end_idx,
                n_anomaly_points=previous.n_anomaly_points + interval.n_anomaly_points,
            )
        else:
            merged.append(interval)
    return merged


def _should_merge(
    previous: AnomalyInterval,
    current: AnomalyInterval,
    *,
    index: pd.DatetimeIndex,
    max_gap: GapLike,
) -> bool:
    if max_gap is None:
        max_gap = 0

    if isinstance(max_gap, int):
        if max_gap < 0:
            raise ValueError("max_gap must be >= 0")
        gap_points = current.start_idx - previous.end_idx - 1
        return gap_points <= max_gap

    max_gap_timedelta = pd.Timedelta(max_gap)
    if max_gap_timedelta < pd.Timedelta(0):
        raise ValueError("max_gap must be >= 0")

    # Treat timestamps as samples on a grid: if one point between intervals is
    # normal, the gap is one sampling step rather than two timestamp jumps.
    sample_step = index[previous.end_idx + 1] - index[previous.end_idx]
    gap = index[current.start_idx] - index[previous.end_idx] - sample_step
    if gap < pd.Timedelta(0):
        gap = pd.Timedelta(0)
    return gap <= max_gap_timedelta
