from abc import ABC, abstractmethod

import holidays
import pandas as pd
from pydantic import BaseModel, field_validator
from typing import Dict, Any, Optional

import numpy as np
from scipy import stats
from statsmodels.robust.scale import qn_scale
from ..core import TimeSeriesWrapper


class ModelResult(BaseModel):
    anomaly_scores: Any
    is_anomaly: Any
    expected_value: Any = None
    expected_bounds: Any = None
    metadata: Dict[str, Any] = {}

    @field_validator("anomaly_scores", "is_anomaly")
    @classmethod
    def check_anomaly_scores_numpy_array(cls, v, info):
        if not isinstance(v, np.ndarray):
            raise TypeError(f"{info.field_name} must be a numpy.ndarray")
        if v.ndim != 1:
            raise ValueError(f"{info.field_name} must be a 1D array, but got {v.ndim}D array with shape {v.shape}")
        return v

    @field_validator("expected_value", "expected_bounds")
    @classmethod
    def check_expected_value_numpy_array(cls, v, info):
        if not isinstance(v, np.ndarray) and v is not None:
            raise TypeError(f"{info.field_name} must be a numpy.ndarray or None")
        return v


class BaseDetector(ABC):
    def __init__(self, apply_holidays=False, holiday_param=None, **kwargs):
        self.params = {**self.get_default_params(), **kwargs}
        if "std_type" not in self.params:
            self.params["std_type"] = "default"
        self.validate_params(self.params)

        self._apply_holidays = apply_holidays
        if apply_holidays:
            self.holiday_param = holiday_param if holiday_param is not None else np.float64(2.0)
            self._holiday_lr = self.params.get("holiday_lr", 0.068)
            self._holiday_decay = 0.999

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        pass

    def validate_params(self, params: Dict[str, Any]) -> None:
        pass

    def _detect_multivariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support multivariate time series. "
            f"Received {time_series.n_series} series."
        )

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement univariate detection.")

    def __call__(self, time_series: TimeSeriesWrapper, dates=None) -> ModelResult:
        if time_series.is_multivariate:
            return self._detect_multivariate(time_series)
        else:
            return self._detect_univariate(time_series, dates=dates)

    def _apply_holiday_envelope(self, std_array: np.ndarray, dates) -> np.ndarray:
        """
        Scale std_array with a raised-cosine envelope around each holiday.
        At the holiday centre: multiplier = holiday_param (maximum widening).
        At the window edge: multiplier = 0.5*(1 + holiday_param) (half the widening).
        Outside the window: multiplier = 1.0 (no effect).
        """
        hols = holidays.RU()
        window = 7  # days of influence on each side of the holiday

        min_date = pd.Timestamp(dates.min()) - pd.Timedelta(days=window)
        max_date = pd.Timestamp(dates.max()) + pd.Timedelta(days=window)
        holiday_dates = [d for d in pd.date_range(min_date, max_date) if d in hols]

        if not holiday_dates:
            return std_array

        days_diff = np.abs(
            (dates.values[:, None] - pd.DatetimeIndex(holiday_dates).values)
            / np.timedelta64(1, 'D')
        )

        # raised cosine: 1.0 at centre, 0.5 at edge, 0.0 outside
        effect = np.where(
            days_diff <= window,
            0.5 * (1.0 + np.cos(np.pi * days_diff / window)),
            0.0,
        ).max(axis=1)

        # multiplier: 1.0 where effect=0, holiday_param where effect=1
        multiplier = 1.0 + (self.holiday_param - 1.0) * effect
        return std_array * multiplier

    def calculate_std(self, residual: np.array, apply_holidays=False, dates=None, return_array=True) -> np.ndarray:
        def _calculate_std(self, residual: np.array) -> float:
            if self.params["std_type"] == "default":
                return np.sqrt(np.mean(residual**2))
            elif self.params["std_type"] == "default_robust":
                r = np.asarray(residual)
                abs_r = np.abs(r)
                if abs_r.size < 20:
                    return np.std(r)
                q_lo, q_hi = 0.75, 0.98
                a_emp, b_emp = np.quantile(abs_r, [q_lo, q_hi])
                mask = (abs_r >= a_emp) & (abs_r <= b_emp)
                trimmed = r[mask]
                if trimmed.size < 10:
                    return np.std(r)
                sigma_raw2 = np.mean(trimmed**2)
                a = stats.norm.ppf(q_lo)
                b = stats.norm.ppf(q_hi)
                phi = stats.norm.pdf
                Phi = stats.norm.cdf
                C = ((-b * phi(b) + Phi(b)) - (-a * phi(a) + Phi(a))) / (Phi(b) - Phi(a))
                return np.sqrt(sigma_raw2 / C)
            elif self.params["std_type"] == "mad":
                return np.median(np.abs(residual)) / stats.norm.ppf(0.75)
            elif self.params["std_type"] == "iqr":
                return np.subtract(*np.percentile(residual, [75, 25])) / (stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
            elif self.params["std_type"] == "qn_scale":
                return qn_scale(residual)
            else:
                raise ValueError(f"Unknown std_type: {self.params['std_type']}")

        ans = _calculate_std(self, residual)

        if apply_holidays and hasattr(self, 'holiday_param') and dates is not None:
            if len(dates) != len(residual):
                raise ValueError(f"Shapes mismatch: {len(dates)} != {len(residual)}")
            std_array = np.full(len(residual), ans) if np.isscalar(ans) else ans.copy()
            std_array = self._apply_holiday_envelope(std_array, dates)
            if return_array:
                return std_array
            else:
                return float(np.mean(std_array))

        if return_array and np.isscalar(ans):
            return np.full(len(residual), ans)
        return ans

    def calculate_std_backward(
        self,
        dates: list,
        predicted: np.array,
        ground_truth: np.array,
        holiday_mask: np.array = None,
    ) -> float:
        if not hasattr(self, 'holiday_param'):
            return np.float64(2.0)

        lr = self._holiday_lr
        self._holiday_lr *= self._holiday_decay

        threshold = self.params.get("threshold", 3.0)

        # Build boolean holiday mask: prefer explicit mask, fall back to holidays.RU
        if holiday_mask is not None:
            is_holiday = np.asarray(holiday_mask, dtype=bool)
        else:
            hols = holidays.RU()
            is_holiday = np.array([
                (d.date() if hasattr(d, 'date') else pd.Timestamp(d).date()) in hols
                for d in dates
            ])

        n_hol = int(is_holiday.sum())
        if n_hol == 0:
            return float(self.holiday_param)

        scores_hol = predicted[is_holiday]
        gt_hol = ground_truth[is_holiday].astype(int)

        def _hard_f1_at(beta):
            ratio = self.holiday_param / beta
            pred_hol = (scores_hol * ratio > threshold).astype(int)
            tp = int(((pred_hol == 1) & (gt_hol == 1)).sum())
            fp = int(((pred_hol == 1) & (gt_hol == 0)).sum())
            fn = int(((pred_hol == 0) & (gt_hol == 1)).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        # Grid search over candidate betas, pick the one maximising holiday F1
        candidates = np.linspace(0.2, 5.0, 49)
        best_beta = max(candidates, key=_hard_f1_at)
        # Move toward best_beta with learning rate (don't jump directly — stay smooth)
        self.holiday_param += lr * (best_beta - self.holiday_param)

        self.holiday_param = np.clip(self.holiday_param, 0.2, 5.0)
        return float(self.holiday_param)

    def calculate_seasonal_std(
        self, residual: np.array, period: int,
        clip: Optional[tuple[float, float]] = None,
        apply_holidays: bool = False,
        dates=None
    ) -> np.ndarray:
        if clip is None:
            clip = (0.5, 3.0)

        statistical_evidence_bound = 500
        num_of_periods = len(residual) // period + 1
        window_size = max((statistical_evidence_bound // num_of_periods) // 2, 1) + 1
        values = []
        overall_std = self.calculate_std(residual, apply_holidays=False, return_array=False)

        for i in range(period):
            phase_values_by_period = []
            for period_idx in range(num_of_periods):
                period_phase_values = []
                for j in range(-window_size, window_size + 1):
                    idx = period_idx * period + (i + j) % period
                    if 0 <= idx < len(residual):
                        period_phase_values.append(residual[idx])
                phase_values_by_period.append(period_phase_values)

            period_stds = []
            for period_phase_values in phase_values_by_period:
                if len(period_phase_values) > 0:
                    period_stds.append(
                        self.calculate_std(
                            np.array(period_phase_values),
                            apply_holidays=False,
                            return_array=False
                        )
                    )
                else:
                    period_stds.append(0)

            if len(period_stds) > 0:
                period_stds_arr = np.array(period_stds)
                percentile_80 = (
                    np.percentile(period_stds_arr[period_stds_arr > 0], 80)
                    if np.sum(period_stds_arr > 0) > 2
                    else np.max(period_stds_arr)
                )
                good_stds = period_stds_arr[period_stds_arr <= percentile_80]

                if len(good_stds) > 0:
                    phase_std = np.median(good_stds)
                else:
                    phase_std = overall_std
            else:
                phase_std = overall_std

            values.append(max(phase_std, 1e-12))

        values_arr = np.array(values)
        values_tiled = np.tile(values_arr, num_of_periods)[:len(residual)]
        clipped = np.clip(values_tiled, overall_std * clip[0], overall_std * clip[1])

        if apply_holidays and hasattr(self, 'holiday_param') and dates is not None:
            if len(dates) != len(clipped):
                raise ValueError(f"Shapes mismatch: {len(dates)} != {len(clipped)}")
            clipped = self._apply_holiday_envelope(clipped, dates)

        return clipped
