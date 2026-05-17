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
            hols = holidays.RU()
            if len(dates) != len(residual):
                raise ValueError(f"Shapes mismatch: {len(dates)} != {len(residual)}")

            if np.isscalar(ans):
                std_array = np.full(len(residual), ans)
            else:
                std_array = ans.copy()

            for i in range(len(dates)):
                if dates[i] in hols:
                    std_array[i] *= self.holiday_param

            if return_array:
                return std_array
            else:
                return float(np.mean(std_array))

        if return_array and np.isscalar(ans):
            return np.full(len(residual), ans)
        return ans

    def calculate_std_backward(self, dates: list, predicted: np.array, ground_truth: np.array) -> float:
        if not hasattr(self, 'holiday_param'):
            return np.float64(2.0)

        hols = holidays.RU()
        lr = self._holiday_lr
        self._holiday_lr *= self._holiday_decay

        threshold = self.params.get("threshold", 3.0)
        missed = 0      # gt=1, pred<threshold → нужно сузить std → уменьшить param
        false_alarm = 0 # gt=0, pred>threshold → нужно расширить std → увеличить param
        n_hol = 0

        for i in range(len(dates)):
            date_only = dates[i].date() if hasattr(dates[i], 'date') else pd.Timestamp(dates[i]).date()
            if date_only not in hols:
                continue
            n_hol += 1
            gt = ground_truth[i]
            pred_score = predicted[i]
            if gt == 1 and pred_score < threshold:
                missed += 1
            elif gt == 0 and pred_score > threshold:
                false_alarm += 1

        if n_hol > 0:
            net = (false_alarm - missed) / n_hol
            self.holiday_param += lr * net

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
        overall_std = self.calculate_std(residual, apply_holidays=apply_holidays, dates=dates, return_array=False)

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
        return np.clip(values_tiled, overall_std * clip[0], overall_std * clip[1])