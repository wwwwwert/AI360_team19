from anomaly_detection import TimeSeriesWrapper, AnomalyDetectionSystem
from anomaly_detection.models import STLDetector
import pandas as pd


def linear_interpolate(
    df: pd.DataFrame,
    value_column: str = "value_0"
):
    """
    Интерполирует нормальные точки (is_anomaly=0)
    по соседним аномальным (is_anomaly=1).

    Аномальные точки остаются без изменений.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с колонками value_column и is_anomaly

    value_column : str
        Колонка для интерполяции
    """

    df = df.copy()

    anomaly_mask = df["is_anomaly"] == 1
    normal_mask = df["is_anomaly"] == 0

    for idx in df.index:

        # Интерполируем только нормальные
        if normal_mask.loc[idx]:

            prev_anomaly = df[
                anomaly_mask & (df.index < idx)
            ]

            next_anomaly = df[
                anomaly_mask & (df.index > idx)
            ]

            if (
                len(prev_anomaly) == 0
                and len(next_anomaly) == 0
            ):
                continue

            elif len(prev_anomaly) == 0:

                right_value = next_anomaly.iloc[0][value_column]

                df.loc[idx, value_column] = right_value

            elif len(next_anomaly) == 0:

                left_value = prev_anomaly.iloc[-1][value_column]

                df.loc[idx, value_column] = left_value

            else:

                left_value = prev_anomaly.iloc[-1][value_column]
                left_idx = prev_anomaly.index[-1]

                right_value = next_anomaly.iloc[0][value_column]
                right_idx = next_anomaly.index[0]

                left_num = left_idx.timestamp()
                right_num = right_idx.timestamp()
                current_num = idx.timestamp()

                fraction = (
                    current_num - left_num
                ) / (
                    right_num - left_num
                )

                interpolated_value = (
                    left_value
                    + fraction
                    * (right_value - left_value)
                )

                df.loc[
                    idx,
                    value_column
                ] = interpolated_value

    return df

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline


def spline_interpolate(
    df: pd.DataFrame,
    value_column: str = "value_0",
    min_points: int = 4
) -> pd.DataFrame:
    """
    Интерполирует нормальные точки (is_anomaly=0) с помощью кубического сплайна,
    построенного по аномальным точкам (is_anomaly=1).

    Аномальные точки остаются без изменений.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame с колонками value_column и is_anomaly
    value_column : str
        Колонка для интерполяции
    min_points : int, default=4
        Минимальное количество аномальных точек для построения сплайна.
        Если их меньше, выполняется линейная интерполяция (через linear_interpolate).

    Returns
    -------
    pd.DataFrame
        Копия DataFrame с интерполированными нормальными точками.
    """
    df = df.copy()

    anomaly_mask = df["is_anomaly"] == 1
    normal_mask = df["is_anomaly"] == 0

    # Если нет нормальных точек, возвращаем копию
    if not normal_mask.any():
        return df

    # Аномальные точки, по которым строим сплайн
    anomaly_df = df[anomaly_mask]


    # Преобразуем индексы в числовые значения (timestamp или порядковый номер)
    # Предполагаем, что индекс — это datetime, у которого есть метод timestamp()
    try:
        x_anomaly = np.array([idx.timestamp() for idx in anomaly_df.index])
    except AttributeError:
        # Если индекс не datetime, используем порядковый номер
        x_anomaly = np.arange(len(df))[anomaly_mask]
    y_anomaly = anomaly_df[value_column].values

    # Строим кубический сплайн
    cs = CubicSpline(x_anomaly, y_anomaly, extrapolate=False)

    # Интерполируем значения для нормальных точек
    normal_indices = df.index[normal_mask]
    try:
        x_normal = np.array([idx.timestamp() for idx in normal_indices])
    except AttributeError:
        x_normal = np.arange(len(df))[normal_mask]

    # Вычисляем интерполированные значения
    interpolated_values = cs(x_normal)

    # Заменяем значения в DataFrame
    df.loc[normal_mask, value_column] = interpolated_values

    return df

def z_scores_anomaly_transform(
    series: TimeSeriesWrapper,
    system: AnomalyDetectionSystem
) -> TimeSeriesWrapper:

    detector = AnomalyDetectionSystem.AVAILABLE_MODELS["STLDetector"](
        **system.detection_model_params
    )

    z_scores = detector.z_scores(series)
    model_output = detector(series)

    

    

    df = series.time_series_pd

    df["is_anomaly"] = model_output.is_anomaly

    df["z_score"] = (
        z_scores.values
        if hasattr(z_scores, "values")
        else z_scores
    )

    df = spline_interpolate(
        df,
        value_column="z_score"
    )

    df["value_0"] = df["z_score"]

    anomaly_mask = pd.Series(
        model_output.is_anomaly,
        index=series.time_series_pd.index
    ).astype(bool)

    z_scores_series = pd.Series(
        z_scores,
        index=series.time_series_pd.index
    )

    return TimeSeriesWrapper(df)