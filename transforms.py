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

    df = linear_interpolate(
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