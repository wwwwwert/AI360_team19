from anomaly_detection import TimeSeriesWrapper, AnomalyDetectionSystem
from anomaly_detection.models import STLDetector
import pandas as pd


def linear_interpolate(df: pd.DataFrame, value_column: str = 'value_0'):
    """
    Линейно интерполирует значения в строках, где is_anomaly=1,
    используя соседние значения, где is_anomaly=0.
    Если соседа нет с какой-то стороны, заменяет на ближайшее известное значение.

    Parameters:
    df: DataFrame с колонками value_column и is_anomaly
    value_column: название колонки со значениями для интерполяции
    """
    df = df.copy()

    # Находим маски
    anomaly_mask = df['is_anomaly'] == 1
    normal_mask = df['is_anomaly'] == 0

    # Итерируемся по всем строкам
    for idx in df.index:
        if anomaly_mask[idx]:
            # Ищем предыдущую нормальную точку
            prev_normal = df[normal_mask & (df.index < idx)]
            # Ищем следующую нормальную точку
            next_normal = df[normal_mask & (df.index > idx)]

            # Определяем значения для интерполяции
            if len(prev_normal) == 0 and len(next_normal) == 0:
                # Если оба соседа отсутствуют, оставляем значение как есть
                # или можно установить 0, но это крайний случай
                df.loc[idx, value_column] = 0
            elif len(prev_normal) == 0:
                # Если нет левого соседа, используем ближайшее правое значение
                right_value = next_normal.iloc[0][value_column]
                df.loc[idx, value_column] = right_value
            elif len(next_normal) == 0:
                # Если нет правого соседа, используем ближайшее левое значение
                left_value = prev_normal.iloc[-1][value_column]
                df.loc[idx, value_column] = left_value
            else:
                # Есть оба соседа - делаем линейную интерполяцию
                left_value = prev_normal.iloc[-1][value_column]
                left_idx = prev_normal.index[-1]
                right_value = next_normal.iloc[0][value_column]
                right_idx = next_normal.index[0]

                # Преобразуем индексы в числовые значения для интерполяции
                left_num = left_idx.timestamp()
                right_num = right_idx.timestamp()
                current_num = idx.timestamp()

                # Линейная интерполяция
                fraction = (current_num - left_num) / (right_num - left_num)
                interpolated_value = left_value + fraction * (right_value - left_value)
                df.loc[idx, value_column] = interpolated_value

    return df


def z_scores_anomaly_transform(series: TimeSeriesWrapper, system: AnomalyDetectionSystem) -> TimeSeriesWrapper:
    detector = AnomalyDetectionSystem.AVAILABLE_MODELS["STLDetector"](**system.detection_model_params)

    z_scores = detector.z_scores(series)
    model_output = detector(series)

    df = series.time_series_pd
    df["is_anomaly"] = model_output.is_anomaly

    # Добавляем z_scores в DataFrame
    df["z_score"] = z_scores.values if hasattr(z_scores, 'values') else z_scores

    # Интерполируем z_scores вместо value_0
    df = linear_interpolate(df, value_column='z_score')

    return TimeSeriesWrapper(df)
