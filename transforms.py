from anomaly_detection import TimeSeriesWrapper, AnomalyDetectionSystem
from anomaly_detection.models import STLDetector
import pandas as pd


def linear_interpolate(df: pd.DataFrame):
    """
    Линейно интерполирует значения в строках, где is_anomaly=1,
    используя соседние значения, где is_anomaly=0.
    Если соседа нет с какой-то стороны, заменяет на 0.
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
            if len(prev_normal) == 0:
                left_value = 0
                left_idx = idx
            else:
                left_value = prev_normal.iloc[-1]['value_0']
                left_idx = prev_normal.index[-1]

            if len(next_normal) == 0:
                right_value = 0
                right_idx = idx
            else:
                right_value = next_normal.iloc[0]['value_0']
                right_idx = next_normal.index[0]

            # Если оба соседа отсутствуют, значение = 0
            if len(prev_normal) == 0 and len(next_normal) == 0:
                df.loc[idx, 'value_0'] = 0
            else:
                # Преобразуем индексы в числовые значения для интерполяции
                left_num = left_idx.timestamp()
                right_num = right_idx.timestamp()
                current_num = idx.timestamp()

                # Линейная интерполяция
                fraction = (current_num - left_num) / (right_num - left_num)
                interpolated_value = left_value + fraction * (right_value - left_value)
                df.loc[idx, 'value_0'] = interpolated_value

    return df

def z_scores_anomaly_transform(series: TimeSeriesWrapper, system: AnomalyDetectionSystem) -> TimeSeriesWrapper:
    detector = AnomalyDetectionSystem.AVAILABLE_MODELS["STLDetector"](**system.detection_model_params)

    z_scores =  detector.z_scores(series)
    model_output = detector(series)

    df = series.time_series_pd
    df["is_anomaly"] = model_output.is_anomaly

    df = linear_interpolate(df)

    return TimeSeriesWrapper(df)
#DANYA
