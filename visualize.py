import pandas as pd
import matplotlib.pyplot as plt
from anomaly_detection.core.time_series import TimeSeriesWrapper
from anomaly_detection.models.stl import STLDetector
from visualize_common import plot_result, redraw_ax, setup_zoom_sync, add_holiday_slider

HOLIDAY_PARAM_INIT = 2.0


def load_dataset(series_num: int) -> pd.DataFrame:
    path = f"data/yandex/series_{series_num:03d}/data.csv"
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    return df


def run_model(df: pd.DataFrame, use_holidays: bool, holiday_param: float = HOLIDAY_PARAM_INIT):
    ts = TimeSeriesWrapper(df[["value"]])
    detector = STLDetector(apply_holidays=use_holidays)
    if use_holidays:
        detector.holiday_param = holiday_param
    result = detector(ts)
    dates = ts.time_series_pd.index
    values = ts.time_series_pd.iloc[:, 0].values
    return dates, values, result


def main():
    series_num = int(input("Enter dataset number (0-13): "))
    df = load_dataset(series_num)

    dates_off, values_off, result_off = run_model(df, use_holidays=False)
    dates_on, values_on, result_on = run_model(df, use_holidays=True, holiday_param=HOLIDAY_PARAM_INIT)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    fig.subplots_adjust(bottom=0.15)
    fig.suptitle(f"STL Anomaly Detection — Dataset {series_num:03d}", fontsize=12)

    plot_result(ax1, dates_off, values_off, result_off, "Without holidays")
    plot_result(ax2, dates_on, values_on, result_on, "With holidays")

    setup_zoom_sync(fig, ax1, ax2)

    def on_slider_change(val):
        nonlocal result_on
        _, _, result_on = run_model(df, use_holidays=True, holiday_param=val)
        redraw_ax(ax2, dates_on, values_on, result_on, f"With holidays (param={val:.2f})")
        fig.canvas.draw_idle()

    fig._holiday_slider = add_holiday_slider(fig, on_slider_change, init=HOLIDAY_PARAM_INIT)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
