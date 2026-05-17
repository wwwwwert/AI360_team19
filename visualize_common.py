import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Slider


def plot_result(ax, dates, values, result, title, true_anomaly_mask=None):
    ax.plot(dates, values, color="steelblue", linewidth=0.8, label="Actual", zorder=3)
    ax.plot(dates, result.expected_value, color="orange", linewidth=0.8, label="Predicted", zorder=2)
    ax.fill_between(
        dates,
        result.expected_bounds[:, 0],
        result.expected_bounds[:, 1],
        alpha=0.25,
        color="orange",
        label="Confidence band",
    )
    detected_dates = dates[result.is_anomaly]
    detected_vals = values[result.is_anomaly]
    ax.scatter(detected_dates, detected_vals, color="red", s=20, zorder=4, label="Detected anomalies")

    if true_anomaly_mask is not None:
        true_dates = dates[true_anomaly_mask]
        true_vals = values[true_anomaly_mask]
        ax.scatter(true_dates, true_vals, color="lime", s=40, marker="^", zorder=5, label="True anomalies")

    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def redraw_ax(ax, dates, values, result, title, true_anomaly_mask=None):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.cla()
    plot_result(ax, dates, values, result, title, true_anomaly_mask)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def setup_zoom_sync(fig, ax1, ax2):
    syncing = [False]

    def _sync_xlim(ax_src, ax_dst):
        def handler(event_ax):
            if event_ax == ax_src and not syncing[0]:
                syncing[0] = True
                ax_dst.set_xlim(ax_src.get_xlim())
                syncing[0] = False
                fig.canvas.draw_idle()
        return handler

    def _sync_ylim(ax_src, ax_dst):
        def handler(event_ax):
            if event_ax == ax_src and not syncing[0]:
                syncing[0] = True
                ax_dst.set_ylim(ax_src.get_ylim())
                syncing[0] = False
                fig.canvas.draw_idle()
        return handler

    ax1.callbacks.connect("xlim_changed", _sync_xlim(ax1, ax2))
    ax2.callbacks.connect("xlim_changed", _sync_xlim(ax2, ax1))
    ax1.callbacks.connect("ylim_changed", _sync_ylim(ax1, ax2))
    ax2.callbacks.connect("ylim_changed", _sync_ylim(ax2, ax1))


def add_holiday_slider(fig, on_change, init=2.0):
    ax_slider = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    slider = Slider(ax_slider, "Holiday param", 0.2, 5.0, valinit=init, valstep=0.05)
    slider.on_changed(on_change)
    return slider
