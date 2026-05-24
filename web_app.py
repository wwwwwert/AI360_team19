import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from anomaly_detection import AnomalyDetectionSystem, AnomalyIntervalOverlapSearch


def _running_inside_streamlit():
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    return get_script_run_ctx(suppress_warning=True) is not None


if __name__ == "__main__" and not _running_inside_streamlit():
    script_path = Path(__file__).resolve()
    cmd = [sys.executable, "-m", "streamlit", "run", str(script_path), *sys.argv[1:]]
    print("Starting Streamlit app:", flush=True)
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))

st.set_page_config(layout="wide", page_title="Time Series Similarity Search")


RESULT_COLUMNS = [
    "series_id",
    "start_idx",
    "end_idx",
    "distance",
    "start_time",
    "end_time",
]
QUERY_AUTO = "Auto"
QUERY_MINIMAP = "Minimap"
SEARCH_SIMILARITY = "Shape similarity"
SEARCH_ANOMALY_OVERLAP = "Anomaly overlap"
SEARCH_EVERYWHERE = "Entire series"
SEARCH_SAME_TIME = "Same time window"
METADATA_COLUMNS = {
    "ground_truth",
    "predicted",
    "is_anomaly",
    "anomaly_score",
    "anomaly_scores",
}


UI_TEXT = {
    "EN": {
        "data_header": "📁 Data",
        "upload_csv": "Upload CSV files",
        "loaded_series": "Loaded series: {count}",
        "file_col": "Series",
        "points_col": "Points",
        "start_col": "Start",
        "end_col": "End",
        "read_failed": "Could not read:\n{errors}",
        "params_header": "🎯 Search Parameters",
        "source_series": "Source series (query comes from here)",
        "window_len": "Window length (points)",
        "query_segment": "Query segment",
        "search_type": "Search type",
        "top_k": "Top-K results",
        "search_in_source": "Search inside the source series",
        "search_scope": "Search scope",
        "overlap_caption": "For anomaly overlap, intervals are compared inside the selected query time window.",
        "overlap_metric": "Overlap metric",
        "overlap_threshold": "Overlap threshold",
        "min_recall": "Minimum recall",
        "max_gap": "Merge gaps up to N points",
        "min_points": "Minimum points per interval",
        "detector": "Anomaly detector",
        "search_button": "🔍 Search",
        "upload_prompt": "⬅️ Upload CSV files in the sidebar",
        "select_nonempty_source": "Select a non-empty source series",
        "window_too_long": "Window length ({window}) is greater than source length ({length} points)",
        "window_too_short": "Similarity search needs at least 3 points in the window",
        "tab_query": "🎯 Query",
        "tab_results": "📊 Results",
        "tab_overlay": "🔀 Shape Overlay",
        "tab_heatmap": "🗺️ Heatmap",
        "tab_anomalies": "⚠️ Anomalies",
        "query_subheader": "Reference query from “{source_id}”",
        "full_series": "Full series",
        "metric_start": "Start",
        "metric_end": "End",
        "metric_points": "Points",
        "source_not_found": "source series was not found among loaded series",
        "overlap_spinner": "Detecting anomalies and computing overlap...",
        "overlap_progress_start": "Preparing anomaly overlap search...",
        "overlap_progress": "Detecting anomalies {current}/{total}: {series}",
        "query_no_intervals": "The detector found no anomaly intervals in the query window.",
        "overlap_only_query": "Overlap was found only for the query series; candidates did not pass thresholds.",
        "overlap_failed": "Could not run anomaly overlap search: {error}",
        "no_targets": "No series to search. Enable source search or upload more CSV files.",
        "searching": "Searching...",
        "distance_failed": "Could not compute distances for {series_id}: {error}",
        "no_similar": "No similar windows were found for the selected parameters.",
        "press_search_overlap": "Click “Search” to compute anomaly overlap results",
        "press_search": "Click “Search” to get results",
        "query_intervals": "Query intervals",
        "matched_series": "Matched series",
        "best_score": "Best score",
        "threshold": "Threshold",
        "overlap_subheader": "Anomaly interval overlap",
        "candidates_no_threshold": "No candidates passed the selected overlap thresholds",
        "no_results": "No results for the selected parameters",
        "similar_top": "Top-{count} similar segments",
        "result_select": "Select a result for detailed view",
        "found_segment": "Found segment",
        "no_overlay_series": "No series to show in the overlay",
        "matched_no_points": "Matched series have no points in the selected query time window",
        "matched_overlay_title": "Matched series in the selected query window",
        "shape_compare": "Shape comparison (z-normalized)",
        "overlay_count": "Segments shown",
        "no_candidates_chart": "No candidate series for this chart",
        "overlap_score_title": "Overlap score by candidate series",
        "min_distance": "Minimum distance by series",
        "choose_overlap_type": "Choose search type “Anomaly overlap”",
        "series_detail": "Series detail",
        "intervals": "Intervals",
        "no_anomaly_intervals": "This series has no anomaly intervals in the selected query window",
        "error_no_timestamp": "missing timestamp column",
        "error_parse_timestamp": "could not parse timestamp",
        "error_no_numeric": "no numeric value column",
        "error_empty_numeric": "numeric columns contain no values",
        "error_empty_source": "source series is empty",
    },
    "RU": {
        "data_header": "📁 Данные",
        "upload_csv": "Загрузите CSV файлы",
        "loaded_series": "Загружено рядов: {count}",
        "file_col": "Ряд",
        "points_col": "Точек",
        "start_col": "Начало",
        "end_col": "Конец",
        "read_failed": "Не удалось прочитать:\n{errors}",
        "params_header": "🎯 Параметры поиска",
        "source_series": "Source ряд (откуда берём query)",
        "window_len": "Длина окна (точек)",
        "query_segment": "Query отрезок",
        "search_type": "Тип поиска",
        "top_k": "Top-K результатов",
        "search_in_source": "Искать и в source ряде",
        "search_scope": "Область поиска",
        "overlap_caption": "Для overlap сравниваются аномальные интервалы внутри выбранного query-времени.",
        "overlap_metric": "Overlap метрика",
        "overlap_threshold": "Порог overlap",
        "min_recall": "Минимальный recall",
        "max_gap": "Склеивать разрывы до N точек",
        "min_points": "Мин. точек в интервале",
        "detector": "Детектор аномалий",
        "search_button": "🔍 Искать",
        "upload_prompt": "⬅️ Загрузите CSV файлы в sidebar",
        "select_nonempty_source": "Выберите непустой source ряд",
        "window_too_long": "Длина окна ({window}) больше длины source ряда ({length} точек)",
        "window_too_short": "Для поиска похожих окон нужно минимум 3 точки в окне",
        "tab_query": "🎯 Query",
        "tab_results": "📊 Результаты",
        "tab_overlay": "🔀 Наложение форм",
        "tab_heatmap": "🗺️ Heatmap",
        "tab_anomalies": "⚠️ Аномалии",
        "query_subheader": "Эталонный query из «{source_id}»",
        "full_series": "Полный ряд",
        "metric_start": "Начало",
        "metric_end": "Конец",
        "metric_points": "Точек",
        "source_not_found": "source ряд не найден среди загруженных рядов",
        "overlap_spinner": "Детектирую аномалии и считаю overlap...",
        "overlap_progress_start": "Подготовка overlap-поиска...",
        "overlap_progress": "Детектирую аномалии {current}/{total}: {series}",
        "query_no_intervals": "В query-окне детектор не нашёл аномальных интервалов.",
        "overlap_only_query": "Overlap найден только для query-ряда; кандидаты не прошли пороги.",
        "overlap_failed": "Не удалось выполнить overlap-поиск: {error}",
        "no_targets": "Нет рядов для поиска. Включите поиск в source ряде или загрузите другие CSV.",
        "searching": "Поиск...",
        "distance_failed": "Не удалось посчитать расстояния для {series_id}: {error}",
        "no_similar": "Похожих окон не найдено для выбранных параметров.",
        "press_search_overlap": "Нажмите «Искать» для получения overlap-результатов",
        "press_search": "Нажмите «Искать» для получения результатов",
        "query_intervals": "Query интервалы",
        "matched_series": "Matched рядов",
        "best_score": "Лучший score",
        "threshold": "Порог",
        "overlap_subheader": "Overlap по аномальным интервалам",
        "candidates_no_threshold": "Кандидаты не прошли выбранные пороги overlap",
        "no_results": "Для выбранных параметров результатов нет",
        "similar_top": "Top-{count} похожих сегментов",
        "result_select": "Выберите результат для детального просмотра",
        "found_segment": "Найденный сегмент",
        "no_overlay_series": "Нет рядов для overlay",
        "matched_no_points": "У matched-рядов нет точек в выбранном query-времени",
        "matched_overlay_title": "Matched ряды в выбранном query-окне",
        "shape_compare": "Сравнение форм (z-normalized)",
        "overlay_count": "Количество сегментов на графике",
        "no_candidates_chart": "Нет candidate-рядов для графика",
        "overlap_score_title": "Overlap score по candidate-рядам",
        "min_distance": "Минимальное расстояние по рядам",
        "choose_overlap_type": "Выберите тип поиска «Overlap аномалий»",
        "series_detail": "Ряд для детального просмотра",
        "intervals": "Интервалов",
        "no_anomaly_intervals": "В выбранном query-окне у этого ряда нет аномальных интервалов",
        "error_no_timestamp": "нет колонки timestamp",
        "error_parse_timestamp": "не удалось распарсить timestamp",
        "error_no_numeric": "нет числовой колонки со значениями",
        "error_empty_numeric": "числовые колонки не содержат значений",
        "error_empty_source": "source ряд пустой",
    },
}

OPTION_TEXT = {
    "EN": {
        QUERY_AUTO: "Auto",
        QUERY_MINIMAP: "Minimap",
        SEARCH_SIMILARITY: "Shape similarity",
        SEARCH_ANOMALY_OVERLAP: "Anomaly overlap",
        SEARCH_EVERYWHERE: "Entire series",
        SEARCH_SAME_TIME: "Same time window",
    },
    "RU": {
        QUERY_AUTO: "Авто",
        QUERY_MINIMAP: "Миникарта",
        SEARCH_SIMILARITY: "Похожая форма",
        SEARCH_ANOMALY_OVERLAP: "Overlap аномалий",
        SEARCH_EVERYWHERE: "По всему ряду",
        SEARCH_SAME_TIME: "В том же времени",
    },
}


def language():
    return st.session_state.get("language", "EN")


def t(key, **kwargs):
    text = UI_TEXT.get(language(), UI_TEXT["EN"]).get(key, UI_TEXT["EN"].get(key, key))
    return text.format(**kwargs)


def option_label(option):
    return OPTION_TEXT.get(language(), OPTION_TEXT["EN"]).get(option, option)


def safe_default(value, options, default):
    return value if value in options else default


def empty_results_df():
    return pd.DataFrame(columns=RESULT_COLUMNS)


def unique_series_name(base_name, existing_names):
    if base_name not in existing_names:
        return base_name

    suffix = 2
    while f"{base_name} ({suffix})" in existing_names:
        suffix += 1
    return f"{base_name} ({suffix})"


def value_columns_from_frame(df):
    numeric_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if str(column).lower() not in METADATA_COLUMNS
    ]
    value_columns = [
        column
        for column in numeric_columns
        if str(column).lower() == "value" or str(column).lower().startswith("value_")
    ]
    return value_columns or numeric_columns


def load_series_from_csv(file_obj, filename):
    file_obj.seek(0)
    df = pd.read_csv(file_obj)

    if "timestamp" not in df.columns:
        raise ValueError(t("error_no_timestamp"))

    numeric_timestamp = pd.to_numeric(df["timestamp"], errors="coerce")
    if numeric_timestamp.notna().mean() >= 0.8:
        timestamp = pd.to_datetime(numeric_timestamp, unit="ms", errors="coerce")
    else:
        timestamp = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.assign(timestamp=timestamp).dropna(subset=["timestamp"])
    if df.empty:
        raise ValueError(t("error_parse_timestamp"))

    df = df.set_index("timestamp").sort_index()
    value_columns = value_columns_from_frame(df)
    if len(value_columns) == 0:
        raise ValueError(t("error_no_numeric"))

    loaded = {}
    file_stem = Path(filename).stem
    for column in value_columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue

        series_name = filename if len(value_columns) == 1 else f"{file_stem}:{column}"
        loaded[series_name] = series.rename("value")

    if not loaded:
        raise ValueError(t("error_empty_numeric"))

    return loaded


def z_norm(arr):
    arr = np.asarray(arr, dtype=float)
    s = np.nanstd(arr)
    if s == 0 or not np.isfinite(s):
        return arr * 0
    return (arr - np.nanmean(arr)) / s


def select_query_by_max_std(series, window_size):
    if series is None or series.empty:
        raise ValueError(t("error_empty_source"))

    window_size = min(int(window_size), len(series))
    rolling_std = series.rolling(window_size).std()
    if rolling_std.isna().all():
        return 0

    end_pos = int(np.nanargmax(rolling_std.values))
    start_pos = max(end_pos - window_size + 1, 0)
    return start_pos


def find_similar(query_values, target_values, k=5, exclusion=None):
    import stumpy

    query_values = np.asarray(query_values, dtype=np.float64)
    target_values = np.asarray(target_values, dtype=np.float64)

    if len(query_values) == 0 or len(query_values) > len(target_values):
        return [], []

    distances = stumpy.mass(query_values, target_values)
    sorted_idx = [idx for idx in np.argsort(distances) if np.isfinite(distances[idx])]

    if exclusion is None:
        exclusion = len(query_values)

    filtered = []
    for idx in sorted_idx:
        if all(abs(idx - sel) >= exclusion for sel in filtered):
            filtered.append(int(idx))
            if len(filtered) == k:
                break

    return filtered, [float(distances[i]) for i in filtered]


def search_slice_for_query_window(series, start_time, end_time):
    window = series.loc[start_time:end_time]
    if window.empty:
        return window, 0

    offset = int(series.index.searchsorted(window.index[0], side="left"))
    return window, offset


def source_signature(source_id, series):
    return (
        source_id,
        len(series),
        str(series.index[0]),
        str(series.index[-1]),
    )


def clamp_query_state(series, start_idx, window_size):
    window_size = max(3, min(int(window_size), len(series)))
    max_start = max(0, len(series) - window_size)
    start_idx = max(0, min(int(start_idx), max_start))
    return start_idx, window_size


def selected_bounds_from_plotly(event):
    selection = event.get("selection", {})
    points = selection.get("points", [])
    indices = []

    for point in points:
        customdata = point.get("customdata")
        if isinstance(customdata, list) and customdata:
            indices.append(int(customdata[0]))
        elif customdata is not None:
            indices.append(int(customdata))
        elif "point_index" in point:
            indices.append(int(point["point_index"]))
        elif "point_number" in point:
            indices.append(int(point["point_number"]))

    if not indices:
        return None

    return min(indices), max(indices)


def infer_alignment_tolerance(index):
    index = pd.DatetimeIndex(index)
    if len(index) < 2:
        return None

    diffs = np.diff(index.asi8)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None

    return pd.Timedelta(int(np.median(diffs) // 2), unit="ns")


def anomaly_points(series, detection, start_time, end_time, detection_index=None):
    if detection_index is None:
        detection_index = series.index

    detection_mask = np.asarray(detection.is_anomaly, dtype=bool).reshape(-1)
    detection_index = pd.DatetimeIndex(detection_index)
    if len(detection_mask) != len(detection_index):
        if len(detection_mask) == len(series.index):
            detection_index = pd.DatetimeIndex(series.index)
        else:
            return series.iloc[0:0]

    mask = pd.Series(detection_mask, index=detection_index).sort_index()
    values = series.loc[start_time:end_time]
    if values.empty:
        return values
    return values[mask.loc[start_time:end_time].reindex(values.index, fill_value=False)]


def prepare_overlap_series(series):
    prepared = pd.to_numeric(series, errors="coerce").dropna()
    if not isinstance(prepared.index, pd.DatetimeIndex):
        prepared.index = pd.to_datetime(prepared.index)

    prepared = prepared[~prepared.index.isna()].sort_index()
    return prepared[~prepared.index.duplicated(keep="last")]


def add_interval_spans(fig, intervals, *, color, opacity=0.18):
    for interval in intervals:
        fig.add_vrect(
            x0=interval.start,
            x1=interval.end,
            fillcolor=color,
            opacity=opacity,
            line_width=0,
        )


def overlap_color(match):
    if match.is_query:
        return "#d62728"
    if match.matched:
        return "#2ca02c"
    return "#7f7f7f"


def has_fresh_results(search_signature):
    return (
        "results_df" in st.session_state
        and st.session_state.get("search_signature") == search_signature
    )


def has_fresh_overlap(search_signature):
    return (
        "overlap_result" in st.session_state
        and st.session_state.get("overlap_signature") == search_signature
    )

with st.sidebar:
    lang_col, _ = st.columns([1, 3])
    with lang_col:
        selected_language = st.segmented_control(
            "Language",
            options=["EN", "RU"],
            default=safe_default(st.session_state.get("language", "EN"), ["EN", "RU"], "EN"),
            label_visibility="collapsed",
            width="content",
        )
        st.session_state["language"] = selected_language or "EN"

    st.header(t("data_header"))
    uploaded_files = st.file_uploader(
        t("upload_csv"),
        type="csv",
        accept_multiple_files=True
    )

    series_dict = {}
    series_errors = []
    if uploaded_files:
        for f in uploaded_files:
            try:
                loaded_series = load_series_from_csv(f, f.name)
                for name, series in loaded_series.items():
                    unique_name = unique_series_name(name, series_dict)
                    series_dict[unique_name] = series
            except Exception as exc:
                series_errors.append(f"{f.name}: {exc}")

        if series_dict:
            st.success(t("loaded_series", count=len(series_dict)))

            summary = pd.DataFrame({
                t("file_col"): list(series_dict.keys()),
                t("points_col"): [len(s) for s in series_dict.values()],
                t("start_col"): [s.index.min() for s in series_dict.values()],
                t("end_col"): [s.index.max() for s in series_dict.values()],
            })
            st.dataframe(summary, width='stretch')

        if series_errors:
            st.warning(t("read_failed", errors="\n".join(f"- {e}" for e in series_errors)))

    st.divider()
    st.header(t("params_header"))

    source_options = list(series_dict.keys())
    source_id = st.selectbox(
        t("source_series"),
        options=source_options if source_options else ["—"],
        disabled=not source_options,
    )

    source_series_for_limits = series_dict.get(source_id)
    if source_series_for_limits is None:
        window_size = st.slider(
            t("window_len"),
            min_value=10,
            max_value=5000,
            value=200,
            step=10,
            disabled=True,
        )
        query_mode = QUERY_AUTO
        search_scope = SEARCH_EVERYWHERE
    else:
        max_window = max(1, min(5000, len(source_series_for_limits)))
        min_window = 10 if max_window >= 10 else 1
        default_window = min(200, max_window)
        default_window = max(min_window, default_window)
        current_source_signature = source_signature(source_id, source_series_for_limits)

        if st.session_state.get("query_source_signature") != current_source_signature:
            st.session_state["query_source_signature"] = current_source_signature
            st.session_state["query_window_size"] = default_window
            st.session_state["query_start_idx"] = select_query_by_max_std(
                source_series_for_limits,
                default_window,
            )
            st.session_state["query_mode_value"] = QUERY_AUTO

        stored_window_size = st.session_state.get("query_window_size", default_window)
        stored_query_start = st.session_state.get("query_start_idx", 0)
        stored_query_start, stored_window_size = clamp_query_state(
            source_series_for_limits,
            stored_query_start,
            stored_window_size,
        )
        stored_window_size = max(min_window, min(stored_window_size, max_window))

        if min_window == max_window:
            window_size = int(st.number_input(
                t("window_len"),
                min_value=min_window,
                max_value=max_window,
                value=stored_window_size,
                disabled=True,
            ))
        else:
            window_size = st.slider(
                t("window_len"),
                min_value=min_window,
                max_value=max_window,
                value=stored_window_size,
                step=1,
            )
        st.session_state["query_window_size"] = int(window_size)

        max_query_start = max(0, len(source_series_for_limits) - int(window_size))
        st.session_state["query_start_idx"] = max(
            0,
            min(stored_query_start, max_query_start),
        )

        query_mode = st.segmented_control(
            t("query_segment"),
            options=[QUERY_AUTO, QUERY_MINIMAP],
            default=safe_default(
                st.session_state.get("query_mode_value", QUERY_AUTO),
                [QUERY_AUTO, QUERY_MINIMAP],
                QUERY_AUTO,
            ),
            format_func=option_label,
            width="stretch",
        )
        query_mode = query_mode or QUERY_AUTO
        st.session_state["query_mode_value"] = query_mode

        if query_mode == QUERY_AUTO:
            selected_start = select_query_by_max_std(source_series_for_limits, window_size)
        else:
            selected_start = st.session_state["query_start_idx"]
        selected_end = min(selected_start + int(window_size), len(source_series_for_limits)) - 1
        selected_query = source_series_for_limits.iloc[selected_start:selected_end + 1]
        st.caption(
            f"{str(selected_query.index[0])[:19]} → "
            f"{str(selected_query.index[-1])[:19]}"
        )

    if source_series_for_limits is None:
        query_mode = QUERY_AUTO
    search_kind = st.segmented_control(
        t("search_type"),
        options=[SEARCH_SIMILARITY, SEARCH_ANOMALY_OVERLAP],
        default=safe_default(
            st.session_state.get("search_kind_value", SEARCH_SIMILARITY),
            [SEARCH_SIMILARITY, SEARCH_ANOMALY_OVERLAP],
            SEARCH_SIMILARITY,
        ),
        format_func=option_label,
        width="stretch",
        disabled=source_series_for_limits is None,
    )
    search_kind = search_kind or SEARCH_SIMILARITY
    st.session_state["search_kind_value"] = search_kind

    top_k = st.slider(t("top_k"), 1, 20, 5)

    if search_kind == SEARCH_SIMILARITY:
        search_in_source = st.checkbox(t("search_in_source"), value=False)

        search_scope = st.segmented_control(
            t("search_scope"),
            options=[SEARCH_EVERYWHERE, SEARCH_SAME_TIME],
            default=safe_default(
                st.session_state.get("search_scope_value", SEARCH_EVERYWHERE),
                [SEARCH_EVERYWHERE, SEARCH_SAME_TIME],
                SEARCH_EVERYWHERE,
            ),
            format_func=option_label,
            width="stretch",
            disabled=source_series_for_limits is None,
        )
        search_scope = search_scope or SEARCH_EVERYWHERE
        st.session_state["search_scope_value"] = search_scope
        overlap_metric = "f1"
        overlap_threshold = 0.5
        min_recall = 0.0
        max_gap = 0
        min_points = 1
        detector_order = 20
        detector_threshold = 3.0
    else:
        search_in_source = False
        search_scope = SEARCH_SAME_TIME
        st.caption(t("overlap_caption"))

        overlap_metric = st.selectbox(
            t("overlap_metric"),
            options=["f1", "iou", "recall", "precision"],
            index=1,
        )
        overlap_threshold = st.slider(
            t("overlap_threshold"),
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
        min_recall = st.slider(
            t("min_recall"),
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
        )
        max_gap = st.slider(t("max_gap"), 0, 50, 3)
        min_points = st.slider(t("min_points"), 1, 50, 3)

        with st.expander(t("detector"), expanded=False):
            detector_order = st.slider("AR order", 1, 100, 20)
            detector_threshold = st.slider(
                "Threshold",
                min_value=0.5,
                max_value=10.0,
                value=3.0,
                step=0.1,
            )

    run_search = st.button(t("search_button"), type="primary", width='stretch')


if not series_dict:
    st.info(t("upload_prompt"))
    st.stop()

source_series = series_dict.get(source_id)
if source_series is None or source_series.empty:
    st.warning(t("select_nonempty_source"))
    st.stop()

if len(source_series) < window_size:
    st.error(
        t("window_too_long", window=window_size, length=len(source_series))
    )
    st.stop()

if window_size < 3:
    st.error(t("window_too_short"))
    st.stop()

search_only_query_window = search_scope == SEARCH_SAME_TIME
if query_mode == QUERY_AUTO:
    query_start = select_query_by_max_std(source_series, window_size)
    st.session_state["query_start_idx"] = query_start
else:
    query_start, window_size = clamp_query_state(
        source_series,
        st.session_state.get("query_start_idx", 0),
        window_size,
    )
    st.session_state["query_start_idx"] = query_start
    st.session_state["query_window_size"] = window_size

query_end = query_start + window_size
query = source_series.iloc[query_start:query_end]
query_time_start = query.index[0]
query_time_end = query.index[-1]
search_signature = (
    search_kind,
    source_id,
    int(window_size),
    query_mode,
    int(query_start),
    str(query_time_start),
    str(query_time_end),
    int(top_k),
    bool(search_in_source),
    bool(search_only_query_window),
    overlap_metric,
    float(overlap_threshold),
    float(min_recall),
    int(max_gap),
    int(min_points),
    int(detector_order),
    float(detector_threshold),
    tuple(
        (name, len(series), str(series.index[0]), str(series.index[-1]))
        for name, series in series_dict.items()
    ),
)

tab_query, tab_results, tab_overlay, tab_heatmap, tab_anomalies = st.tabs([
    t("tab_query"), t("tab_results"), t("tab_overlay"), t("tab_heatmap"), t("tab_anomalies")
])

with tab_query:
    st.subheader(t("query_subheader", source_id=source_id))

    mini_fig = go.Figure()
    point_ids = np.arange(len(source_series))
    mini_fig.add_trace(go.Scatter(
        x=source_series.index,
        y=source_series.values,
        customdata=point_ids,
        mode="lines",
        name="Source",
        line=dict(color="lightsteelblue", width=1),
    ))
    mini_fig.add_trace(go.Scatter(
        x=source_series.index,
        y=source_series.values,
        customdata=point_ids,
        mode="markers",
        name="Selection points",
        marker=dict(color="steelblue", size=6, opacity=0.001),
        hoverinfo="skip",
        showlegend=False,
    ))
    mini_fig.add_vrect(
        x0=query_time_start,
        x1=query_time_end,
        fillcolor="orange",
        opacity=0.2,
        line_width=0,
    )
    mini_fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=10, b=20),
        showlegend=False,
        dragmode="select",
        selectdirection="h",
        xaxis_title=None,
        yaxis_title=None,
    )
    mini_fig.update_yaxes(visible=False)
    minimap_event = st.plotly_chart(
        mini_fig,
        key="query_minimap",
        on_select="rerun",
        selection_mode="box",
        width='stretch',
        config={"displaylogo": False},
    )
    selected_bounds = selected_bounds_from_plotly(minimap_event)
    if selected_bounds is not None:
        selected_start, selected_end = selected_bounds
        selected_window_size = selected_end - selected_start + 1
        selected_start, selected_window_size = clamp_query_state(
            source_series,
            selected_start,
            selected_window_size,
        )
        if (
            query_mode != QUERY_MINIMAP
            or selected_start != st.session_state.get("query_start_idx")
            or selected_window_size != st.session_state.get("query_window_size")
        ):
            st.session_state["query_start_idx"] = selected_start
            st.session_state["query_window_size"] = selected_window_size
            st.session_state["query_mode_value"] = QUERY_MINIMAP
            st.rerun()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=source_series.index, y=source_series.values,
        mode="lines", name=t("full_series"),
        line=dict(color="lightsteelblue", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=query.index, y=query.values,
        mode="lines", name="Query",
        line=dict(color="crimson", width=3)
    ))
    fig.add_vrect(
        x0=query.index[0], x1=query.index[-1],
        fillcolor="orange", opacity=0.15, line_width=0
    )
    fig.update_layout(height=400, xaxis_title="Time", yaxis_title="Value")
    st.plotly_chart(fig, width='stretch')

    col1, col2, col3 = st.columns(3)
    col1.metric(t("metric_start"), str(query.index[0])[:19])
    col2.metric(t("metric_end"), str(query.index[-1])[:19])
    col3.metric(t("metric_points"), f"{len(query)}")


if run_search:
    if search_kind == SEARCH_ANOMALY_OVERLAP:
        try:
            overlap_series_dict = {
                name: prepare_overlap_series(series)
                for name, series in series_dict.items()
                if not series.empty
            }
            overlap_series_dict = {
                name: series
                for name, series in overlap_series_dict.items()
                if not series.empty
            }
            if source_id not in overlap_series_dict:
                raise ValueError(t("source_not_found"))

            detector = AnomalyDetectionSystem(
                detection_model_params={
                    "model_name": "Autoregressive Dev",
                    "order": int(detector_order),
                    "threshold": float(detector_threshold),
                    "stable": True,
                }
            )
            required_metrics = {"recall": float(min_recall)} if min_recall > 0 else None
            searcher = AnomalyIntervalOverlapSearch(
                detector,
                max_gap=int(max_gap),
                min_points=int(min_points),
                metric=overlap_metric,
                overlap_threshold=float(overlap_threshold),
                required_metrics=required_metrics,
            )

            alignment_tolerance = infer_alignment_tolerance(query.index)
            overlap_progress = st.progress(0, text=t("overlap_progress_start"))

            def update_overlap_progress(current, total, series_name):
                progress_value = current / total if total else 1.0
                overlap_progress.progress(
                    progress_value,
                    text=t(
                        "overlap_progress",
                        current=current,
                        total=total,
                        series=series_name,
                    ),
                )

            try:
                overlap_result = searcher.search_series_dict(
                    overlap_series_dict,
                    query_series=source_id,
                    start_time=query_time_start,
                    end_time=query_time_end,
                    alignment_tolerance=alignment_tolerance,
                    progress_callback=update_overlap_progress,
                )
            finally:
                overlap_progress.empty()

            overlap_df = overlap_result.to_frame(include_query=True)
            st.session_state["overlap_result"] = overlap_result
            st.session_state["overlap_df"] = overlap_df
            st.session_state["overlap_series_dict"] = overlap_series_dict
            st.session_state["overlap_alignment_tolerance"] = alignment_tolerance
            st.session_state["overlap_signature"] = search_signature

            if not overlap_result.query_intervals:
                st.warning(t("query_no_intervals"))
            elif not overlap_result.matching_series:
                st.warning(t("overlap_only_query"))
        except Exception as exc:
            st.session_state["overlap_result"] = None
            st.session_state["overlap_df"] = pd.DataFrame()
            st.session_state["overlap_series_dict"] = {}
            st.session_state["overlap_alignment_tolerance"] = None
            st.session_state["overlap_signature"] = search_signature
            st.error(t("overlap_failed", error=exc))
    else:
        all_results = []

        target_ids = list(series_dict.keys())
        if not search_in_source:
            target_ids = [k for k in target_ids if k != source_id]

        if not target_ids:
            st.warning(t("no_targets"))

        progress = None
        if target_ids:
            progress = st.progress(0, text=t("searching"))
            for i, tid in enumerate(target_ids):
                target = series_dict[tid]
                target_for_search = target
                target_offset = 0
                if search_only_query_window:
                    target_for_search, target_offset = search_slice_for_query_window(
                        target, query_time_start, query_time_end
                    )

                if len(target_for_search) >= window_size:
                    try:
                        idxs, dists = find_similar(
                            query.values, target_for_search.values, k=top_k
                        )
                    except Exception as exc:
                        st.warning(t("distance_failed", series_id=tid, error=exc))
                        idxs, dists = [], []

                    for idx, dist in zip(idxs, dists):
                        target_idx = target_offset + idx
                        all_results.append({
                            "series_id": tid,
                            "start_idx": target_idx,
                            "end_idx": target_idx + window_size,
                            "distance": dist,
                            "start_time": target.index[target_idx],
                            "end_time": target.index[min(target_idx + window_size - 1, len(target) - 1)],
                        })
                progress.progress((i + 1) / len(target_ids), text=t("searching"))

        if all_results:
            results_df = (
                pd.DataFrame(all_results)
                .sort_values("distance")
                .head(top_k)
                .reset_index(drop=True)
            )
        else:
            results_df = empty_results_df()
            if target_ids:
                st.warning(t("no_similar"))

        st.session_state["results_df"] = results_df
        st.session_state["query_values"] = query.values.copy()
        st.session_state["search_signature"] = search_signature
        if progress is not None:
            progress.empty()


with tab_results:
    if search_kind == SEARCH_ANOMALY_OVERLAP:
        if not has_fresh_overlap(search_signature) or st.session_state.get("overlap_result") is None:
            st.info(t("press_search_overlap"))
        else:
            overlap_result = st.session_state["overlap_result"]
            overlap_df = st.session_state["overlap_df"]
            matched_df = overlap_result.to_frame(matched_only=True)
            candidate_df = overlap_df[~overlap_df["is_query"]].copy()
            best_score = candidate_df["score"].max() if not candidate_df.empty else 0.0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(t("query_intervals"), len(overlap_result.query_intervals))
            col2.metric(t("matched_series"), len(overlap_result.matching_series))
            col3.metric(t("best_score"), f"{best_score:.3f}")
            col4.metric(t("threshold"), f"{overlap_result.overlap_threshold:.2f}")

            st.subheader(t("overlap_subheader"))
            table = overlap_df.copy()
            table = table.rename(columns={"series_name": "series_id"})
            st.dataframe(
                table[
                    [
                        "series_id",
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
                        "n_intervals",
                        "intervals",
                    ]
                ],
                width='stretch',
            )

            if matched_df.empty:
                st.info(t("candidates_no_threshold"))
    elif not has_fresh_results(search_signature):
        st.info(t("press_search"))
    else:
        results_df = st.session_state["results_df"]
        if results_df.empty:
            st.info(t("no_results"))
        else:
            st.subheader(t("similar_top", count=len(results_df)))
            st.dataframe(
                results_df[["series_id", "distance", "start_time", "end_time"]],
                width='stretch'
            )

            selected_rank = st.selectbox(
                t("result_select"),
                options=range(len(results_df)),
                format_func=lambda i: (
                    f"#{i+1}: {results_df.iloc[i]['series_id']} "
                    f"(d={results_df.iloc[i]['distance']:.4f})"
                )
            )

            row = results_df.iloc[selected_rank]
            target_series = series_dict[row["series_id"]]
            segment = target_series.iloc[int(row["start_idx"]):int(row["end_idx"])]

            context_pad = pd.Timedelta(days=2)
            ctx = target_series.loc[
                row["start_time"] - context_pad : row["end_time"] + context_pad
            ]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ctx.index, y=ctx.values,
                mode="lines", name=row["series_id"],
                line=dict(color="gray", width=1)
            ))
            fig.add_trace(go.Scatter(
                x=segment.index, y=segment.values,
                mode="lines", name=t("found_segment"),
                line=dict(color="green", width=3)
            ))
            fig.add_vrect(
                x0=row["start_time"], x1=row["end_time"],
                fillcolor="green", opacity=0.1, line_width=0
            )
            fig.update_layout(
                height=350,
                title=f"#{selected_rank+1}: {row['series_id']} — distance={row['distance']:.4f}"
            )
            st.plotly_chart(fig, width='stretch')


with tab_overlay:
    if search_kind == SEARCH_ANOMALY_OVERLAP:
        if not has_fresh_overlap(search_signature) or st.session_state.get("overlap_result") is None:
            st.info(t("press_search_overlap"))
        else:
            overlap_result = st.session_state["overlap_result"]
            overlap_series_dict = st.session_state["overlap_series_dict"]
            matched_columns = [source_id, *overlap_result.matching_series]
            matched_columns = [column for column in matched_columns if column in overlap_series_dict]

            if not matched_columns:
                st.info(t("no_overlay_series"))
            else:
                fig = go.Figure()
                colors = ["crimson", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b"]
                plotted = 0
                for i, column in enumerate(matched_columns):
                    window = overlap_series_dict[column].loc[query_time_start:query_time_end]
                    if window.empty:
                        continue

                    normalized = pd.Series(
                        z_norm(window.values),
                        index=window.index,
                        name=column,
                    )
                    line_color = colors[i % len(colors)]
                    fig.add_trace(go.Scatter(
                        x=normalized.index,
                        y=normalized.values,
                        mode="lines",
                        name=column,
                        line=dict(
                            color=line_color,
                            width=3 if column == source_id else 1.6,
                        ),
                        opacity=1.0 if column == source_id else 0.82,
                    ))
                    points = anomaly_points(
                        normalized,
                        overlap_result.detections[column],
                        query_time_start,
                        query_time_end,
                        detection_index=overlap_series_dict[column].index,
                    )
                    if not points.empty:
                        fig.add_trace(go.Scatter(
                            x=points.index,
                            y=points.values,
                            mode="markers",
                            name=f"{column} anomalies",
                            marker=dict(color=line_color, size=8, line=dict(color="white", width=1)),
                            showlegend=False,
                        ))
                    plotted += 1

                if plotted == 0:
                    st.info(t("matched_no_points"))
                else:
                    add_interval_spans(fig, overlap_result.query_intervals, color="#d62728", opacity=0.12)
                    fig.update_layout(
                        height=520,
                        xaxis_title="Time",
                        yaxis_title="z-score",
                        title=t("matched_overlay_title"),
                    )
                    st.plotly_chart(fig, width='stretch')
    elif not has_fresh_results(search_signature):
        st.info(t("press_search"))
    else:
        results_df = st.session_state["results_df"]
        query_values = st.session_state["query_values"]

        if results_df.empty:
            st.info(t("no_results"))
        else:
            st.subheader(t("shape_compare"))

            n_overlay = st.slider(
                t("overlay_count"),
                1, len(results_df), min(5, len(results_df))
            )

            fig = go.Figure()
            x = np.arange(len(query_values))

            fig.add_trace(go.Scatter(
                x=x, y=z_norm(query_values),
                mode="lines", name=f"Query ({source_id})",
                line=dict(color="crimson", width=3)
            ))

            colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd",
                      "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
            for i, row in results_df.head(n_overlay).iterrows():
                seg = series_dict[row["series_id"]].iloc[
                    int(row["start_idx"]):int(row["end_idx"])
                ]
                fig.add_trace(go.Scatter(
                    x=x, y=z_norm(seg.values),
                    mode="lines",
                    name=f"#{i+1} {row['series_id']} (d={row['distance']:.3f})",
                    line=dict(width=1.5, color=colors[i % len(colors)]),
                    opacity=0.8
                ))

            fig.update_layout(
                height=500,
                xaxis_title="Position in window",
                yaxis_title="z-score"
            )
            st.plotly_chart(fig, width='stretch')


with tab_heatmap:
    if search_kind == SEARCH_ANOMALY_OVERLAP:
        if not has_fresh_overlap(search_signature) or st.session_state.get("overlap_result") is None:
            st.info(t("press_search_overlap"))
        else:
            overlap_df = st.session_state["overlap_df"]
            candidate_df = overlap_df[~overlap_df["is_query"]].copy()
            if candidate_df.empty:
                st.info(t("no_candidates_chart"))
            else:
                candidate_df = candidate_df.sort_values("score", ascending=True)
                fig = go.Figure(go.Bar(
                    x=candidate_df["score"],
                    y=candidate_df["series_name"],
                    orientation="h",
                    marker_color=[
                        "#2ca02c" if matched else "#9aa0a6"
                        for matched in candidate_df["matched"]
                    ],
                    customdata=np.column_stack([
                        candidate_df["recall"],
                        candidate_df["precision"],
                        candidate_df["iou"],
                        candidate_df["f1"],
                    ]),
                    hovertemplate=(
                        "%{y}<br>score=%{x:.3f}<br>"
                        "recall=%{customdata[0]:.3f}<br>"
                        "precision=%{customdata[1]:.3f}<br>"
                        "iou=%{customdata[2]:.3f}<br>"
                        "f1=%{customdata[3]:.3f}<extra></extra>"
                    ),
                ))
                fig.add_vline(
                    x=overlap_threshold,
                    line_dash="dash",
                    line_color="crimson",
                    opacity=0.8,
                )
                fig.update_layout(
                    height=max(320, len(candidate_df) * 38),
                    xaxis_title=f"{overlap_metric} score",
                    yaxis_title="Series",
                    title=t("overlap_score_title"),
                )
                st.plotly_chart(fig, width='stretch')
    elif not has_fresh_results(search_signature):
        st.info(t("press_search"))
    else:
        results_df = st.session_state["results_df"]
        if results_df.empty:
            st.info(t("no_results"))
        else:
            st.subheader(t("min_distance"))

            best_per_series = (
                results_df
                .groupby("series_id")["distance"]
                .min()
                .sort_values()
            )

            fig = go.Figure(go.Bar(
                x=best_per_series.values,
                y=best_per_series.index,
                orientation="h",
                marker_color=[
                    "green" if d < best_per_series.quantile(0.3)
                    else "orange" if d < best_per_series.quantile(0.7)
                    else "red"
                    for d in best_per_series.values
                ]
            ))
            fig.update_layout(
                height=max(300, len(best_per_series) * 35),
                xaxis_title="Distance",
                yaxis_title="Series",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, width='stretch')


with tab_anomalies:
    if search_kind != SEARCH_ANOMALY_OVERLAP:
        st.info(t("choose_overlap_type"))
    elif not has_fresh_overlap(search_signature) or st.session_state.get("overlap_result") is None:
        st.info(t("press_search_overlap"))
    else:
        overlap_result = st.session_state["overlap_result"]
        overlap_df = st.session_state["overlap_df"]
        overlap_series_dict = st.session_state["overlap_series_dict"]

        series_options = list(overlap_df["series_name"])
        selected_series = st.selectbox(
            t("series_detail"),
            options=series_options,
            format_func=lambda name: (
                f"{name} — query"
                if overlap_result.series_matches[name].is_query
                else (
                    f"{name} — match, score={overlap_result.series_matches[name].score:.3f}"
                    if overlap_result.series_matches[name].matched
                    else f"{name} — score={overlap_result.series_matches[name].score:.3f}"
                )
            ),
        )

        match = overlap_result.series_matches[selected_series]
        selected_series_values = overlap_series_dict[selected_series]
        window_delta = query_time_end - query_time_start
        context_pad = max(window_delta, pd.Timedelta(hours=1))
        context_start = query_time_start - context_pad
        context_end = query_time_end + context_pad
        context = selected_series_values.loc[context_start:context_end]
        points = anomaly_points(
            selected_series_values,
            overlap_result.detections[selected_series],
            context_start,
            context_end,
            detection_index=selected_series_values.index,
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Score", f"{match.score:.3f}")
        col2.metric("Recall", f"{match.metrics.recall:.3f}")
        col3.metric("IoU", f"{match.metrics.iou:.3f}")
        col4.metric(t("intervals"), len(match.intervals))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=context.index,
            y=context.values,
            mode="lines",
            name=selected_series,
            line=dict(color="#4d4d4d", width=1.4),
        ))
        fig.add_vrect(
            x0=query_time_start,
            x1=query_time_end,
            fillcolor="#f28e2b",
            opacity=0.10,
            line_width=0,
        )
        add_interval_spans(fig, match.intervals, color=overlap_color(match), opacity=0.20)
        if not points.empty:
            fig.add_trace(go.Scatter(
                x=points.index,
                y=points.values,
                mode="markers",
                name="anomaly points",
                marker=dict(
                    color=overlap_color(match),
                    size=9,
                    line=dict(color="white", width=1),
                ),
            ))

        title_status = "query" if match.is_query else "matched" if match.matched else "not matched"
        fig.update_layout(
            height=420,
            title=(
                f"{selected_series}: {title_status}, "
                f"score={match.score:.3f}, recall={match.metrics.recall:.3f}, IoU={match.metrics.iou:.3f}"
            ),
            xaxis_title="Time",
            yaxis_title="Value",
        )
        st.plotly_chart(fig, width='stretch')

        intervals_df = pd.DataFrame(
            [
                {
                    "start": interval.start,
                    "end": interval.end,
                    "start_idx": interval.start_idx,
                    "end_idx": interval.end_idx,
                    "anomaly_points": interval.n_anomaly_points,
                }
                for interval in match.intervals
            ]
        )
        if intervals_df.empty:
            st.info(t("no_anomaly_intervals"))
        else:
            st.dataframe(intervals_df, width='stretch')
