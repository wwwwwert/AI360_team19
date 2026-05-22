import subprocess
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def _running_inside_streamlit():
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    return get_script_run_ctx(suppress_warning=True) is not None


if __name__ == "__main__" and not _running_inside_streamlit():
    script_path = Path(__file__).resolve()
    cmd = [sys.executable, "-m", "streamlit", "run", str(script_path), *sys.argv[1:]]
    print("Запускаю Streamlit-приложение:", flush=True)
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
QUERY_AUTO = "Авто"
QUERY_MINIMAP = "Миникарта"
SEARCH_EVERYWHERE = "По всему ряду"
SEARCH_SAME_TIME = "В том же времени"


def empty_results_df():
    return pd.DataFrame(columns=RESULT_COLUMNS)


def load_series_from_csv(file_obj):
    file_obj.seek(0)
    df = pd.read_csv(file_obj)

    if "timestamp" not in df.columns:
        raise ValueError("нет колонки timestamp")

    numeric_timestamp = pd.to_numeric(df["timestamp"], errors="coerce")
    if numeric_timestamp.notna().mean() >= 0.8:
        timestamp = pd.to_datetime(numeric_timestamp, unit="ms", errors="coerce")
    else:
        timestamp = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.assign(timestamp=timestamp).dropna(subset=["timestamp"])
    if df.empty:
        raise ValueError("не удалось распарсить timestamp")

    df = df.set_index("timestamp").sort_index()
    numeric_columns = df.select_dtypes(include="number").columns
    if len(numeric_columns) == 0:
        raise ValueError("нет числовой колонки со значениями")

    series = pd.to_numeric(df[numeric_columns[0]], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"колонка {numeric_columns[0]} не содержит числовых значений")

    return series.rename("value")


def z_norm(arr):
    arr = np.asarray(arr, dtype=float)
    s = np.nanstd(arr)
    if s == 0 or not np.isfinite(s):
        return arr * 0
    return (arr - np.nanmean(arr)) / s


def select_query_by_max_std(series, window_size):
    """Выбираем окно с максимальной вариативностью."""
    if series is None or series.empty:
        raise ValueError("source ряд пустой")

    window_size = min(int(window_size), len(series))
    rolling_std = series.rolling(window_size).std()
    if rolling_std.isna().all():
        return 0

    end_pos = int(np.nanargmax(rolling_std.values))
    start_pos = max(end_pos - window_size + 1, 0)
    return start_pos


def find_similar(query_values, target_values, k=5, exclusion=None):
    """Поиск top-k похожих окон через stumpy.mass."""
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


def has_fresh_results(search_signature):
    return (
        "results_df" in st.session_state
        and st.session_state.get("search_signature") == search_signature
    )

# ──────────────────────────────────────────────
# SIDEBAR: загрузка и параметры
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Данные")
    uploaded_files = st.file_uploader(
        "Загрузите CSV файлы",
        type="csv",
        accept_multiple_files=True
    )

    # Парсим файлы
    series_dict = {}
    series_errors = []
    if uploaded_files:
        for f in uploaded_files:
            try:
                series_dict[f.name] = load_series_from_csv(f)
            except Exception as exc:
                series_errors.append(f"{f.name}: {exc}")

        if series_dict:
            st.success(f"Загружено рядов: {len(series_dict)}")

            # Превью: компактная таблица вместо 20 графиков
            summary = pd.DataFrame({
                "Файл": list(series_dict.keys()),
                "Точек": [len(s) for s in series_dict.values()],
                "Начало": [s.index.min() for s in series_dict.values()],
                "Конец": [s.index.max() for s in series_dict.values()],
            })
            st.dataframe(summary, use_container_width=True)

        if series_errors:
            st.warning("Не удалось прочитать:\n" + "\n".join(f"- {e}" for e in series_errors))

    st.divider()
    st.header("🎯 Параметры поиска")

    source_options = list(series_dict.keys())
    source_id = st.selectbox(
        "Source ряд (откуда берём query)",
        options=source_options if source_options else ["—"],
        disabled=not source_options,
    )

    source_series_for_limits = series_dict.get(source_id)
    if source_series_for_limits is None:
        window_size = st.slider(
            "Длина окна (точек)",
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
                "Длина окна (точек)",
                min_value=min_window,
                max_value=max_window,
                value=stored_window_size,
                disabled=True,
            ))
        else:
            window_size = st.slider(
                "Длина окна (точек)",
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
            "Query отрезок",
            options=[QUERY_AUTO, QUERY_MINIMAP],
            default=st.session_state.get("query_mode_value", QUERY_AUTO),
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
    top_k = st.slider("Top-K результатов", 1, 20, 5)

    search_in_source = st.checkbox("Искать и в source ряде", value=False)

    search_scope = st.segmented_control(
        "Область поиска",
        options=[SEARCH_EVERYWHERE, SEARCH_SAME_TIME],
        default=st.session_state.get("search_scope_value", SEARCH_EVERYWHERE),
        width="stretch",
        disabled=source_series_for_limits is None,
    )
    search_scope = search_scope or SEARCH_EVERYWHERE
    st.session_state["search_scope_value"] = search_scope

    run_search = st.button("🔍 Искать", type="primary", use_container_width=True)


# ──────────────────────────────────────────────
# MAIN AREA
# ──────────────────────────────────────────────
if not series_dict:
    st.info("⬅️ Загрузите CSV файлы в sidebar")
    st.stop()

source_series = series_dict.get(source_id)
if source_series is None or source_series.empty:
    st.warning("Выберите непустой source ряд")
    st.stop()

if len(source_series) < window_size:
    st.error(
        f"Длина окна ({window_size}) больше длины source ряда "
        f"({len(source_series)} точек)"
    )
    st.stop()

if window_size < 3:
    st.error("Для поиска похожих окон нужно минимум 3 точки в окне")
    st.stop()

# ── Выбор query ──
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
    source_id,
    int(window_size),
    query_mode,
    int(query_start),
    str(query_time_start),
    str(query_time_end),
    int(top_k),
    bool(search_in_source),
    bool(search_only_query_window),
    tuple((name, len(series)) for name, series in series_dict.items()),
)

# ── Табы ──
tab_query, tab_results, tab_overlay, tab_heatmap = st.tabs([
    "🎯 Query", "📊 Результаты", "🔀 Наложение форм", "🗺️ Heatmap"
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: Query — один график
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_query:
    st.subheader(f"Эталонный query из «{source_id}»")

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
        use_container_width=True,
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
        mode="lines", name="Полный ряд",
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
    st.plotly_chart(fig, use_container_width=True)

    # Метаданные query
    col1, col2, col3 = st.columns(3)
    col1.metric("Начало", str(query.index[0])[:19])
    col2.metric("Конец", str(query.index[-1])[:19])
    col3.metric("Точек", f"{len(query)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Поиск (если нажали кнопку)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if run_search:
    all_results = []

    target_ids = list(series_dict.keys())
    if not search_in_source:
        target_ids = [k for k in target_ids if k != source_id]

    if not target_ids:
        st.warning("Нет рядов для поиска. Включите поиск в source ряде или загрузите другие CSV.")

    progress = None
    if target_ids:
        progress = st.progress(0, text="Поиск...")
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
                    st.warning(f"Не удалось посчитать расстояния для {tid}: {exc}")
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
            progress.progress((i + 1) / len(target_ids), text="Поиск...")

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
            st.warning("Похожих окон не найдено для выбранных параметров.")

    # Сохраняем в session_state чтобы табы видели
    st.session_state["results_df"] = results_df
    st.session_state["query_values"] = query.values.copy()
    st.session_state["search_signature"] = search_signature
    if progress is not None:
        progress.empty()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: Результаты — карточки
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_results:
    if not has_fresh_results(search_signature):
        st.info("Нажмите «Искать» для получения результатов")
    else:
        results_df = st.session_state["results_df"]
        if results_df.empty:
            st.info("Для выбранных параметров результатов нет")
        else:
            st.subheader(f"Top-{len(results_df)} похожих сегментов")
            # Таблица-сводка
            st.dataframe(
                results_df[["series_id", "distance", "start_time", "end_time"]],
                use_container_width=True
            )

            # Навигация по результатам — selectbox вместо 20 графиков
            selected_rank = st.selectbox(
                "Выберите результат для детального просмотра",
                options=range(len(results_df)),
                format_func=lambda i: (
                    f"#{i+1}: {results_df.iloc[i]['series_id']} "
                    f"(d={results_df.iloc[i]['distance']:.4f})"
                )
            )

            row = results_df.iloc[selected_rank]
            target_series = series_dict[row["series_id"]]
            segment = target_series.iloc[int(row["start_idx"]):int(row["end_idx"])]

            # Контекст ±
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
                mode="lines", name="Найденный сегмент",
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
            st.plotly_chart(fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: Overlay — наложение форм
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_overlay:
    if not has_fresh_results(search_signature):
        st.info("Нажмите «Искать» для получения результатов")
    else:
        results_df = st.session_state["results_df"]
        query_values = st.session_state["query_values"]

        if results_df.empty:
            st.info("Для выбранных параметров результатов нет")
        else:
            st.subheader("Сравнение форм (z-normalized)")

            # Сколько показать на overlay
            n_overlay = st.slider(
                "Количество сегментов на графике",
                1, len(results_df), min(5, len(results_df))
            )

            fig = go.Figure()
            x = np.arange(len(query_values))

            # Query
            fig.add_trace(go.Scatter(
                x=x, y=z_norm(query_values),
                mode="lines", name=f"Query ({source_id})",
                line=dict(color="crimson", width=3)
            ))

            # Top-N результатов
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
            st.plotly_chart(fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: Distance heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_heatmap:
    if not has_fresh_results(search_signature):
        st.info("Нажмите «Искать» для получения результатов")
    else:
        results_df = st.session_state["results_df"]
        if results_df.empty:
            st.info("Для выбранных параметров результатов нет")
        else:
            st.subheader("Минимальное расстояние по рядам")

            # Агрегируем: лучший distance на каждый ряд
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
            st.plotly_chart(fig, use_container_width=True)
