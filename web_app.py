import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Time Series Similarity Search")

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
    if uploaded_files:
        for f in uploaded_files:
            df = pd.read_csv(f)
            # простой парсинг — адаптируйте под свой формат
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")
            # берём первую числовую колонку как value
            value_col = df.select_dtypes(include="number").columns[0]
            series_dict[f.name] = df[value_col].rename("value")

        st.success(f"Загружено рядов: {len(series_dict)}")

        # Превью: компактная таблица вместо 20 графиков
        summary = pd.DataFrame({
            "Файл": list(series_dict.keys()),
            "Точек": [len(s) for s in series_dict.values()],
            "Начало": [s.index.min() for s in series_dict.values()],
            "Конец": [s.index.max() for s in series_dict.values()],
        })
        st.dataframe(summary, use_container_width=True)

    st.divider()
    st.header("🎯 Параметры поиска")

    source_id = st.selectbox(
        "Source ряд (откуда берём query)",
        options=list(series_dict.keys()) if series_dict else ["—"]
    )

    window_size = st.slider(
        "Длина окна (точек)", 
        min_value=10, max_value=5000, value=200, step=10
    )

    top_k = st.slider("Top-K результатов", 1, 20, 5)

    search_in_source = st.checkbox("Искать и в source ряде", value=False)

    run_search = st.button("🔍 Искать", type="primary", use_container_width=True)


# ──────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────
def z_norm(arr):
    arr = np.asarray(arr, dtype=float)
    s = np.nanstd(arr)
    if s == 0 or not np.isfinite(s):
        return arr * 0
    return (arr - np.nanmean(arr)) / s


def select_query_by_max_std(series, window_size):
    """Выбираем окно с максимальной вариативностью."""
    rolling_std = series.rolling(window_size).std()
    if rolling_std.isna().all():
        return 0
    end_idx = int(rolling_std.idxmax().timestamp())  # или argmax
    # упрощённо:
    end_pos = np.nanargmax(rolling_std.values)
    start_pos = max(end_pos - window_size + 1, 0)
    return start_pos


def find_similar(query_values, target_values, k=5, exclusion=None):
    """Поиск top-k похожих окон через stumpy.mass."""
    import stumpy
    
    # Приводим к float64 для stumpy
    query_values = np.asarray(query_values, dtype=np.float64)
    target_values = np.asarray(target_values, dtype=np.float64)
    
    distances = stumpy.mass(query_values, target_values)
    sorted_idx = np.argsort(distances)

    if exclusion is None:
        exclusion = len(query_values)

    filtered = []
    for idx in sorted_idx:
        if all(abs(idx - sel) >= exclusion for sel in filtered):
            filtered.append(idx)
            if len(filtered) == k:
                break

    return filtered, [float(distances[i]) for i in filtered]


# ──────────────────────────────────────────────
# MAIN AREA
# ──────────────────────────────────────────────
if not series_dict:
    st.info("⬅️ Загрузите CSV файлы в sidebar")
    st.stop()

source_series = series_dict.get(source_id)
if source_series is None:
    st.stop()

# ── Выбор query ──
query_start = select_query_by_max_std(source_series, window_size)
query_end = query_start + window_size
query = source_series.iloc[query_start:query_end]

# ── Табы ──
tab_query, tab_results, tab_overlay, tab_heatmap = st.tabs([
    "🎯 Query", "📊 Результаты", "🔀 Наложение форм", "🗺️ Heatmap"
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: Query — один график
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_query:
    st.subheader(f"Эталонный query из «{source_id}»")

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
    col3.metric("Std", f"{query.std():.2f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Поиск (если нажали кнопку)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if run_search:
    all_results = []

    target_ids = list(series_dict.keys())
    if not search_in_source:
        target_ids = [k for k in target_ids if k != source_id]

    progress = st.progress(0, text="Поиск...")
    for i, tid in enumerate(target_ids):
        target = series_dict[tid]
        if len(target) < window_size:
            continue
        idxs, dists = find_similar(
            query.values, target.values, k=top_k
        )
        for idx, dist in zip(idxs, dists):
            all_results.append({
                "series_id": tid,
                "start_idx": idx,
                "end_idx": idx + window_size,
                "distance": dist,
                "start_time": target.index[idx],
                "end_time": target.index[min(idx + window_size - 1, len(target) - 1)],
            })
        progress.progress((i + 1) / len(target_ids))

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("distance").head(top_k).reset_index(drop=True)

    # Сохраняем в session_state чтобы табы видели
    st.session_state["results_df"] = results_df
    st.session_state["query_values"] = query.values
    progress.empty()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: Результаты — карточки
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_results:
    if "results_df" not in st.session_state:
        st.info("Нажмите «Искать» для получения результатов")
    else:
        results_df = st.session_state["results_df"]
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
    if "results_df" not in st.session_state:
        st.info("Нажмите «Искать» для получения результатов")
    else:
        results_df = st.session_state["results_df"]
        query_values = st.session_state["query_values"]

        st.subheader("Сравнение форм (z-normalized)")

        # Сколько показать на overlay
        n_overlay = st.slider(
            "Количество сегментов на графике",
            1, len(results_df), min(5, len(results_df))
        )

        fig = go.Figure()
        x = np.arange(window_size)

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
    if "results_df" not in st.session_state:
        st.info("Нажмите «Искать» для получения результатов")
    else:
        results_df = st.session_state["results_df"]
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
