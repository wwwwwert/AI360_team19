# AI360_team19


- [Презентация по адаптации сезонального алгоритма под праздники](https://docs.google.com/presentation/d/14tZaid9kj2LPkCkacO3Kdhbm7Kf3nvaSelr7HvWTTI8/edit?usp=sharing)
	- [kmes branch](https://github.com/wwwwwert/AI360_team19/tree/kmes)
	- [Report](https://github.com/wwwwwert/AI360_team19/blob/kmes/Report.md)
- [find_similar_demo.ipynb](https://github.com/wwwwwert/AI360_team19/blob/find-similar/find_similar_demo.ipynb) - Демонстрация класса `TimeSeriesSubsequenceSearcher`: загружаем CSV, приводим ряды к общей частоте, вырезаем query-отрезок по времени и визуально сравниваем его с найденными похожими отрезками в других рядах.
	- [find_similar.py](https://github.com/wwwwwert/AI360_team19/blob/find-similar/find_similar.py) - там класс `TimeSeriesSubsequenceSearcher`, который приводит набор временных рядов к единой регулярной временной сетке и находит в них подпоследовательности, наиболее похожие по форме на заданный эталонный фрагмент, используя `mass`
	- [ai](https://ai.yandex-team.ru/t/41b29456-1f5b-4876-bd5d-f5f29cc1db49)
- [Similarity_search.ipynb](https://github.com/wwwwwert/AI360_team19/blob/another_realization_of_similarity_search/Similarity_search.ipynb) - Демонстрация поиска похожих паттернов между двумя временными рядами с помощью `stumpy.mass`: вырезаем query-отрезок из первого ряда, вычисляем евклидово расстояние до всех окон второго ряда (`mass` вычисляет z-нормализованное евклидово расстояние через FFT), фильтруем перекрывающиеся совпадения и визуализируем top-5 найденных паттернов.
- [timeseries_clasterisation/notebook.ipynb](https://github.com/wwwwwert/AI360_team19/blob/clasterisation/timeseries_clasterisation/notebook.ipynb) - взяли датасет `GesturePebbleZ1_TRAIN`, применили `tslearn.clustering.TimeSeriesKMeans`
- [anomaly_interval_overlap_example.ipynb](https://github.com/wwwwwert/AI360_team19/blob/find_similar_anomalies/anomaly_interval_overlap_example.ipynb) - добавили класс `AnomalyIntervalOverlapSearch`. Детектор прогоняют на каждом временном ряде по-отдельности. 
	- [Report](https://github.com/wwwwwert/AI360_team19/blob/find_similar_anomalies/Report.md)
	- [ai](https://ai.yandex-team.ru/t/09f0e250-063e-4892-b512-15fac7ec0d18)
- [web_app.py](https://github.com/wwwwwert/AI360_team19/blob/streamlit_app/web_app.py)

***

Install environment: `uv sync`

Datasets should be placed like: `./data/Yahoo/...`, `./data/AIOPS/...`

Run benchmark:
```bash
uv run run_benchmark.py \
  --datasets "NAB, TODS, UCR, WSD, Yahoo" \
  --models models.json5
```

Run web app:
```bash
uv run streamlit run web_app.py
```
