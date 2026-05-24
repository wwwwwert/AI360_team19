# Отчет о реализации учета праздников в системе детекции аномалий

## Обзор проекта

Студенты успешно реализовали систему учета российских праздников в алгоритме детекции аномалий на временных рядах. Система основана на STL-декомпозиции (Seasonal Trend decomposition) и включает адаптивный механизм настройки доверительных интервалов для праздничных дней.

## Архитектура системы

### Базовые компоненты

1. **[`AnomalyDetectionSystem`](anomaly_detection/core/system.py)** - главный класс системы детекции
2. **[`BaseDetector`](anomaly_detection/models/base.py)** - базовый класс детекторов с поддержкой праздников
3. **[`STLDetector`](anomaly_detection/models/stl.py)** - основной алгоритм детекции на основе STL-декомпозиции
4. **[`TimeSeriesWrapper`](anomaly_detection/core/time_series.py)** - обертка для временных рядов

### Поддержка праздников

Система поддерживает два режима работы:
- **Без учета праздников** (`apply_holidays=False`) - стандартная детекция
- **С учетом праздников** (`apply_holidays=True`) - адаптивная детекция с расширенными доверительными интервалами

## Реализация учета праздников

### 1. Подготовка данных

#### Скрипт [`prepare_1c.py`](prepare_1c.py)
- Агрегирует данные продаж 1С в дневные временные ряды
- Добавляет колонку `holiday_mark` с кодами российских праздников
- Использует библиотеку `holidays` для определения праздничных дней
- Применяет окно `WINDOW_FACTOR` для расширения праздничного периода

```python
def mark_holidays(dates: pd.Series, window_factor: float) -> pd.Series:
    windows = get_holiday_windows(dates.min().year, dates.max().year, window_factor)
    marks = pd.Series("", index=dates.index)
    # Маркировка праздничных дней с расширенным окном
```

#### Скрипт [`expand_hourly.py`](expand_hourly.py)
- Расширяет дневные данные до почасовых с помощью кубической интерполяции
- Распространяет `holiday_mark` на все 24 часа каждого дня
- Создает гладкие переходы между точками данных

### 2. Алгоритм детекции с праздниками

#### Базовый детектор ([`BaseDetector`](anomaly_detection/models/base.py))

**Инициализация:**
```python
def __init__(self, apply_holidays=False, holiday_param=None, **kwargs):
    self._apply_holidays = apply_holidays
    if apply_holidays:
        self.holiday_param = holiday_param if holiday_param is not None else np.float64(2.0)
        self._holiday_lr = self.params.get("holiday_lr", 0.068)
        self._holiday_decay = 0.999
```

**Расчет стандартного отклонения с учетом праздников:**
```python
def calculate_std(self, residual: np.array, apply_holidays=False, dates=None, return_array=True):
    # Базовый расчет стандартного отклонения
    ans = _calculate_std(self, residual)
    
    if apply_holidays and hasattr(self, 'holiday_param') and dates is not None:
        hols = holidays.RU()  # Российские праздники
        std_array = np.full(len(residual), ans) if np.isscalar(ans) else ans.copy()
        
        # Увеличение стандартного отклонения для праздничных дней
        for i in range(len(dates)):
            if dates[i] in hols:
                std_array[i] *= self.holiday_param  # Умножение на параметр праздника
```

**Адаптивное обучение параметра праздника:**

Используется для экспериментов (поскольку на практике недоступна). Это нужно, чтобы проверить, какой на практике должен быть коэффициент. Вдруг он одинаковый на всех рядах

```python
def calculate_std_backward(self, dates: list, predicted: np.array, ground_truth: np.array):
    hols = holidays.RU()
    lr = self._holiday_lr
    self._holiday_lr *= self._holiday_decay  # Затухающая скорость обучения
    
    threshold = self.params.get("threshold", 3.0)
    missed = 0      # gt=1, pred<threshold → нужно сузить std → уменьшить param
    false_alarm = 0 # gt=0, pred>threshold → нужно расширить std → увеличить param
    
    # Анализ ошибок только на праздничных днях
    for i in range(len(dates)):
        if date_only not in hols:
            continue
        gt = ground_truth[i]
        pred_score = predicted[i]
        if gt == 1 and pred_score < threshold:
            missed += 1
        elif gt == 0 and pred_score > threshold:
            false_alarm += 1
    
    # Обновление параметра на основе баланса ошибок
    if n_hol > 0:
        net = (false_alarm - missed) / n_hol
        self.holiday_param += lr * net
    
    self.holiday_param = np.clip(self.holiday_param, 0.2, 5.0)
```

#### STL детектор ([`STLDetector`](anomaly_detection/models/stl.py))

Интегрирует праздники в процесс STL-декомпозиции:

```python
def _detect_univariate(self, time_series: TimeSeriesWrapper, dates=None):
    # STL декомпозиция: trend + seasonal + residual
    trend = series - detrend(series, 1)
    seasonal = self._extract_seasonal_component(detrended, period, window)
    residual = series - trend - seasonal
    
    # Расчет стандартного отклонения с учетом праздников
    if day_period * 7 <= len(residual):
        std_dev = self.calculate_seasonal_std(
            residual, day_period,
            apply_holidays=self._apply_holidays,
            dates=time_series.time_series_pd.index
        )
    else:
        std_dev = self.calculate_std(
            residual,
            apply_holidays=self._apply_holidays,
            dates=time_series.time_series_pd.index
        )
    
    # Расчет аномальности и доверительных интервалов
    anomaly_scores = np.abs(residual) / std_dev
    expected_bounds = np.column_stack((
        expected - self.params["threshold"] * std_dev,
        expected + self.params["threshold"] * std_dev,
    ))
```

### 3. Система управления ([`AnomalyDetectionSystem`](anomaly_detection/core/system.py))

**Конфигурация праздников:**
```python
def __init__(self, detection_model_params, ...):
    self.apply_holidays = self.detection_model_params.pop("apply_holidays", False)
    if self.apply_holidays:
        self.holiday_param = np.float64(2.0)  # Начальное значение параметра
```

**Передача параметров в детектор:**
```python
def _detect_anomalies(time_series, model_name, detection_model_params, 
                     apply_holidays=False, holiday_param=None):
    detector = AnomalyDetectionSystem.AVAILABLE_MODELS[model_name](
        **detection_model_params,
        apply_holidays=apply_holidays,
        holiday_param=holiday_param,
    )
    return detector(time_series)
```

## Структура данных

### Дневные данные ([`data/1c/daily_sales.csv`](data/1c/daily_sales.csv))
```csv
date,total_sells,holiday_mark
2013-01-01,1951.0,NEW_YEAR
2013-01-02,8198.0,NEW_YEAR
2013-01-06,5858.0,CHRISTMAS_ORTHODOX
2013-01-07,4984.0,CHRISTMAS_ORTHODOX
```

### Почасовые данные ([`data/1c/hourly_sales_with_anomalies.csv`](data/1c/hourly_sales_with_anomalies.csv))
```csv
timestamp,total_sells,holiday_mark,is_anomaly
2013-01-01 00:00:00,1951.0,NEW_YEAR,0
2013-01-01 01:00:00,2463.91,NEW_YEAR,0
```

## Визуализация и тестирование

### Инструменты визуализации

1. **[`visualize.py`](visualize.py)** - сравнение детекции с праздниками и без
2. **[`visualize2.py`](visualize2.py)** - расширенная визуализация с интерактивным слайдером
3. **[`visualize_common.py`](visualize_common.py)** - общие функции визуализации

**Интерактивный слайдер для настройки параметра праздника:**
```python
def add_holiday_slider(fig, on_change, init=2.0):
    ax_slider = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    # Создание слайдера для динамической настройки holiday_param
```

### Тестирование ([`test_detection.py`](test_detection.py))

**Адаптивное обучение на 100 эпохах:**
```python
CONFIG = DEFAULT_CONFIGURATION.copy()
CONFIG["detection_model_params"]["threshold"] = 3.49
CONFIG["detection_model_params"]["apply_holidays"] = True

system = AnomalyDetectionSystem(**CONFIG)

for epoch in range(N_EPOCHS):
    detection_result = system.detect(time_series=time_series, dates=dates)
    holiday_param = system.calculate_std_backward(
        time_series=time_series,
        dates=dates,
        ground_truth=true_labels,
        predicted=anomaly_scores
    )
    # Отслеживание метрик: precision, recall, F1
```

## Ключевые особенности реализации

### 1. Адаптивный параметр праздника
- **Начальное значение**: `holiday_param = 2.0`
- **Диапазон**: `[0.2, 5.0]`
- **Обучение**: Градиентный спуск с затухающей скоростью обучения
- **Цель**: Минимизация ложных срабатываний и пропусков на праздничных днях

### 2. Российские праздники
- Использование библиотеки `holidays` с календарем `holidays.RU()`
- Поддержка основных российских праздников: Новый год, Рождество, 23 февраля, 8 марта и др.
- Расширенные окна вокруг праздничных дней

### 3. Гибкая архитектура
- Модульная система с возможностью включения/выключения учета праздников
- Поддержка различных типов расчета стандартного отклонения (`mad`, `iqr`, `qn_scale`)
- Интеграция с существующими алгоритмами детекции

### 4. Сезонная адаптация
- Отдельный расчет стандартного отклонения для разных фаз суточного цикла
- Учет праздников в сезонном анализе через `calculate_seasonal_std()`

## Результаты и эффективность

### Метрики качества
Система отслеживает следующие метрики:
- **Precision** - точность детекции
- **Recall** - полнота детекции  
- **F1-score** - гармоническое среднее precision и recall
- **Holiday β** - текущее значение параметра праздника

### Адаптивное обучение
- Автоматическая настройка параметра праздника на основе обратной связи
- Балансировка между ложными срабатываниями и пропусками аномалий
- Затухающая скорость обучения для стабилизации параметров

## Заключение

Студенты успешно реализовали комплексную систему учета праздников в детекции аномалий временных рядов. Ключевые достижения:

1. **Полная интеграция** праздников в алгоритм STL-декомпозиции
2. **Адаптивный механизм** настройки доверительных интервалов
3. **Автоматическое обучение** параметров на основе обратной связи
4. **Гибкая архитектура** с поддержкой различных режимов работы
5. **Комплексные инструменты** визуализации и тестирования
6. **Подготовка данных** с маркировкой праздничных периодов

Реализация демонстрирует глубокое понимание проблемы детекции аномалий и творческий подход к решению задачи учета праздничных эффектов в временных рядах. Изменения включают добавление поддержки праздников во все компоненты системы детекции аномалий.