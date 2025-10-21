# 🔮 Prophet Backtest - Документация

## 📋 Описание

Скрипт для бэктестинга модели **Prophet** от Facebook для прогнозирования объема водохранилища.

Prophet - это мощная библиотека для временных рядов, которая:
- ✅ Автоматически обрабатывает сезонность (годовую, месячную, квартальную)
- ✅ Robust к выбросам и пропускам в данных
- ✅ Позволяет легко добавлять внешние регрессоры
- ✅ Не требует стационарности данных (в отличие от SARIMA)

---

## 🎯 Почему Prophet для 30+ дневных прогнозов?

### Проблемы SARIMA на длинных горизонтах:
```
30 дней:  R² = 0.406, NSE = 0.221 ⚠️
365 дней: R² = 0.006, NSE = -26.2 ❌
```

### Преимущества Prophet:
1. **Лучше обрабатывает сезонность** - monthly, quarterly, yearly
2. **Может учитывать тренды** - linear или logistic growth
3. **Легко добавить регрессоры** - температура, осадки, приток
4. **Не требует стационарности** - работает с сырыми данными

---

## 🚀 Запуск

### Базовый запуск (без внешних регрессоров):
```bash
cd c:\Users\vladi\Downloads\Data\post-processing_data
python backtest_prophet.py
```

### С внешними регрессорами:
В коде измените:
```python
metrics = backtest_prophet_horizons(
    df,
    horizons,
    use_external_regressors=True,  # Включить регрессоры
    external_data=external_df       # Ваши данные
)
```

---

## 📊 Настройки модели

### Текущие параметры Prophet:

```python
Prophet(
    growth='linear',              # Тип тренда (linear/logistic)
    yearly_seasonality=True,      # Годовая сезонность
    weekly_seasonality=False,     # Недельная (отключена для ежедневных данных)
    daily_seasonality=False,      # Дневная (не нужна)
    seasonality_mode='additive',  # Тип сезонности (additive/multiplicative)
    seasonality_prior_scale=10.0, # Гибкость сезонности (выше = более гибко)
    changepoint_prior_scale=0.05  # Гибкость тренда (выше = больше изломов)
)
```

### Дополнительные сезонности:

```python
# Месячная (период ~30.5 дней)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Квартальная (период ~91 день)
model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
```

---

## 🎛️ Как добавить внешние регрессоры

### 1. Подготовить данные:

```python
# Формат: DataFrame с колонкой 'ds' (дата) и регрессорами
external_data = pd.DataFrame({
    'ds': dates,
    'temperature': temp_values,
    'precipitation': precip_values,
    'inflow': inflow_values,
    'evaporation': evap_values
})
```

### 2. Передать в функцию:

```python
metrics = backtest_prophet_horizons(
    df,
    horizons=[1, 30, 365],
    use_external_regressors=True,
    external_data=external_data
)
```

### 3. Prophet автоматически использует их:

```python
# В функции build_prophet_model:
for col in external_data.columns:
    if col != 'ds':
        model.add_regressor(col)
```

---

## 📈 Ожидаемые улучшения

### Горизонт 30 дней:
```
Текущий SARIMA:  R² = 0.406, RMSE = 42.1 млн.м³
Цель Prophet:    R² > 0.65,  RMSE < 30 млн.м³
Улучшение:       ~60% по R², ~30% по RMSE
```

### Горизонт 365 дней:
```
Текущий SARIMA:  R² = 0.006, NSE = -26.2 ❌
Цель Prophet:    R² > 0.40,  NSE > 0.30
Улучшение:       Из "хуже среднего" в "приемлемо"
```

---

## 🔍 Интерпретация результатов

### Метрики качества:

- **R²** (Coefficient of Determination)
  - 1.0 = идеальный прогноз
  - 0.0 = прогноз на уровне среднего
  - < 0 = хуже среднего

- **NSE** (Nash-Sutcliffe Efficiency)
  - 1.0 = идеальный прогноз
  - 0.0 = прогноз равен среднему
  - < 0 = хуже среднего

- **KGE** (Kling-Gupta Efficiency)
  - 1.0 = идеальный прогноз
  - > 0.7 = хорошо
  - > 0.5 = приемлемо

- **RMSE** (Root Mean Square Error)
  - В млн.м³
  - Чем ниже, тем лучше
  - < 10% от среднего объема = отлично

---

## 📁 Выходные файлы

### Создаются автоматически:

1. **forecast_backtest_prophet_metrics.json**
   - Финальные метрики по каждому горизонту
   - R², NSE, KGE, RMSE, MAE, MAPE

2. Автоматическое сравнение с SARIMA (если есть)

---

## ⚙️ Оптимизация производительности

### Текущие настройки:
- **n_jobs = -1** - все CPU ядра
- **chunk_size = 100** - обработка по 100 origins
- **max_origins = 200** - ограничение для скорости

### Время выполнения:
- **Без регрессоров**: ~30-60 минут
- **С регрессорами**: ~1-2 часа

### Ускорение:
```python
# Уменьшить количество origins
step = max(1, (end_i - start_i) // 100)  # вместо 200

# Уменьшить chunk_size (меньше памяти)
chunk_size = 50

# Меньше ядер (если нужна память для других задач)
n_jobs = 4
```

---

## 🐛 Troubleshooting

### Prophet устанавливается долго?
```bash
# Prophet требует компиляции C++ кода
# Установка может занять 5-10 минут
pip install prophet
```

### Ошибки при установке Prophet?
```bash
# Установите зависимости:
pip install pystan==2.19.1.1
pip install prophet
```

### Out of Memory?
```python
# Уменьшите chunk_size и количество origins
chunk_size = 50
step = max(1, (end_i - start_i) // 100)
```

---

## 🎯 Следующие шаги

### После бэктестинга Prophet:

1. **Сравните с SARIMA** - автоматически выводится
2. **Если Prophet лучше** - используйте для production
3. **Добавьте регрессоры** - для дальнейшего улучшения
4. **Создайте ensemble** - Prophet + SARIMA + XGBoost

### Рекомендуемый workflow:

```
1. Запустить backtest_prophet.py (без регрессоров)
2. Посмотреть метрики и сравнение
3. Если улучшение есть → добавить регрессоры
4. Если не помогает → попробовать ensemble
```

---

## 📊 Пример вывода

```
======================================================================
🔮 PROPHET BACKTEST - Water Volume Forecasting
======================================================================
📊 Total origins: 200
🔧 Parallelism: all CPUs
📈 Horizons: [1, 30, 365] days
🌊 External regressors: No

Processing 2 chunks...
Chunks: 100%|████████████████████████████| 2/2 [15:32<00:00, 466.24s/chunk]

======================================================================
✅ PROPHET BACKTEST COMPLETE
======================================================================
{
  "1": {
    "r2": 0.998,
    "nse": 0.997,
    "rmse": 2.5
  },
  "30": {
    "r2": 0.689,
    "nse": 0.654,
    "rmse": 28.3
  },
  "365": {
    "r2": 0.423,
    "nse": 0.381,
    "rmse": 165.2
  }
}

======================================================================
📊 COMPARISON: Prophet vs SARIMA
======================================================================

🔹 Horizon: 30 days
----------------------------------------------------------------------
  R²:
    Prophet: 0.6890
    SARIMA:  0.4056
    Winner:  🏆 Prophet

  RMSE:
    Prophet: 28.30 млн.м³
    SARIMA:  42.10 млн.м³
    Winner:  🏆 Prophet
```

---

**Автор:** GitHub Copilot  
**Дата:** 2025-10-16  
**Версия:** 1.0
