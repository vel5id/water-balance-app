# 🚀 QUICK START: Использование Prophet и SARIMA в симуляции

## За 5 минут до первого прогноза

### 1️⃣ Установка зависимостей

```bash
# Установить ВСЕ требуемые пакеты
pip install -r requirements.txt

# Установить дополнительно (если хотите SARIMA + Prophet)
pip install pmdarima pystan==2.19.1.1 prophet
```

**Время:** 2-5 минут (Prophet может долго устанавливаться)

---

### 2️⃣ Запуск приложения

```bash
streamlit run app.py
```

Приложение откроется в браузере на `http://localhost:8501`

---

### 3️⃣ Выбор метода прогноза в UI

Слева в панели управления:

1. Найти раздел **"Forecast Options"** (под "Seasonal + trend options")
2. Увидите новую опцию: **"📊 Forecast Method"**
3. Выбрать один из трёх:
   - ✅ **Theil-Sen (fast, robust)** ← **РЕКОМЕНДУЕТСЯ** по умолчанию
   - 🔬 **SARIMA (experimental)**
   - 🎯 **Prophet (advanced)**

4. Установить горизонт прогноза: 180 дней
5. Нажать **"Run Scenario"**

---

### 4️⃣ Результаты

После запуска увидите:

**Для Theil-Sen:**
```
✅ Готово < 0.1 сек
```

**Для SARIMA:**
```
✅ SARIMA (ARIMA): AIC=1234.56
```

**Для Prophet:**
```
✅ Prophet: 1825 observations used
```

---

## 🎯 Рекомендации

| Ваша ситуация | Выбрать |
|---|---|
| Первый раз, не знаю что выбрать | **Theil-Sen** ✅ |
| История данных 1-2 года | **Theil-Sen** ✅ |
| История данных 2+ года | **SARIMA** 🔬 |
| Нужны доверительные интервалы | **Prophet** 🎯 |
| Нерегулярные скачки в данных | **Prophet** 🎯 |
| Максимальная скорость | **Theil-Sen** ✅ |

---

## 📝 Примеры кода

### Использовать в Python скрипте

```python
from wbm.forecast import (
    build_robust_season_trend_series,
    build_sarima_forecast_enhanced,
    build_prophet_forecast
)
import pandas as pd

# Загрузить данные
data = pd.read_csv("precipitation.csv")
series = pd.Series(data['precip_mm'].values, 
                   index=pd.to_datetime(data['date']))

# Метод 1: Theil-Sen (быстро)
forecast_ts = build_robust_season_trend_series(
    series, 
    future_days=180
).deterministic

# Метод 2: SARIMA (авторегрессия)
forecast_sarima, info = build_sarima_forecast_enhanced(
    series,
    future_days=180,
    max_history=730
)
print(f"AIC: {info['aic']}")

# Метод 3: Prophet (продвинутый)
forecast_prophet, info = build_prophet_forecast(
    series,
    future_days=180,
    yearly_seasonality=True
)
print(f"Observations: {info['n_obs']}")

# Сравнить
import matplotlib.pyplot as plt
plt.plot(forecast_ts.index, forecast_ts, label='Theil-Sen')
plt.plot(forecast_sarima.index, forecast_sarima, label='SARIMA')
plt.plot(forecast_prophet.index, forecast_prophet, label='Prophet')
plt.legend()
plt.show()
```

---

## ⚠️ Если что-то не работает

### Prophet требует pystan
```
Error: ModuleNotFoundError: No module named 'pystan'

Решение:
pip install pystan==2.19.1.1
```

### SARIMA требует pmdarima
```
Error: ModuleNotFoundError: No module named 'pmdarima'

Решение:
pip install pmdarima
```

### SARIMA медленно или "Out of Memory"
```
Норма! SARIMA требует больше ресурсов.
Автоматически использует max_history=730 для экономии памяти.
```

### Prophet вообще не работает
```
Prophet может быть тяжелым на Windows.
Рекомендуется использовать Theil-Sen или попробовать Docker.
```

---

## 📊 Что дальше?

После выбора метода:

1. **График видит прогноз** на следующие 180 дней
2. **Компоненты** показывают сезонность и тренд
3. **Статистика** в нижней части
4. **Скачать** результаты как CSV

---

## 🔧 Расширенные параметры

### Если хотите изменить параметры (в коде):

```python
# forecast.py → build_sarima_forecast_enhanced()
forecast, info = build_sarima_forecast_enhanced(
    series,
    future_days=365,      # Дней прогноза
    min_history=90,       # Мин. история
    max_history=1095,     # Макс. история (для ОМ)
    max_p=7,              # Макс. AR порядок
    max_q=7,              # Макс. MA порядок
    seasonal=True,        # Сезонность вкл.
)

# forecast.py → build_prophet_forecast()
forecast, info = build_prophet_forecast(
    series,
    future_days=365,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95   # 95% интервалы
)
```

---

## 📚 Полная документация

- 📖 **TECHNICAL_DOCUMENTATION.md** – полный обзор архитектуры
- 📖 **FORECAST_METHODS_GUIDE.md** – подробные примеры
- 📖 **INTEGRATION_COMPLETE.md** – что изменилось
- 📖 **PROPHET_SARIMA_STATUS.md** – статус внедрения

---

## ✅ Проверка что все работает

```bash
# Быстрый тест
python -c "
from wbm.forecast import *
import pandas as pd
import numpy as np

# Синтетические данные
dates = pd.date_range('2020-01-01', periods=730, freq='D')
data = np.sin(np.arange(730) * 2 * np.pi / 365) + np.random.randn(730) * 0.1
series = pd.Series(data, index=dates)

# Протестировать каждый метод
print('Testing Theil-Sen...')
ts = build_robust_season_trend_series(series, future_days=90)
print(f'✅ Theil-Sen OK: {len(ts.deterministic)} дней прогноза')

print('Testing SARIMA...')
try:
    sarima, _ = build_sarima_forecast_enhanced(series, future_days=90)
    print(f'✅ SARIMA OK: {len(sarima)} дней прогноза')
except Exception as e:
    print(f'⚠️ SARIMA пропущена (требует pmdarima): {e}')

print('Testing Prophet...')
try:
    prophet, _ = build_prophet_forecast(series, future_days=90)
    print(f'✅ Prophet OK: {len(prophet)} дней прогноза')
except Exception as e:
    print(f'⚠️ Prophet пропущена (требует prophet): {e}')
"
```

---

**Готово! 🎉 Теперь у вас есть три метода прогнозирования в симуляции.**

Начните с **Theil-Sen**, экспериментируйте с **SARIMA** и **Prophet** для улучшения точности.
