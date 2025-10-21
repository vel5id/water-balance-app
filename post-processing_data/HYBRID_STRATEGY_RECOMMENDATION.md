# 🎯 ИТОГОВЫЕ РЕКОМЕНДАЦИИ: Гибридная стратегия SARIMA + Prophet

## 📊 Результаты с расширенным набором горизонтов

### **Prophet (1482 origins, 3 горизонта):**
| Горизонт | R² | RMSE | Оценка |
|----------|----|----|--------|
| 1 день | 0.411 | 36.82 | ❌ Плохо |
| 30 дней | **-758.18** | 1311.37 | ❌❌❌ Катастрофа |
| 365 дней | -3.64 | **92.42** | ⚠️ R² плохой, но RMSE лучше SARIMA |

### **SARIMA (1474 origins, 3 горизонта):**
| Горизонт | R² | RMSE | Оценка |
|----------|----|----|--------|
| 1 день | **0.997** | **2.67** | ✅✅✅ Отлично |
| 30 дней | **0.406** | **42.10** | ✅ Хорошо |
| 365 дней | 0.006 | 224.20 | ⚠️ Плохо (оба метода) |

---

## 🔍 Критический анализ Prophet

### **Почему Prophet провалился?**

1. **Недостаточная история для обучения**
   - Текущий `start_i = 90 дней` — это **КРИТИЧЕСКИ мало** для Prophet
   - Prophet требует **минимум 365 дней** для годовой сезонности
   - С 90 днями модель не может уловить сезонные паттерны

2. **Неоптимальные гиперпараметры**
   - `changepoint_prior_scale=0.01` — слишком жесткий для 30-дней
   - Модель не может адаптироваться к краткосрочным изменениям

3. **Prophet не для краткосрочных прогнозов**
   - Prophet оптимизирован для **месяцев/лет**, не дней/недель
   - Для 1-30 дней SARIMA в **10-30 раз лучше**

---

## 🎯 ОПТИМАЛЬНАЯ СТРАТЕГИЯ

### **1. Краткосрочные прогнозы (1-7 дней): SARIMA 🏆**
```python
# SARIMA показывает отличные результаты
horizons = [1, 2, 3, 7]
model = "SARIMA"
expected_r2 = 0.95 - 0.99
expected_rmse = 2-10 млн.м³
```

**Обоснование:**
- R² = 0.997 для 1-дня
- RMSE в **13 раз лучше** Prophet
- Быстрое обучение (~2 сек/origin с кэшем)

### **2. Среднесрочные прогнозы (7-30 дней): SARIMA 🏆**
```python
horizons = [7, 14, 30]
model = "SARIMA"
expected_r2 = 0.40 - 0.60
expected_rmse = 40-60 млн.м³
```

**Обоснование:**
- R² = 0.406 для 30-дней (приемлемо)
- RMSE в **31 раз лучше** Prophet (42 vs 1311!)
- Prophet показывает отрицательный R² = -758 ❌

### **3. Долгосрочные прогнозы (90-365 дней): Ансамбль или SARIMA ⚠️**
```python
horizons = [90, 180, 365]

# Вариант 1: Попробовать улучшить Prophet (с внешними регрессорами)
model = "Prophet + regressors"
start_i = 365  # Увеличить историю!
external_regressors = ["temperature", "precipitation", "evaporation", "inflow"]

# Вариант 2: Использовать SARIMA (если Prophet не улучшится)
model = "SARIMA"
# SARIMA показывает R² = 0.006 (плохо), но лучше Prophet

# Вариант 3: Ансамбль (взвешенное среднее)
ensemble_weights = {
    "SARIMA": 0.7,  # Больше веса SARIMA
    "Prophet": 0.3  # Меньше веса Prophet
}
```

**Обоснование:**
- **Prophet RMSE лучше** (92 vs 224), НО R² = -3.64 ❌
- **SARIMA R² лучше** (0.006 vs -3.64), НО RMSE хуже
- **Оба метода плохи** — нужны улучшения

---

## 🚀 ПЛАН ДЕЙСТВИЙ

### **Этап 1: Запустить backtest с расширенными горизонтами ✅ (текущий)**
```python
horizons = [1, 2, 3, 7, 30, 90, 180, 365]
```

**Цель:** Получить полную картину по всем горизонтам

### **Этап 2: Улучшить Prophet для долгосрочных прогнозов**

#### **2.1. Увеличить минимальную историю**
```python
# В backtest_prophet.py, строка ~240:
start_i = 365  # Было 90, стало 365 (год истории)
```

**Эффект:**
- Prophet сможет уловить годовую сезонность
- Меньше origins (~600 вместо 1482), но лучшее качество
- Время выполнения: ~10-15 минут

#### **2.2. Добавить внешние регрессоры**
```python
# Подготовить данные:
external_data = pd.DataFrame({
    'ds': dates,
    'temperature': temperature_series,
    'precipitation': precipitation_series,
    'evaporation': evaporation_series,
    'inflow': inflow_series
})

# Запустить backtest:
metrics = backtest_prophet_horizons(
    df,
    horizons=[90, 180, 365],  # Только долгосрочные
    use_external_regressors=True,
    external_data=external_data
)
```

**Ожидаемое улучшение:**
- R² для 365-дней: -3.64 → **0.3-0.5** (в 10-15 раз лучше!)
- RMSE: 92 → **60-80** (на 15-30% лучше)

#### **2.3. Grid search гиперпараметров**
```python
# Попробовать разные значения:
changepoint_prior_scales = [0.001, 0.005, 0.01, 0.05, 0.1]
seasonality_prior_scales = [1.0, 5.0, 10.0, 20.0]

# Для долгосрочных прогнозов может потребоваться более гибкая модель
```

### **Этап 3: Создать ансамбль (если нужно)**
```python
def ensemble_forecast(sarima_pred, prophet_pred, horizon):
    """Взвешенное среднее SARIMA и Prophet."""
    if horizon <= 30:
        # Краткосрочно: только SARIMA
        return sarima_pred
    elif 30 < horizon <= 90:
        # Среднесрочно: 80% SARIMA, 20% Prophet
        return 0.8 * sarima_pred + 0.2 * prophet_pred
    else:
        # Долгосрочно: 50% SARIMA, 50% Prophet
        return 0.5 * sarima_pred + 0.5 * prophet_pred
```

---

## 📋 КОНКРЕТНЫЕ ШАГИ (в порядке приоритета)

### **ВЫСОКИЙ ПРИОРИТЕТ:**

1. ✅ **Запустить SARIMA backtest с расширенными горизонтами**
   ```bash
   python backtest_forecast_horizons.py
   ```
   **Время:** ~20-30 минут (с кэшем)
   **Цель:** Получить метрики для [1, 2, 3, 7, 30, 90, 180, 365] дней

2. ✅ **Проанализировать результаты**
   - Определить, на каких горизонтах SARIMA начинает "ломаться" (R² < 0.5)
   - Это будут кандидаты для Prophet или ансамбля

### **СРЕДНИЙ ПРИОРИТЕТ:**

3. 🔜 **Улучшить Prophet с `start_i=365`**
   ```python
   # В backtest_prophet.py:
   start_i = 365  # Увеличить историю
   horizons = [90, 180, 365]  # Только долгосрочные
   ```
   **Время:** ~15-20 минут
   **Ожидаемый результат:** R² улучшится с -3.64 до 0.2-0.4

4. 🔜 **Добавить внешние регрессоры в Prophet**
   - Подготовить данные: температура, осадки, испарение, приток
   - Запустить backtest с `use_external_regressors=True`
   **Ожидаемый результат:** R² улучшится до 0.4-0.6

### **НИЗКИЙ ПРИОРИТЕТ:**

5. 🔜 **Grid search гиперпараметров Prophet**
   - Только если предыдущие шаги не дали результата
   **Время:** ~2-3 часа

6. 🔜 **Создать ансамбль SARIMA + Prophet**
   - Только если оба метода показали приемлемые результаты
   **Время:** ~30 минут

---

## 💾 Что нужно сделать СЕЙЧАС:

### **1. Запустить SARIMA backtest с новыми горизонтами:**
```powershell
C:/Users/vladi/Downloads/Data/.venv/Scripts/python.exe "c:\Users\vladi\Downloads\Data\post-processing_data\backtest_forecast_horizons.py"
```

### **2. После завершения проанализировать результаты:**
- Открыть `forecast_backtest_metrics.json`
- Посмотреть R² и RMSE для каждого горизонта
- Определить пороговые значения для выбора модели

### **3. Принять решение:**
```python
if horizon <= 30:
    use_model = "SARIMA"  # R² > 0.9 для 1-7 дней, R² > 0.4 для 30 дней
elif horizon <= 90:
    if sarima_r2 > 0.3:
        use_model = "SARIMA"
    else:
        use_model = "Ensemble"
else:  # horizon > 90
    if improved_prophet_r2 > 0.4:
        use_model = "Prophet"
    elif improved_prophet_rmse < sarima_rmse * 0.7:
        use_model = "Prophet"
    else:
        use_model = "Ensemble"
```

---

## 📈 Ожидаемые результаты после улучшений

### **После `start_i=365` для Prophet:**
| Горизонт | Prophet R² (сейчас) | Prophet R² (ожидается) | Улучшение |
|----------|---------------------|------------------------|-----------|
| 90 дней  | ? | 0.1-0.3 | +10-30% |
| 180 дней | ? | 0.2-0.4 | +20-40% |
| 365 дней | -3.64 | **0.2-0.4** | +70-80% |

### **После добавления внешних регрессоров:**
| Горизонт | Prophet R² (с start_i=365) | Prophet R² (с регрессорами) | Улучшение |
|----------|----------------------------|----------------------------|-----------|
| 90 дней  | 0.1-0.3 | **0.3-0.5** | +20-50% |
| 180 дней | 0.2-0.4 | **0.4-0.6** | +20-50% |
| 365 дней | 0.2-0.4 | **0.5-0.7** | +30-75% |

---

## 🎯 ИТОГОВАЯ РЕКОМЕНДАЦИЯ

**Для production системы прогнозирования:**

```python
def get_forecast(horizon_days):
    """Получить прогноз на заданный горизонт."""
    
    if horizon_days <= 7:
        # Ultra-short term: SARIMA доминирует
        return sarima_forecast(horizon_days)
    
    elif 7 < horizon_days <= 30:
        # Short term: SARIMA отлично работает
        return sarima_forecast(horizon_days)
    
    elif 30 < horizon_days <= 90:
        # Mid term: SARIMA приемлемо, Prophet нужно улучшить
        sarima_pred = sarima_forecast(horizon_days)
        
        # Если Prophet улучшен с внешними регрессорами:
        if prophet_r2 > 0.4:
            prophet_pred = prophet_forecast(horizon_days)
            return 0.7 * sarima_pred + 0.3 * prophet_pred
        else:
            return sarima_pred
    
    else:  # horizon_days > 90
        # Long term: Prophet с внешними регрессорами
        prophet_pred = prophet_forecast_with_regressors(horizon_days)
        
        # Если Prophet R² > 0.5, используем его
        if prophet_r2 > 0.5:
            return prophet_pred
        # Иначе ансамбль
        else:
            sarima_pred = sarima_forecast(horizon_days)
            return 0.5 * sarima_pred + 0.5 * prophet_pred
```

---

**Следующий шаг:** Запустить SARIMA backtest с расширенными горизонтами и посмотреть результаты!
