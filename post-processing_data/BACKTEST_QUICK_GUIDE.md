# ⚡ Быстрое руководство по Backtest v2.0

## 🚀 Запуск

### Простой запуск (рекомендуется для 80GB RAM):
```bash
cd c:\Users\vladi\Downloads\Data\post-processing_data
python backtest_forecast_horizons.py
```

## 🎯 Параметры по умолчанию (v2.0)

```python
n_jobs = -1              # Все CPU ядра
chunk_size = 500         # Оптимально для 80GB RAM
max_origins = 800        # Удвоено с 400
use_cache = True         # Кэширование включено
cache_size_gb = 5.0      # Макс размер кэша
```

## 📊 Что ожидать

### Первый запуск:
```
🔍 Валидация данных... ✅
💾 Кэш: ВКЛЮЧЕН
📊 800 origins в 2 chunks
⏱️  Время: ~6-8 часов
💾 RAM: ~55-65GB пик
```

### Повторный запуск:
```
🔍 Валидация данных... ✅
💾 Кэш: hit rate 97%
⏱️  Время: ~20-40 минут ⚡
```

## 📁 Результаты

После выполнения смотрите:
```
processed_data/water_balance_output/
├── forecast_backtest_metrics.json      # ← Финальные метрики
├── forecast_backtest_progress.jsonl    # Промежуточный прогресс
├── forecast_backtest_samples.csv       # Примеры прогнозов
├── best_sarima_params.json             # Лучшие параметры
└── model_cache/                        # Кэш моделей (~0.5-5GB)
    └── sarima_*.pkl
```

## 🔧 Настройка под вашу систему

### Если RAM < 80GB (например, 32GB):
```python
metrics = backtest_volume_horizons(
    df, 
    horizons=[1, 30, 365],
    chunk_size=200,      # Уменьшено
    n_jobs=4             # Меньше процессов
)
```

### Если нужна скорость > точность:
```python
# В коде измените:
step = max(1, (end_i - start_i) // 400)  # было 800
```

### Если нужна точность > скорость:
```python
# В коде измените:
step = max(1, (end_i - start_i) // 1200)  # было 800
```

## 🐛 Решение проблем

### Out of Memory:
```bash
# Уменьшите chunk_size в main():
metrics = backtest_volume_horizons(df, horizons, chunk_size=300)
```

### Очистка кэша:
```bash
# PowerShell
Remove-Item -Recurse -Force processed_data\water_balance_output\model_cache
```

### Валидация не прошла:
```
Проверьте:
1. Есть ли колонки 'date' и 'volume_mcm'
2. Есть ли минимум 90 точек данных
3. Нет ли отрицательных значений объема
```

## 📈 Интерпретация результатов

### forecast_backtest_metrics.json:
```json
{
  "1": {    // Горизонт 1 день
    "n": 800,           // Количество прогнозов
    "rmse": 2.3,        // RMSE в млн.м³
    "r2": 0.92,         // R² (чем ближе к 1, тем лучше)
    "nse": 0.91,        // NSE (чем ближе к 1, тем лучше)
    "mape_pct": 1.8     // MAPE в % (чем меньше, тем лучше)
  },
  "30": { ... },        // Горизонт 30 дней
  "365": { ... }        // Горизонт 365 дней
}
```

### Хорошие метрики:
- ✅ R² > 0.70
- ✅ NSE > 0.70
- ✅ RMSE < 10% от среднего объема
- ✅ MAPE < 5%

### Плохие метрики:
- ❌ R² < 0.50
- ❌ NSE < 0.50
- ❌ RMSE > 20% от среднего
- ❌ MAPE > 15%

## 💡 Советы

### 1. Мониторинг прогресса в реальном времени:
```bash
# PowerShell - tail для Windows
Get-Content processed_data\water_balance_output\forecast_backtest_progress.jsonl -Wait -Tail 10
```

### 2. Проверка использования памяти:
```bash
# Task Manager или PowerShell:
Get-Process python | Select-Object ProcessName, @{Name="Memory(GB)";Expression={[math]::Round($_.WS/1GB,2)}}
```

### 3. Оценка оставшегося времени:
```
Смотрите в progress.jsonl:
"pct": 25.5  ← 25% выполнено
```

## 🎓 Дополнительная информация

Полная документация: [BACKTEST_IMPROVEMENTS.md](./BACKTEST_IMPROVEMENTS.md)

---

**Версия:** 2.0  
**Дата:** 2025-10-15  
**Оптимизировано для:** 80GB RAM
