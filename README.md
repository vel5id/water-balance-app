<!-- ======================================================================= -->
# Water Balance Model (Streamlit) / Модель Водного Баланса
<!-- ======================================================================= -->

English version first. Русская версия ниже (пролистайте вниз).

---
## ENGLISH 🇬🇧

### Overview
An interactive, modular reservoir water balance and forecasting application. It ingests:
- Sentinel‑2 derived water surface area (converted via area–volume–elevation curve)
- Bathymetry + Copernicus DEM (integrated depth raster)
- GLEAM evaporation (daily)
- Precipitation time series (IMERG or equivalent)

It supports deterministic and (experimental) ensemble forecasting with seasonal + robust trend decomposition, residual block bootstrap, and seasonal spread diagnostics.

### Key Features
- Modular Streamlit UI (`wbm/ui/...`) with separable sections (trends, snow/temperature, runoff-temperature, P & ET diagnostics, phase analysis, ensemble, interactive map)
- Robust seasonal + Theil–Sen trend forecaster (`build_robust_season_trend_series`)
- Adaptive fallback logic when history is insufficient (ensemble precipitation deterministic core)
- Seasonal spread quantiles (p10 / p25 / median / p75 / p90) foundation (`wbm/seasonal.py`)
- Experimental ensemble: block bootstrap of residuals → volume forecast quantile fan
- Area–Volume curve integration; simulated volume → elevation / surface extent logic
- Downloadable scenario CSV
- Backtest procedure for 2024 forecasting using only 2022–2023 data (`URGENT_BACKTEST_2024.md`)
- Legacy monolith preserved (`legacy_app_full.py`) for audit/comparison
- Date normalization & Arrow export safety (mitigates PyArrow serialization errors)

### Repository Structure (Simplified)
```
app.py                    # Orchestrator (imports modular UI & simulation)
wbm/
	curve.py, simulate.py, forecast.py, seasonal.py, ensemble.py, ...
	ui/
		controls.py           # Sidebar controls factory
		data.py               # Data loading & sanitation
		simulation.py         # Scenario driver build & deterministic run
		sections/             # Discrete UI panels
			trends.py
			snow_temp.py
			runoff_temp.py
			p_et_diag.py
			phase.py
			ensemble.py
			map_view.py
		state.py              # (Extensible) shared UI state dataclasses
processing_output/        # Generated artifacts (area-volume curve, rasters)
water_balance_output/     # Baseline & scenario outputs
URGENT_BACKTEST_2024.md   # Backtest instructions
```

### Data Requirements (`DATA_ROOT`)
Environment variable `DATA_ROOT` points to a folder containing at minimum:
```
processing_output/
	area_volume_curve.csv
	integrated_bathymetry_copernicus.tif   # preferred depth DEM
	ndwi_mask_0275.tif                     # NDWI water mask (1=water)
GLEAM/processed/
	gleam_summary_all_years.csv            # includes evaporation_mm / E
precipitation_timeseries.csv             # precipitation_mm (or mean_precip_mm_per_h*24)
water_balance_output/
	water_balance_final.csv                # optional baseline
```
Fallback: if `integrated_bathymetry_copernicus.tif` missing, tries `bathymetry_reprojected_epsg4326.tif`.

### Installation
PowerShell (Windows):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Set Data Path
PowerShell:
```powershell
$env:DATA_ROOT = 'C:\\wbm-data'
```
Linux/macOS:
```bash
export DATA_ROOT=/srv/wbm-data
```

### Optional Artifact Build
```powershell
python dem_processor.py          # builds integrated DEM & area_volume_curve.csv
python water_balance_model.py    # builds baseline (optional)
```

### Run App
```powershell
streamlit run app.py --server.port 8501 --server.headless true
```
Navigate to the shown local URL.

### Forecasting Logic
1. Load historical daily P & ET
2. Build seasonal (DOY or monthly) median mapping
3. Theil–Sen slope on de-seasonalised residual → robust trend
4. Extend deterministic future: season(d) + trend(d)
5. (Ensemble) Bootstrap residual blocks; add to deterministic path; propagate through volume model

### Ensemble (Experimental)
Parameters inside the Ensemble expander:
- Member count (N)
- History window & dynamic min_history adaptation
- Season basis: DOY vs MONTH
- Residual block bootstrap (Auto block length via ACF or manual)
- Quantile fan (configurable low/median/high)
Graceful diagnostics appear if deterministic precipitation cannot be built (insufficient history).

### Seasonal Spread
`seasonal_spread` (in `seasonal.py`) provides per-season quantiles enabling future UI band visualization (pending integration task).

### Backtesting
Documented in `URGENT_BACKTEST_2024.md`. Core idea: train on 2022–2023, forecast 2024, compute MAE/RMSE/Bias/MAPE (and NSE for volume), compare vs climatology baseline.

### Roadmap / Open Tasks
- [ ] Smoothing window UI & application to seasonal template
- [ ] Integrate seasonal spread fan in main forecast plot
- [ ] Caching layer (`@st.cache_data` / `@st.cache_resource`) for ACF, bootstrap, seasonal template
- [ ] Optional backtest automation script `backtest_2024.py`
- [ ] Robust anomaly filtering (outlier precipitation masking)

### Docker (Optional)
```Dockerfile
FROM python:3.10-slim
RUN apt-get update && apt-get install -y gdal-bin libgdal-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV DATA_ROOT=/data
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]
```
Run:
```bash
docker build -t wbm-app .
docker run --rm -p 8501:8501 -e DATA_ROOT=/data -v /host/wbm-data:/data wbm-app
```

### Troubleshooting
| Issue | Suggestion |
|-------|------------|
| Missing area_volume_curve.csv | Run `dem_processor.py` |
| Zero precipitation | Ensure column `precipitation_mm` or convert hourly → daily |
| GDAL errors | Install GDAL libs (Linux) or use provided Dockerfile |
| Inverted water extent | Switch map mode or confirm depth raster sign convention |
| ArrowInvalid / date serialization | Dates are normalized; ensure new data is date-normalized too |

### License
Proprietary / internal use (adjust as needed).

---
## РУССКАЯ ВЕРСИЯ 🇷🇺

### Обзор
Интерактивная модульная модель водного баланса водохранилища (Streamlit). Использует:
- Площадь зеркала воды (Sentinel‑2 / NDWI)
- Интегрированный батиметрический/рельефный DEM (глубины)
- Испарение GLEAM (суточное)
- Осадки (IMERG или эквивалент)

Поддерживает детерминированный и экспериментальный ансамблевый прогноз (сезонность + робастный тренд, бутстрап остатков, диагностический разброс сезонных квантилей).

### Основные возможности
- Модульная архитектура UI (`wbm/ui/...`)
- Робастная сезонно-трендовая модель (Theil–Sen по остаткам)
- Адаптивное снижение требований к истории при её нехватке
- Квантили сезонного разброса (p10, p25, median, p75, p90)
- Ансамбль: блочный бутстрап остатков осадков → веер прогноза объёма
- Кривая площадь–объём–отметка; расчёт уровня
- Выгрузка сценария в CSV
- Документ бэк-теста 2024 (`URGENT_BACKTEST_2024.md`)
- Сохранён легаси монолит для сравнения

### Структура
```
app.py
wbm/ (логика прогноза, сезонность, статистика, ансамбль)
wbm/ui/ (контролы, загрузка данных, симуляция, секции интерфейса)
processing_output/ (кривая, DEM, маски)
water_balance_output/ (базовый и рассчитанные ряды)
```

### Требуемые данные (`DATA_ROOT`)
См. блок в английской части. Минимум: кривая, DEM глубин, NDWI, осадки, испарение.

### Установка
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Настройка пути:
```powershell
$env:DATA_ROOT = 'C:\\wbm-data'
```

### Запуск
```powershell
streamlit run app.py --server.port 8501 --server.headless true
```

### Логика прогноза
1. Сезонная медиана (по дню года или месяцу)
2. Theil–Sen тренд по десезонированным остаткам
3. Сложение сезонности + тренда → детерминированный будущий ряд
4. Ансамбль: бутстрап блоков остатков + сложение с детерминированным рядом

### Ансамбль
Параметры: число участников, окно истории, база сезонности (DOY/MONTH), блок бутстрапа (Auto/Manual), квантильный диапазон. Диагностика выводит причину, если не удалось построить детерминированную серию осадков.

### Сезонный разброс
Функция квантильного профиля (p10–p90) подготовлена для визуализации (планируется интеграция в основной график).

### Бэк-тест 2024
Процедура в `URGENT_BACKTEST_2024.md`: обучаемся на 2022–2023, прогнозируем 2024, считаем MAE/RMSE/Bias/MAPE/NSE, сравниваем с климатологией.

### Дорожная карта
- [ ] Слайдер сглаживания сезонного шаблона
- [ ] Визуализация веера сезонного квантильного разброса
- [ ] Кэширование тяжёлых расчётов
- [ ] Авто-скрипт бэк-теста `backtest_2024.py`
- [ ] Фильтрация выбросов (осадки)

### Docker (пример)
См. Dockerfile в английской части.

### Частые проблемы
| Проблема | Решение |
|----------|---------|
| Нет area_volume_curve.csv | Запустить `dem_processor.py` |
| Нули в осадках | Проверить колонку или преобразовать часы→сутки |
| Ошибки GDAL | Установить gdal-bin / libgdal-dev |
| Инвертированная вода на карте | Проверить режим отображения / знак глубины |
| Ошибки Arrow при экспорте | Убедиться в нормализации дат |

### Лицензия
Proprietary / internal use (уточните при необходимости).

---
Generated: 2025-10-08. Bilingual README.
