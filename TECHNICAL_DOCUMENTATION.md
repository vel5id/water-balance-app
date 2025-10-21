# Техническая Документация: Интерактивная Модель Водного Баланса
**Версия: 1.0**  
**Дата: Октябрь 2025**  
**Язык: Русский**

---

## Оглавление
1. [Обзор проекта](#обзор-проекта)
2. [Архитектура системы](#архитектура-системы)
3. [Структура проекта](#структура-проекта)
4. [Входные данные и источники](#входные-данные-и-источники)
5. [Процессы обработки данных](#процессы-обработки-данных)
6. [Ядро модели водного баланса](#ядро-модели-водного-баланса)
7. [Выходные данные](#выходные-данные)
8. [Web-приложение (Streamlit)](#web-приложение-streamlit)
9. [Установка и развёртывание](#установка-и-развёртывание)
10. [Примеры использования](#примеры-использования)

---

## Обзор проекта

**Название:** Интерактивная модель водного баланса водохранилища  
**Назначение:** Расчёт и прогнозирование объёма воды в водохранилище на основе:
- Площади водной поверхности (Sentinel-2 NDWI)
- Испарения (GLEAM)
- Осадков (ERA5 / IMERG)
- Кривой площадь-объём-отметка (батиметрия + DEM)

**Ключевые характеристики:**
- Модульная архитектура Python
- Интерактивное web-приложение (Streamlit)
- Поддержка детерминированного и ансамблевого прогноза
- Сезонно-трендовая декомпозиция (Theil-Sen)
- Реконструкция гипсометрической кривой из DEM

**Технический стек:**
- Python 3.10+
- Pandas, NumPy, SciPy (анализ данных)
- Rasterio, Shapely (геопространственные операции)
- Streamlit (web-интерфейс)
- Plotly (визуализация)

---

## Архитектура системы

```
┌─────────────────────────────────────────────────────────────────┐
│                    Входные данные (Источники)                   │
├─────────────────────────────────────────────────────────────────┤
│  Sentinel-2    │    GLEAM      │    ERA5/IMERG    │   DEM        │
│  (NDWI, МСИС)  │  (испарение)  │   (осадки)       │ (батиметрия) │
└────────┬────────┴────────┬──────┴────────┬────────┴───────┬──────┘
         │                 │               │                │
         ├─────────────────┴───────────────┴────────────────┤
         │                                                  │
         │  ┌────────────────────────────────────────────┐  │
         ├─→│   ЭТАП 1: Обработка и подготовка данных  │←─┤
         │  ├────────────────────────────────────────────┤  │
         │  │ • ingest_sentinel.py      (площадь)       │  │
         │  │ • era5_process.py         (осадки)        │  │
         │  │ • gleam_processor.py      (испарение)     │  │
         │  │ • reconstruct_hypsometry.py (кривая V-A) │  │
         │  └────────────────┬─────────────────────────┘  │
         │                   │                             │
         │  ┌────────────────▼─────────────────────────┐  │
         │  │  ЭТАП 2: Централизованное хранилище    │  │
         │  ├────────────────────────────────────────────┤  │
         │  │ CSV: area_km2, volume_mcm, осадки, ET    │  │
         │  │ TIFF: integrated_dem_reconstructed.tif   │  │
         │  │ CSV: area_volume_curve_reconstructed.csv │  │
         │  └────────────────┬─────────────────────────┘  │
         │                   │                             │
         │  ┌────────────────▼─────────────────────────┐  │
         │  │   ЭТАП 3: Ядро модели (wbm пакет)     │  │
         │  ├────────────────────────────────────────────┤  │
         │  │ • simulate.py       (основное уравнение)  │  │
         │  │ • forecast.py       (прогноз)             │  │
         │  │ • seasonal.py       (сезонность)          │  │
         │  │ • ensemble.py       (ансамбль)            │  │
         │  │ • analysis.py       (тренды)              │  │
         │  │ • curve.py          (V-A интерполяция)    │  │
         │  └────────────────┬─────────────────────────┘  │
         │                   │                             │
         │  ┌────────────────▼─────────────────────────┐  │
         │  │  ЭТАП 4: Web-приложение (Streamlit)   │  │
         │  ├────────────────────────────────────────────┤  │
         │  │ app.py → wbm/ui/ (интерактивный UI)      │  │
         │  │  • Тренды и прогнозы                      │  │
         │  │  • Сценарии и параметры                  │  │
         │  │  • Визуализация графиков                 │  │
         │  │  • Загрузка результатов                  │  │
         │  └────────────────────────────────────────────┘  │
         │                   │                             │
         └───────────────────┼─────────────────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │  Выходные данные   │
                  ├────────────────────┤
                  │ CSV сценариев      │
                  │ Статистика         │
                  │ Визуализация       │
                  └────────────────────┘
```

---

## Структура проекта

```
water-balance-app/
│
├── app.py                          # Главный Streamlit ортестратор
├── requirements.txt                # Зависимости Python
├── README.md                       # Документация (ENG + RUS)
│
├── wbm/                            # ⭐ Основной пакет модели
│   ├── __init__.py
│   ├── simulate.py                 # Основное уравнение баланса
│   ├── forecast.py                 # Детерминированный прогноз
│   ├── forecast_ensemble.py        # Ансамблевый прогноз
│   ├── seasonal.py                 # Сезонная декомпозиция
│   ├── trends.py                   # Анализ трендов (Theil-Sen)
│   ├── ensemble.py                 # Блочный бутстрап
│   ├── analysis.py                 # Анализ: корреляции, лаги
│   ├── curve.py                    # Интерполяция V↔A↔E
│   ├── data.py                     # Загрузка и санитизация данных
│   ├── db.py                       # Работа с БД (опционально)
│   ├── plots.py                    # Plotly визуализация
│   ├── paths.py                    # Управление путями файлов
│   │
│   └── ui/                         # Web-интерфейс (Streamlit)
│       ├── __init__.py
│       ├── app_state.py            # Состояние приложения
│       ├── controls.py             # Фабрика управления параметрами
│       ├── data.py                 # Загрузка данных для UI
│       ├── simulation.py           # Драйвер сценариев
│       └── sections/               # Дискретные панели UI
│           ├── trends.py           # График трендов
│           ├── snow_temp.py        # Снег и температура
│           ├── runoff_temp.py      # Сток и температура
│           ├── p_et_diag.py        # Диагностика P & ET
│           ├── phase.py            # Фазовый анализ
│           ├── ensemble.py         # Ансамбль и квантили
│           └── map_view.py         # Картографический вид
│
├── app_modules/                    # Модули управления приложением
│   ├── state.py                    # Состояние (альтернативное)
│   ├── controls.py                 # Управление параметрами
│   └── data.py                     # Работа с данными
│
├── processing_data/                # Обработка данных
│   ├── DEM(hh)/
│   │   ├── reconstruct_hypsometry.py # ⭐ Реконструкция кривой V-A
│   │   └── ...
│   │
│   ├── dem_processor.py            # Обработка DEM
│   ├── era5_process_csv.py         # Обработка ERA5 → CSV
│   ├── gleam_processor.py          # Обработка GLEAM → CSV
│   ├── ingest_sentinel.py          # ⭐ Обработка Sentinel-2
│   ├── gee.py                      # Google Earth Engine интеграция
│   └── ...
│
├── processing_output/              # 📊 Выходные артефакты
│   ├── integrated_dem_reconstructed.tif        # Синтезированная DEM
│   ├── area_volume_curve_reconstructed.csv     # Реконструированная кривая
│   ├── area_volume_curve.csv                   # Исходная кривая
│   ├── bathymetry_reprojected_epsg4326.tif     # Батиметрия
│   ├── integrated_bathymetry_copernicus.tif    # Интегрированная батиметрия
│   ├── ndwi_mask_*.tif                        # NDWI маски
│   └── ...
│
├── water_balance_output/           # 📈 Результаты моделирования
│   ├── water_balance_final.csv      # Базовый сценарий
│   ├── water_balance_statistics.txt # Статистика
│   ├── scenario_test.csv            # Тестовый сценарий
│   ├── forecast_2025_gap.csv        # Прогноз пропусков
│   └── ...
│
├── Bathymetry/                     # Батиметрические данные
│   ├── Main/
│   │   ├── bathymetry_hh.tif
│   │   ├── shoreline.shp/dbf/prj/shx
│   │   └── Interpolation.py / Kriging.py
│   └── ...
│
├── Hillshade + DEM/                # Рельефные данные (DEM)
│   ├── output_hh.tif               # DEM водосбора
│   └── ...
│
├── Santinel/                       # Данные Sentinel-2 (снимки МСИС)
│   ├── ndwi_output_masked/         # Готовые NDWI маски
│   │   ├── ndwi_mask_035.tif
│   │   └── ...
│   ├── 2020/ 2021/ 2022/ 2023/ 2024/ 2025/
│   │   ├── True_Color/
│   │   ├── NDVI/
│   │   ├── NDWI/
│   │   └── ...
│   └── ndwi.py                     # Расчёт NDWI
│
├── GLEAM/                          # Данные испарения
│   ├── processed/
│   │   ├── gleam_summary_2020.csv
│   │   └── ...
│   ├── download.py / download_fixed.py
│   └── gleam_processor.py
│
├── raw_nc/                         # Сырые NetCDF файлы
│   ├── temperature/
│   ├── precipitation/
│   ├── snow/
│   ├── runoff/
│   ├── total_evaporation/
│   └── ...
│
├── raw_data/                       # Сырые данные
│   └── Hillshade + DEM/
│       └── output_hh.tif           # Исходный DEM
│
├── documentation/                  # Документация проекта
│   ├── Project_Documentation.md
│   └── Документ по логике модели.md
│
└── [прочие файлы конфигурации]

```

---

## Входные данные и источники

### 1. Sentinel-2 (Площадь поверхности воды)

**Источник:** Европейское космическое агентство (ESA)  
**Продукт:** L2A (уровень приземной обработки)  
**Спектральные каналы:**
- B03 (красный, 660 нм) – для True Color
- B08 (ближний ИК, 842 нм) – для NDVI/NDWI
- SCL (Scene Classification) – маска облаков и качества

**Расположение:** `Santinel/YYYY/` (структурирован по годам)

**Обработка:** 
```python
# NDWI = (B08 - B04) / (B08 + B04)
# Маска воды: NDWI > 0.3
# Площадь = Σ(пиксели воды) × размер пикселя
```

**Выход:** `processing_output/ndwi_mask_035.tif` (маска NDWI ≥ 0.35)

---

### 2. GLEAM (Испарение)

**Источник:** Global Land Evaporation Amsterdam Model (Martens et al.)  
**Частота:** Суточная  
**Переменная:** `E` (испарение, мм/день)  
**Расположение:** `GLEAM/processed/gleam_summary_YYYY.csv`  
**Формат:**
```
date,evaporation_mm
2020-01-01,2.5
2020-01-02,2.3
...
```

---

### 3. ERA5 / IMERG (Осадки)

**Источник:** ECMWF (European Center for Medium-Range Weather Forecasts)  
**Переменная:** Осадки (мм/день)  
**Расположение:** `precipitation_timeseries.csv`  
**Формат:**
```
date,mean_precip_mm_per_h,precipitation_mm
2020-01-01,0.0,0.0
...
```

---

### 4. Батиметрия и DEM

**Батиметрия (озеро):**
- Файл: `Bathymetry/Main/bathymetry_hh.tif` (исходный)
- Размер: высокое разрешение (~10 м)
- Формат: GeoTIFF, отметки в метрах (отрицательные = глубины)
- Интерполяция: kriging / IDW

**DEM (водосбор):**
- Файл 1: `raw_data/Hillshade + DEM/output_hh.tif` (исходный)
- Файл 2: `processing_output/integrated_bathymetry_copernicus.tif` (Copernicus)
- Файл 3: `processing_output/bathymetry_reprojected_epsg4326.tif` (репроецированный)

**Синтезированный DEM:**
- Файл: `processing_output/integrated_dem_reconstructed.tif`
- Создан: скриптом `reconstruct_hypsometry.py`
- Объединяет батиметрию и DEM рельефа

---

### 5. Кривая Площадь-Объём-Отметка

**Исходная кривая:**
- Файл: `processing_output/area_volume_curve.csv`
- Столбцы: `elevation_m, area_km2, volume_mcm`
- Источник: Ручная интерполяция

**Реконструированная кривая:**
- Файл: `processing_output/area_volume_curve_reconstructed.csv`
- Создана: `reconstruct_hypsometry.py`
- Метод: Интеграция по гипсометрической кривой DEM
- Калибровка: $A_{max} = 93.7$ км², $V_{max} = 791$ MCM, $H_{max} = 19.8$ м

---

## Процессы обработки данных

### 1. Обработка Sentinel-2 → Площадь

**Скрипт:** `processing_data/ingest_sentinel.py`  
**Вход:** Снимки Sentinel L2A (МСИС, B03, B08, SCL)  
**Выход:** CSV с площадью и объёмом по дате

**Алгоритм:**
```
1. Для каждого снимка Sentinel-2:
   a. Загрузить B08 (ИК), B04 (красный), SCL (качество)
   b. Вычислить NDWI = (B08 - B04) / (B08 + B04)
   c. Создать маску воды: NDWI > порог (обычно 0.3)
   d. Убрать облачные пиксели (SCL < 4)
   e. Вычислить площадь: Σ(пиксели) × 10 м² → км²
   f. Область фильтра: площадь ≥ 40 км² (исключить шум)

2. Конверсия площадь → объём:
   Режим --volume-mode:
   - "dem": использовать DEM интеграцию (предпочтительно)
   - "curve": использовать интерполяцию кривой
   - "auto": DEM если доступен, иначе curve (в диапазоне)

3. Выходной CSV: columns = [date, area_km2, volume_mcm, volume_source]
   volume_source = "DEM" или "curve"
```

**Команда:**
```bash
python processing_data/ingest_sentinel.py \
  --volume-mode dem \
  --min-area-km2 40 \
  --out processing_output/sentinel_area_volume.csv
```

---

### 2. Обработка GLEAM → Испарение

**Скрипт:** `processing_data/gleam_processor.py`  
**Вход:** NetCDF GLEAM (E, мм/день)  
**Выход:** CSV временного ряда испарения

**Структура GLEAM:**
```
GLEAM/
├── E/                     # Испарение
│   ├── 2020/ 2021/ ... 2025/
│   │   ├── E_YYYY0101.nc
│   │   ├── E_YYYY0102.nc
│   │   └── ...
```

**Выход:** `GLEAM/processed/gleam_summary_YYYY.csv`
```
date,evaporation_mm
2020-01-01,2.5
...
```

---

### 3. Реконструкция гипсометрической кривой

**Скрипт:** `processing_data/DEM(hh)/reconstruct_hypsometry.py`

**Назначение:** Синтезировать кривую V-A из батиметрии и DEM

**Входные параметры:**
```
--bathy <путь_батиметрия>       # высокоточная батиметрия озера
--dem <путь_dem>                 # DEM водосбора (обычно Copernicus)
--amax <значение>                # целевая макс.площадь (км²)
--vmax <значение>                # целевой макс.объём (MCM)
--hmax <значение>                # целевой макс.диапазон (м)
--out <папка>                    # выходная папка
--ndwi-mask <путь_маска>         # опциональная маска воды
--auto-buffer                    # автоматическое буферирование
--overwrite                      # перезаписать существующие
```

**Алгоритм:**
```
1. Загрузить батиметрию и DEM из разных CRS
2. Перепроектировать в общую CRS (обычно UTM)
3. Объединить: батиметрия (приоритет внизу) + DEM (приоритет вверху)
4. Создать DEM сетку уровней от min_elev до max_elev
5. Для каждого уровня: вычислить площадь и объём численно
6. Откалибровать: масштабировать по отношению к (A_max, V_max, H_max)
7. Сохранить кривую и синтезированный DEM (TIFF)
```

**Выходные файлы:**
```
processing_output/
├── integrated_dem_reconstructed.tif          # Синтезированная DEM
├── area_volume_curve_reconstructed.csv       # Кривая V-A-E
└── area_volume_curve_reconstructed.png       # График
```

**Пример команды:**
```bash
python "processing_data/DEM(hh)/reconstruct_hypsometry.py" \
  --bathy "processing_output/bathymetry_reprojected_epsg4326.tif" \
  --dem "processing_output/integrated_bathymetry_copernicus.tif" \
  --out "processing_output" \
  --amax 93.7 --vmax 791 --hmax 19.8 \
  --auto-buffer --buffer-step 50 \
  --overwrite
```

---

## Ядро модели водного баланса

### Уравнение баланса

Основное уравнение на каждый день $t$:

$$\Delta V(t) = P(t) - ET(t) + R(t) + q_{in}(t) - q_{out}(t)$$

Где:
- $\Delta V(t)$ – изменение объёма (MCM/день)
- $P(t)$ – объём осадков (MCM/день)
- $ET(t)$ – объём испарения (MCM/день)
- $R(t)$ – объём стока (MCM/день, если доступен)
- $q_{in}(t)$ – внешний приток (MCM/день)
- $q_{out}(t)$ – водозабор (MCM/день)

**Конверсия в объём на единицу осадков:**
$$P(t) = P_{mm}(t) \times A(t) / 1000$$

Где:
- $P_{mm}(t)$ – осадки (мм/день)
- $A(t)$ – площадь поверхности (км²)

---

### Модули ядра (wbm/)

#### 1. **simulate.py** – Основное уравнение

```python
def simulate_forward(
    start_date, end_date,
    init_volume_mcm,      # Начальный объём
    p_clim, et_clim,      # Сезонные климатологии (DOY-индексированы)
    vol_to_area,          # Функция V → A (из кривой)
    p_scale=1.0, et_scale=1.0,  # Масштабирование факторов
    p_daily=None, et_daily=None, # Опциональные временные ряды
    q_in_mcm_per_day=0.0,
    q_out_mcm_per_day=0.0
) → DataFrame:
    """
    Итеративный расчёт дневного баланса.
    Выход: [date, volume_mcm, area_km2, precipitation_volume_mcm, 
            evaporation_volume_mcm, delta_volume_mcm, residual_mcm]
    """
```

**Алгоритм:**
```
for каждый день t:
    1. area_km2[t] = vol_to_area(volume_mcm[t])
    2. p_mm[t] = p_clim[DOY] × p_scale  (или p_daily[t])
    3. et_mm[t] = et_clim[DOY] × et_scale  (или et_daily[t])
    4. p_mcm[t] = p_mm[t] × area_km2[t] / 1000
    5. et_mcm[t] = et_mm[t] × area_km2[t] / 1000
    6. volume_mcm[t+1] = volume_mcm[t] + p_mcm[t] - et_mcm[t] + q_in - q_out
    7. delta_volume_mcm[t] = volume_mcm[t+1] - volume_mcm[t]
    8. residual_mcm[t] = delta_mcm[t] - p_mcm[t] + et_mcm[t]
```

---

#### 2. **curve.py** – Интерполяция V↔A

```python
def load_curve(csv_path) → Callable:
    """Загрузить кривую V-A-E и вернуть интерполятор."""
    df = pd.read_csv(csv_path)  # [elevation_m, area_km2, volume_mcm]
    
    # Создать интерполяторы
    vol_to_area = interp1d(df.volume_mcm, df.area_km2, 
                          kind='linear', bounds_error=False,
                          fill_value='extrapolate')
    area_to_vol = interp1d(df.area_km2, df.volume_mcm, ...)
    elev_from_vol = interp1d(df.volume_mcm, df.elevation_m, ...)
    
    return vol_to_area, area_to_vol, elev_from_vol
```

**Особенность:** Использует scipy.interpolate.interp1d с `fill_value='extrapolate'` для выхода за пределы диапазона кривой (осторожно: может привести к неправильным результатам!)

---

#### 3. **data.py** – Загрузка данных

```python
def load_baseline(csv_path: str) → DataFrame:
    """Загрузить базовый выполненный временной ряд."""
    
def load_gleam(year: int) → Series:
    """Загрузить GLEAM испарение на год (DOY-индексировано)."""
    
def load_precipitation(csv_path: str) → Series:
    """Загрузить осадки (дата-индексированы)."""
```

---

#### 4. **forecast.py** – Детерминированный прогноз

```python
def build_robust_season_trend_series(
    hist_df: DataFrame,      # История: [date, value]
    start_date, end_date,
    season_basis='DOY',      # 'DOY' или 'MONTH'
    min_history=365
) → Series:
    """
    Декомпозиция: Value = Season(DOY) + Trend(t) + Residual(t)
    
    1. Вычислить сезонный медиан для каждого DOY
    2. Вычесть сезон → остаток (deseasonalised)
    3. Применить Theil-Sen регрессию к остатку
    4. Продлить: Season(DOY) + Trend_slope × t
    
    Выход: детерминированный прогноз [start_date, end_date]
    """
```

**Theil-Sen оценка:**
$$\beta = \text{median}\left(\frac{x_j - x_i}{t_j - t_i}\right) \text{ для всех } i < j$$

Робастна к выбросам; используется вместо OLS.

---

#### 5. **seasonal.py** – Сезонная декомпозиция

```python
def seasonal_spread(
    hist_df: DataFrame,
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
) → DataFrame:
    """
    Для каждого DOY вычислить эмпирические квантили.
    Выход: [doy, p10, p25, p50, p75, p90]
    
    Используется для визуализации неопределённости в UI.
    """
```

---

#### 6. **ensemble.py** – Ансамблевый прогноз

```python
def ensemble_forecast(
    hist_residuals: Series,      # Остатки (осадков, обычно)
    n_members=100,
    block_length='auto',         # ACF или ручной
    n_bootstrap_samples=None
) → DataFrame:
    """
    Блочный бутстрап остатков:
    1. Декомпозировать историю: Value = Trend + Season + Residual
    2. Определить длину блока (ACF автокорреляции)
    3. Для каждого члена ансамбля:
       a. Заново отбить случайные блоки остатков
       b. Пересоздать: Value_ens = Trend + Season + Residual_boot
       c. Пропустить через симуляцию объёма
    4. Вычислить квантили (p5, p25, p50, p75, p95)
    
    Выход: матрица ансамбля [дата × член]
    """
```

---

#### 7. **analysis.py** – Анализ трендов

```python
def theil_sen_slope(x, y) → float:
    """Коэффициент Theil-Sen для (x, y)."""
    
def kendall_tau_pvalue(x, y) → float:
    """Тест Kendall Tau на тренд."""
    
def rolling_correlation(df, col1, col2, window=90) → Series:
    """Скользящая корреляция."""
```

---

### Основные рабочие процессы

#### Сценарий 1: Базовая симуляция

```python
# 1. Загрузить данные
baseline_df = load_baseline("water_balance_final.csv")
gleam_yr_clim = load_gleam_climatology()
precip_clim = load_precip_climatology()
vol_to_area, area_to_vol, elev_from_vol = load_curve(
    "area_volume_curve_reconstructed.csv"
)

# 2. Запустить симуляцию
result_df = simulate_forward(
    start_date=pd.Timestamp("2025-01-01"),
    end_date=pd.Timestamp("2025-12-31"),
    init_volume_mcm=baseline_df.iloc[-1]['volume_mcm'],
    p_clim=precip_clim,
    et_clim=gleam_yr_clim,
    vol_to_area=vol_to_area,
    p_scale=1.1,  # Увеличить осадки на 10%
    et_scale=0.95,  # Снизить испарение на 5%
)

# 3. Сохранить
result_df.to_csv("scenario_output.csv", index=False)
```

---

#### Сценарий 2: Прогноз с ансамблем

```python
# 1. Построить сезонно-трендовый прогноз осадков
det_precip_fcast = build_robust_season_trend_series(
    precip_clim.reset_index(),
    start_date="2025-01-01",
    end_date="2025-12-31"
)

# 2. Создать ансамбль остатков
residuals = precip_clim.values - det_precip_fcast.values
ensemble_precip = ensemble_forecast(
    hist_residuals=residuals,
    n_members=100,
    block_length='auto'
)

# 3. Пропустить каждый член ансамбля через симуляцию
ensemble_volumes = []
for member_idx in range(100):
    p_daily = ensemble_precip[member_idx]
    vol_member = simulate_forward(
        ...,
        p_daily=p_daily
    )
    ensemble_volumes.append(vol_member['volume_mcm'].values)

# 4. Вычислить квантили
ensemble_array = np.array(ensemble_volumes)  # [100, n_days]
quantiles = np.percentile(ensemble_array, [5, 25, 50, 75, 95], axis=0)
```

---

## Выходные данные

### 1. CSV результаты

#### `water_balance_final.csv`
```
date,volume_mcm,area_km2,precipitation_volume_mcm,evaporation_volume_mcm,delta_volume_mcm,residual_mcm,elevation_m
2020-03-28,170.639,55.23,0.000,0.055,NaN,NaN,0.45
2020-03-29,170.683,55.45,0.000,0.054,0.044,0.098,...
...
```

**Столбцы:**
- `date` – дата (YYYY-MM-DD)
- `volume_mcm` – объём (млн. м³)
- `area_km2` – площадь зеркала (км²)
- `precipitation_volume_mcm` – осадки (MCM/день)
- `evaporation_volume_mcm` – испарение (MCM/день)
- `delta_volume_mcm` – изменение объёма
- `residual_mcm` – остаток (неучтённые потоки)
- `elevation_m` – уровень воды (м)

---

#### `water_balance_statistics.txt`

```
СТАТИСТИКА МОДЕЛИ ВОДНОГО БАЛАНСА
========================================

Период анализа: 2020-03-28 — 2025-07-16
Количество дней: 1937

КОМПОНЕНТЫ БАЛАНСА (среднее ± стд.откл.):
----------------------------------------
volume_mcm: 170.639 ± 44.195 млн.м³
delta_volume_mcm: 0.044 ± 4.183 млн.м³/день
precipitation_volume_mcm: 0.000 ± 0.000 млн.м³/день
evaporation_volume_mcm: 0.055 ± 0.062 млн.м³/день
residual_mcm: 0.061 ± 4.081 млн.м³/день

ЭКСТРЕМУМЫ:
-------------------------------------------
volume_mcm: min = 75.423, max = 252.891 MCM
area_km2: min = 28.456, max = 93.712 км²
```

---

### 2. Визуализация (Plotly)

**Интерактивные графики в web-приложении:**

1. **Тренды** – временной ряд объёма с трендом Theil-Sen
2. **Компоненты баланса** – разложение P, ET, ΔV, Residual
3. **Сезонность** – боксплоты по месяцам / DOY
4. **Ансамбль** – веер прогноза (p5, p25, p50, p75, p95)
5. **Фаза** – диаграмма рассеяния (ΔV vs. Residual)
6. **Карта** – картографический вид NDWI маски (Folium)

---

## Web-приложение (Streamlit)

### Запуск

```bash
streamlit run app.py --server.port 8501
```

### Структура UI (`wbm/ui/`)

**`app.py` (главный файл):**
```python
import streamlit as st
from wbm.ui.controls import build_sidebar_controls
from wbm.ui.simulation import run_scenario
from wbm.ui.sections import (
    show_trends, show_ensemble, show_seasonal_spread,
    show_p_et_diagnostics, show_phase_analysis
)

def main():
    st.set_page_config(layout="wide")
    st.title("💧 Модель Водного Баланса")
    
    # Панель управления
    params = build_sidebar_controls()
    
    # Запустить сценарий
    result_df = run_scenario(params)
    
    # Показать результаты
    show_trends(result_df)
    show_ensemble(result_df)
    show_seasonal_spread(result_df)
    show_p_et_diagnostics(result_df)
    show_phase_analysis(result_df)
    
    # Загрузка
    csv = result_df.to_csv(index=False)
    st.download_button("📥 CSV сценария", csv)

if __name__ == "__main__":
    main()
```

---

### Параметры управления

**Sidebar контролы:**

| Параметр | Тип | Диапазон | По умолчанию |
|----------|------|----------|----------|
| Начальная дата | date | 2020–2025 | min(данные) |
| Конечная дата | date | 2020–2025 | max(данные) |
| Начальный объём | float | 0–300 MCM | данные |
| Масштаб осадков | slider | 0.5–2.0 | 1.0 |
| Масштаб испарения | slider | 0.5–2.0 | 1.0 |
| Приток (q_in) | float | 0–10 MCM/д | 0 |
| Водозабор (q_out) | float | 0–10 MCM/д | 0 |
| Ансамбль – количество | int | 10–500 | 100 |
| Ансамбль – основа сезона | radio | DOY / MONTH | DOY |
| Ансамбль – длина блока | radio | Auto / Manual | Auto |

---

### Секции результатов

1. **Trends** (`wbm/ui/sections/trends.py`)
   - Временной ряд объёма
   - Регрессия Theil-Sen (линия тренда)
   - Доверительные интервалы

2. **Ensemble** (`wbm/ui/sections/ensemble.py`)
   - Веер квантилей (p5, p25, p50, p75, p95)
   - Диагностика сходимости
   - Гистограмма распределения

3. **Seasonal Spread** (`wbm/ui/sections/...`)
   - Боксплоты по месяцам
   - Спектр сезонной изменчивости

4. **P & ET Diagnostics** (`wbm/ui/sections/p_et_diag.py`)
   - Временные ряды осадков и испарения
   - Сезонная климатология

5. **Phase Analysis** (`wbm/ui/sections/phase.py`)
   - Диаграмма рассеяния (ΔV vs. Residual)
   - Корреляция компонентов

6. **Map View** (`wbm/ui/sections/map_view.py`)
   - Интерактивная карта (Folium)
   - NDWI маска, батиметрия

---

## Установка и развёртывание

### 1. Локальная установка (Windows)

```powershell
# Клонировать репозиторий
git clone https://github.com/vel5id/water-balance-app.git
cd water-balance-app

# Создать виртуальное окружение
python -m venv .venv
.venv\Scripts\Activate.ps1

# Установить зависимости
pip install -r requirements.txt

# Установить переменную окружения (DATA_ROOT)
$env:DATA_ROOT = 'C:\path\to\data'

# Запустить приложение
streamlit run app.py --server.port 8501
```

### 2. Docker развёртывание

```dockerfile
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.headless", "true"]
```

```bash
docker build -t water-balance-app .
docker run -p 8501:8501 \
  -e DATA_ROOT=/data \
  -v /path/to/data:/data \
  water-balance-app
```

### 3. Требуемые зависимости

```
pandas>=1.3.0
numpy>=1.20.0
scipy>=1.7.0
rasterio>=1.2.0
shapely>=1.8.0
streamlit>=1.0.0
plotly>=5.0.0
folium>=0.12.0
```

---

## Примеры использования

### Пример 1: Расчёт базового сценария

```python
from wbm import simulate, data, curve

# Загрузить исходные данные
baseline_df = data.load_baseline("water_balance_final.csv")
gleam_clim = data.load_gleam_climatology()
precip_clim = data.load_precipitation_climatology()

# Загрузить кривую
vol_to_area, _, _ = curve.load_curve("area_volume_curve_reconstructed.csv")

# Запустить симуляцию
result = simulate.simulate_forward(
    start_date=pd.Timestamp("2025-01-01"),
    end_date=pd.Timestamp("2025-12-31"),
    init_volume_mcm=170.6,
    p_clim=precip_clim,
    et_clim=gleam_clim,
    vol_to_area=vol_to_area,
    p_scale=1.0,
    et_scale=1.0
)

# Сохранить
result.to_csv("output.csv", index=False)
print(result.head())
```

---

### Пример 2: Сценарий с изменением климата

```python
# Сценарий: осадки +20%, испарение +10%
result_rcp = simulate.simulate_forward(
    start_date=pd.Timestamp("2025-01-01"),
    end_date=pd.Timestamp("2050-12-31"),
    init_volume_mcm=170.6,
    p_clim=precip_clim,
    et_clim=gleam_clim,
    vol_to_area=vol_to_area,
    p_scale=1.2,  # +20%
    et_scale=1.1   # +10%
)

print(f"Средний объём: {result_rcp['volume_mcm'].mean():.2f} MCM")
print(f"Макс. объём: {result_rcp['volume_mcm'].max():.2f} MCM")
print(f"Мин. объём: {result_rcp['volume_mcm'].min():.2f} MCM")
```

---

### Пример 3: Ансамблевый прогноз

```python
from wbm import ensemble

# История остатков
residuals = baseline_df['residual_mcm'].dropna()

# Ансамбль
ens_df = ensemble.ensemble_forecast(
    hist_residuals=residuals,
    n_members=200,
    block_length='auto'
)

print(f"Ансамбль размер: {ens_df.shape}")
# Каждый столбец – один член ансамбля
quantiles = ens_df.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
print(quantiles)
```

---

## Заключение

Эта модель представляет собой комплексную систему для:
- **Обработки** спутниковых (Sentinel-2), климатических (ERA5/GLEAM) и батиметрических данных
- **Симуляции** водного баланса на основе физического уравнения
- **Прогнозирования** объёма с детерминированными и ансамблевыми методами
- **Взаимодействия** пользователя через интерактивное web-приложение

**Ключевые инновации:**
1. Синтезированная DEM из батиметрии + цифровой рельеф
2. Робастная сезонно-трендовая декомпозиция (Theil-Sen)
3. Блочный бутстрап для ансамблевого прогноза
4. Модульная архитектура для гибкости и расширения

---

**Контакты и поддержка:**
- Репозиторий: https://github.com/vel5id/water-balance-app
- Лицензия: MIT
- Дата документации: Октябрь 2025
