"""
Интерактивная модель водного баланса
====================================

Объединяет данные Sentinel-2, GLEAM, IMERG и кривую "Площадь-Объём"
для расчета компонентов водного баланса водохранилища.

Автор: GitHub Copilot
Дата: 2025-08-07
"""

import os
from pathlib import Path
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- КОНФИГУРАЦИЯ ---

# Входные данные (относительно каталога проекта)
ROOT = Path(__file__).resolve().parent
SENTINEL_ROOT_DIR = str(ROOT)
GLEAM_DATA_PATH = str(ROOT / 'GLEAM' / 'processed' / 'gleam_summary_all_years.csv')
IMERG_DATA_PATH = str(ROOT / 'precipitation_timeseries.csv')
AREA_VOLUME_CURVE_PATH = str(ROOT / 'processing_output' / 'area_volume_curve.csv')

# Выходные данные
OUTPUT_DIR = str(ROOT / 'water_balance_output')

# Параметры модели
NDWI_THRESHOLD = 0.275  # Порог для определения воды
INTERPOLATION_METHOD = 'linear'  # Метод интерполяции для заполнения пропусков

print("=== МОДЕЛЬ ВОДНОГО БАЛАНСА ===")
print("Автор: GitHub Copilot")
print("================================\n")

# --- МОДУЛЬ 1: ОБРАБОТКА SENTINEL-2 ---

def find_sentinel_files(root_dir):
    """Находит все доступные наборы Sentinel-2 файлов."""
    print("🛰️  Поиск снимков Sentinel-2...")
    
    files = glob.glob(os.path.join(root_dir, '**', '*_Sentinel-2_L2A_*.tiff'), recursive=True)
    file_groups = {}
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
    
    for f in files:
        match = date_pattern.search(os.path.basename(f))
        if not match:
            continue
        date_str = match.group(1)
        if date_str not in file_groups:
            file_groups[date_str] = {}
        
        if '_B03_' in f:
            file_groups[date_str]['b03'] = f
        elif '_B08_' in f:
            file_groups[date_str]['b08'] = f
        elif 'Scene_classification_map' in f:
            file_groups[date_str]['scl'] = f
    
    # Оставляем только полные наборы
    valid_groups = {d: p for d, p in file_groups.items() 
                   if 'b03' in p and 'b08' in p and 'scl' in p}
    
    sorted_dates = sorted(valid_groups.keys())
    print(f"   Найдено {len(sorted_dates)} полных наборов снимков")
    if sorted_dates:
        print(f"   Период: {sorted_dates[0]} — {sorted_dates[-1]}")
    
    return [(d, valid_groups[d]['b03'], valid_groups[d]['b08'], valid_groups[d]['scl']) 
            for d in sorted_dates]

def calculate_water_area(b03_path, b08_path, scl_path, pixel_area_km2_cache):
    """Рассчитывает площадь водной поверхности в км² для одного снимка."""
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        
        # Читаем данные и используем B03 (Green) как основу
        with rasterio.open(b03_path) as green_src:
            green = green_src.read(1).astype('float32')
            profile = green_src.profile
            target_shape = green_src.shape
            
            # Получаем точный размер пикселя в км² (только один раз)
            if pixel_area_km2_cache is None:
                transform = green_src.transform
                crs = green_src.crs
                # Вычисляем площадь пикселя с учётом типа CRS
                if crs is not None and not crs.is_geographic:
                    # Проектed CRS: единицы в метрах. Площадь пикселя = |a*e - b*d| (детерминант аффинной матрицы)
                    a = transform.a
                    b = transform.b
                    d_ = transform.d
                    e = transform.e
                    pixel_area_m2 = abs(a * e - b * d_)
                    pixel_area_km2 = pixel_area_m2 / 1_000_000.0
                else:
                    # Географическая CRS (градусы): конвертируем градусы в метры с поправкой на широту
                    bounds = green_src.bounds
                    pixel_width_deg = abs(transform.a)
                    pixel_height_deg = abs(transform.e)
                    center_lat = (bounds.bottom + bounds.top) / 2
                    # Конверсия градусов в метры
                    m_per_deg_lat = 111132.954
                    m_per_deg_lon = 111320.0 * float(np.cos(np.radians(center_lat)))
                    pixel_area_m2 = (pixel_width_deg * m_per_deg_lon) * (pixel_height_deg * m_per_deg_lat)
                    pixel_area_km2 = pixel_area_m2 / 1_000_000.0
            else:
                pixel_area_km2 = pixel_area_km2_cache

        # Перепроецируем NIR (B08) в разрешение Green-канала для точного совмещения
        with rasterio.open(b08_path) as nir_src:
            nir = np.empty(target_shape, dtype='float32')
            reproject(
                source=nir_src.read(1),
                destination=nir,
                src_transform=nir_src.transform,
                src_crs=nir_src.crs,
                dst_transform=green_src.transform,
                dst_crs=green_src.crs,
                resampling=Resampling.bilinear
            )

        # Перепроецируем SCL в разрешение Green-канала
        with rasterio.open(scl_path) as scl_src:
            scl_data = np.empty(target_shape, dtype='uint8')
            reproject(
                source=scl_src.read(1),
                destination=scl_data,
                src_transform=scl_src.transform,
                src_crs=scl_src.crs,
                dst_transform=green_src.transform,
                dst_crs=green_src.crs,
                resampling=Resampling.nearest
            )
        
        # Маскируем облака
        cloud_mask = np.isin(scl_data, [3, 8, 9, 10, 11])
        green[cloud_mask] = np.nan
        nir[cloud_mask] = np.nan

        # Вычисляем NDWI
        np.seterr(divide='ignore', invalid='ignore')
        ndwi = (green - nir) / (green + nir)
        
        # Создаем маску воды
        water_mask = np.nan_to_num(ndwi) > NDWI_THRESHOLD
        water_pixels = np.sum(water_mask)
        water_area_km2 = water_pixels * pixel_area_km2
        
        return water_area_km2, pixel_area_km2
        
    except Exception as e:
        print(f"   ⚠️ Ошибка обработки снимка: {e}")
        return np.nan, pixel_area_km2_cache

def process_sentinel_timeseries():
    """Создает временной ряд площадей водной поверхности из снимков Sentinel-2."""
    print("\n📊 МОДУЛЬ 1: Анализ временного ряда Sentinel-2")
    print("-" * 50)
    
    file_groups = find_sentinel_files(SENTINEL_ROOT_DIR)
    if not file_groups:
        raise FileNotFoundError("❌ Снимки Sentinel-2 не найдены!")
    
    results = []
    pixel_area_km2_cache = None
    
    print(f"🔄 Обработка {len(file_groups)} снимков...")
    for i, (date_str, b03, b08, scl) in enumerate(file_groups):
        print(f"   {i+1:2d}/{len(file_groups)} {date_str}", end=" ")
        
        area_km2, pixel_area_km2_cache = calculate_water_area(b03, b08, scl, pixel_area_km2_cache)
        
        if not np.isnan(area_km2):
            print(f"→ {area_km2:.2f} км²")
            results.append({
                'date': pd.to_datetime(date_str),
                'area_km2': area_km2,
                'source': 'Sentinel-2'
            })
        else:
            print("→ Ошибка")
    
    if not results:
        raise ValueError("❌ Ни один снимок не был обработан успешно!")
    
    df = pd.DataFrame(results)
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"\n✅ Временной ряд Sentinel-2 создан:")
    print(f"   📅 Период: {df['date'].min().strftime('%Y-%m-%d')} — {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   🌊 Площадь: {df['area_km2'].min():.2f} — {df['area_km2'].max():.2f} км²")
    if pixel_area_km2_cache is not None:
        print(f"   📏 Размер пикселя: {pixel_area_km2_cache*1_000_000:.1f} м²")
    
    return df

# --- МОДУЛЬ 2: ЗАГРУЗКА ВНЕШНИХ ДАННЫХ ---

def load_external_data():
    """Загружает данные GLEAM, IMERG и кривую площадь-объём."""
    print("\n📂 МОДУЛЬ 2: Загрузка внешних данных")
    print("-" * 50)
    
    # Проверяем наличие файлов
    files_to_check = {
        'GLEAM (испарение)': GLEAM_DATA_PATH,
        'IMERG (осадки)': IMERG_DATA_PATH,
        'Кривая площадь-объём': AREA_VOLUME_CURVE_PATH
    }
    
    for name, path in files_to_check.items():
        if os.path.exists(path):
            print(f"   ✅ {name}: найден")
        else:
            print(f"   ❌ {name}: НЕ НАЙДЕН - {path}")
    
    # Загружаем GLEAM
    print("\n📈 Загрузка данных GLEAM...")
    try:
        gleam_df = pd.read_csv(GLEAM_DATA_PATH)
        gleam_df['date'] = pd.to_datetime(gleam_df['date'])
        # Нормализация названий столбцов: используем E как evaporation_mm (мм/день)
        if 'evaporation_mm' not in gleam_df.columns:
            if 'E' in gleam_df.columns:
                gleam_df = gleam_df.rename(columns={'E': 'evaporation_mm'})
            elif 'evaporation' in gleam_df.columns:
                gleam_df = gleam_df.rename(columns={'evaporation': 'evaporation_mm'})
        print(f"   📅 Период: {gleam_df['date'].min().strftime('%Y-%m-%d')} — {gleam_df['date'].max().strftime('%Y-%m-%d')}")
        if 'evaporation_mm' in gleam_df.columns:
            print(f"   💧 Испарение: {gleam_df['evaporation_mm'].min():.2f} — {gleam_df['evaporation_mm'].max():.2f} мм/день")
        else:
            print("   ⚠️ Столбец evaporation_mm не найден в GLEAM")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки GLEAM: {e}")
        gleam_df = pd.DataFrame()
    
    # Загружаем IMERG
    print("\n🌧️  Загрузка данных IMERG...")
    try:
        imerg_df = pd.read_csv(IMERG_DATA_PATH)
        imerg_df['date'] = pd.to_datetime(imerg_df['date'])
        # Нормализация осадков: mean_precip_mm_per_h -> precipitation_mm (×24)
        if 'precipitation_mm' not in imerg_df.columns:
            if 'mean_precip_mm_per_h' in imerg_df.columns:
                imerg_df = imerg_df.rename(columns={'mean_precip_mm_per_h': 'precipitation_mm'})
                imerg_df['precipitation_mm'] = imerg_df['precipitation_mm'] * 24.0
            elif 'precipitation' in imerg_df.columns:
                imerg_df = imerg_df.rename(columns={'precipitation': 'precipitation_mm'})
        print(f"   📅 Период: {imerg_df['date'].min().strftime('%Y-%m-%d')} — {imerg_df['date'].max().strftime('%Y-%m-%d')}")
        if 'precipitation_mm' in imerg_df.columns:
            print(f"   🌧️ Осадки: {imerg_df['precipitation_mm'].min():.2f} — {imerg_df['precipitation_mm'].max():.2f} мм/день")
        else:
            print("   ⚠️ Столбец precipitation_mm не найден в IMERG")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки IMERG: {e}")
        imerg_df = pd.DataFrame()
    
    # Загружаем кривую площадь-объём
    print("\n📏 Загрузка кривой площадь-объём...")
    try:
        curve_df = pd.read_csv(AREA_VOLUME_CURVE_PATH)
        print(f"   📊 Точек на кривой: {len(curve_df)}")
        print(f"   🌊 Площадь: {curve_df['area_km2'].min():.2f} — {curve_df['area_km2'].max():.2f} км²")
        print(f"   💧 Объём: {curve_df['volume_mcm'].min():.2f} — {curve_df['volume_mcm'].max():.2f} млн.м³")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки кривой: {e}")
        curve_df = pd.DataFrame()
    
    return gleam_df, imerg_df, curve_df

# --- МОДУЛЬ 3: КОНВЕРТАЦИЯ ПЛОЩАДЬ → ОБЪЁМ ---

def area_to_volume_converter(curve_df):
    """Создает интерполятор для конвертации площади в объём."""
    print("\n🔄 МОДУЛЬ 3: Создание конвертера площадь → объём")
    print("-" * 50)
    
    if curve_df.empty:
        print("   ❌ Кривая площадь-объём недоступна!")
        return None
    
    from scipy.interpolate import interp1d
    
    # Сортируем по площади
    curve_sorted = curve_df.sort_values('area_km2')
    areas = curve_sorted['area_km2'].values
    volumes = curve_sorted['volume_mcm'].values
    
    # Создаем интерполятор
    try:
        # Важно: запрещаем экстраполяцию. Клампим значения к [min, max] кривой,
        # чтобы исключить нереалистичные объёмы при ошибках расчёта площади.
        interpolator = interp1d(
            areas,
            volumes,
            kind='linear',
            bounds_error=False,
            fill_value=(float(volumes[0]), float(volumes[-1]))
        )
        
        print(f"   ✅ Интерполятор создан")
        print(f"   📏 Диапазон площадей: {areas.min():.2f} — {areas.max():.2f} км²")
        print(f"   💧 Диапазон объёмов: {volumes.min():.2f} — {volumes.max():.2f} млн.м³")
        
        return interpolator
        
    except Exception as e:
        print(f"   ❌ Ошибка создания интерполятора: {e}")
        return None

def convert_areas_to_volumes(sentinel_df, area_to_volume_func):
    """Конвертирует временной ряд площадей в объёмы."""
    if area_to_volume_func is None:
        print("   ❌ Конвертер недоступен, используем площади как есть")
        sentinel_df['volume_mcm'] = sentinel_df['area_km2']  # Заглушка
        return sentinel_df
    
    print(f"   🔄 Конвертация {len(sentinel_df)} значений площади в объёмы...")
    
    volumes = []
    for area in sentinel_df['area_km2']:
        try:
            volume = float(area_to_volume_func(area))
            volumes.append(volume)
        except:
            volumes.append(np.nan)
    
    sentinel_df['volume_mcm'] = volumes
    
    valid_volumes = sentinel_df['volume_mcm'].dropna()
    if len(valid_volumes) > 0:
        print(f"   ✅ Объёмы: {valid_volumes.min():.2f} — {valid_volumes.max():.2f} млн.м³")
    else:
        print("   ⚠️ Не удалось конвертировать ни одного значения")
    
    return sentinel_df

# --- МОДУЛЬ 4: СИНХРОНИЗАЦИЯ И РАСЧЁТ БАЛАНСА ---

def create_unified_timeseries(sentinel_df, gleam_df, imerg_df):
    """Создает единую синхронизированную временную серию."""
    print("\n🔗 МОДУЛЬ 4: Синхронизация данных и расчёт баланса")
    print("-" * 50)
    
    # Определяем общий период
    all_dates = []
    if not sentinel_df.empty:
        all_dates.extend(sentinel_df['date'].tolist())
    if not gleam_df.empty:
        all_dates.extend(gleam_df['date'].tolist())
    if not imerg_df.empty:
        all_dates.extend(imerg_df['date'].tolist())
    
    if not all_dates:
        raise ValueError("❌ Нет данных для синхронизации!")
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    
    print(f"   📅 Общий период: {start_date.strftime('%Y-%m-%d')} — {end_date.strftime('%Y-%m-%d')}")
    print(f"   📊 Общая длительность: {(end_date - start_date).days + 1} дней")
    
    # Создаем ежедневную сетку
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    unified_df = pd.DataFrame({'date': date_range})
    
    print(f"\n🔄 Объединение данных...")
    
    # Добавляем данные Sentinel (объёмы)
    if not sentinel_df.empty:
        unified_df = unified_df.merge(
            sentinel_df[['date', 'area_km2', 'volume_mcm']], 
            on='date', how='left'
        )
        valid_sentinel = unified_df['volume_mcm'].dropna()
        print(f"   🛰️ Sentinel: {len(valid_sentinel)} дней с данными")
    else:
        unified_df['area_km2'] = np.nan
        unified_df['volume_mcm'] = np.nan
        print(f"   🛰️ Sentinel: нет данных")
    
    # Добавляем данные GLEAM (испарение)
    if not gleam_df.empty:
        unified_df = unified_df.merge(
            gleam_df[['date', 'evaporation_mm']], 
            on='date', how='left'
        )
        valid_gleam = unified_df['evaporation_mm'].dropna()
        print(f"   💧 GLEAM: {len(valid_gleam)} дней с данными")
    else:
        unified_df['evaporation_mm'] = np.nan
        print(f"   💧 GLEAM: нет данных")
    
    # Добавляем данные IMERG (осадки)
    if not imerg_df.empty:
        unified_df = unified_df.merge(
            imerg_df[['date', 'precipitation_mm']], 
            on='date', how='left'
        )
        valid_imerg = unified_df['precipitation_mm'].dropna()
        print(f"   🌧️ IMERG: {len(valid_imerg)} дней с данными")
    else:
        unified_df['precipitation_mm'] = np.nan
        print(f"   🌧️ IMERG: нет данных")
    
    return unified_df

def interpolate_missing_values(df):
    """Заполняет пропуски интерполяцией."""
    print(f"\n🔧 Интерполяция пропущенных значений ({INTERPOLATION_METHOD})...")
    
    numeric_columns = ['area_km2', 'volume_mcm', 'evaporation_mm', 'precipitation_mm']
    
    for col in numeric_columns:
        if col in df.columns:
            before_count = df[col].isna().sum()
            if before_count > 0:
                df[col] = df[col].interpolate(method=INTERPOLATION_METHOD)
                after_count = df[col].isna().sum()
                filled_count = before_count - after_count
                print(f"   📈 {col}: заполнено {filled_count} пропусков")
            else:
                print(f"   ✅ {col}: пропусков нет")
    
    return df

def calculate_water_balance(df):
    """Рассчитывает компоненты водного баланса."""
    print(f"\n⚖️ Расчёт водного баланса...")
    
    # Проверяем наличие необходимых данных
    required_cols = ['volume_mcm', 'area_km2', 'evaporation_mm', 'precipitation_mm']
    missing_cols = [col for col in required_cols if col not in df.columns or df[col].isna().all()]
    
    if missing_cols:
        print(f"   ⚠️ Недостающие данные: {missing_cols}")
        print("   🔧 Создаем нулевые значения для недостающих компонентов")
        for col in missing_cols:
            df[col] = 0.0
    
    # 1. Изменение объёма (ΔS)
    df['delta_volume_mcm'] = df['volume_mcm'].diff()  # млн.м³/день
    print(f"   📊 ΔS (изменение объёма): рассчитано")
    
    # 2. Объём осадков (P)
    # P = осадки (мм/день) × площадь (км²) × 0.001 (конвертация в млн.м³)
    df['precipitation_volume_mcm'] = df['precipitation_mm'] * df['area_km2'] * 0.001
    print(f"   🌧️ P (объём осадков): рассчитано")
    
    # 3. Объём испарения (ET)
    # ET = испарение (мм/день) × площадь (км²) × 0.001 (конвертация в млн.м³)
    df['evaporation_volume_mcm'] = df['evaporation_mm'] * df['area_km2'] * 0.001
    print(f"   💨 ET (объём испарения): рассчитано")
    
    # 4. Остаточный сток (Residual)
    # Residual = ΔS - P + ET (приток/отток не учтённый в P и ET)
    df['residual_mcm'] = (df['delta_volume_mcm'] - 
                         df['precipitation_volume_mcm'] + 
                         df['evaporation_volume_mcm'])
    print(f"   🔄 Residual (остаточный сток): рассчитано")
    
    # Убираем первую строку (NaN в delta_volume_mcm)
    df = df.dropna(subset=['delta_volume_mcm']).reset_index(drop=True)
    
    print(f"   ✅ Водный баланс рассчитан для {len(df)} дней")
    
    return df

# --- МОДУЛЬ 5: СОХРАНЕНИЕ И ВИЗУАЛИЗАЦИЯ ---

def save_results(df, output_dir):
    """Сохраняет результаты в CSV файл."""
    print(f"\n💾 Сохранение результатов...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"   📁 Создана папка: {output_dir}")
    
    # Основной файл результатов
    main_output_path = os.path.join(output_dir, 'water_balance_final.csv')
    df.to_csv(main_output_path, index=False)
    print(f"   💾 Основные результаты: {main_output_path}")
    
    # Статистика
    stats_path = os.path.join(output_dir, 'water_balance_statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("СТАТИСТИКА МОДЕЛИ ВОДНОГО БАЛАНСА\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Период анализа: {df['date'].min().strftime('%Y-%m-%d')} — {df['date'].max().strftime('%Y-%m-%d')}\n")
        f.write(f"Количество дней: {len(df)}\n\n")
        
        f.write("КОМПОНЕНТЫ БАЛАНСА (среднее ± стд.откл.):\n")
        f.write("-" * 40 + "\n")
        
        numeric_cols = ['volume_mcm', 'delta_volume_mcm', 'precipitation_volume_mcm', 
                       'evaporation_volume_mcm', 'residual_mcm']
        
        for col in numeric_cols:
            if col in df.columns and not df[col].isna().all():
                mean_val = df[col].mean()
                std_val = df[col].std()
                f.write(f"{col}: {mean_val:.3f} ± {std_val:.3f} млн.м³\n")
        
    print(f"   📊 Статистика: {stats_path}")
    
    return main_output_path

def create_visualizations(df, output_dir):
    """Создает графики компонентов водного баланса."""
    print(f"\n📈 Создание визуализаций...")
    
    # Настройка стиля
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    # Основной график - 4 панели
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    
    # 1. Объём водохранилища
    axes[0].plot(df['date'], df['volume_mcm'], 'b-', linewidth=2, alpha=0.8)
    axes[0].set_ylabel('Объём\n(млн.м³)', fontsize=12)
    axes[0].set_title('Компоненты водного баланса водохранилища', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Изменение объёма
    colors = ['red' if x < 0 else 'green' for x in df['delta_volume_mcm']]
    axes[1].bar(df['date'], df['delta_volume_mcm'], color=colors, alpha=0.6, width=1)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1].set_ylabel('ΔS\n(млн.м³/день)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Осадки и испарение
    axes[2].bar(df['date'], df['precipitation_volume_mcm'], color='blue', alpha=0.6, 
               label='Осадки (P)', width=1)
    axes[2].bar(df['date'], -df['evaporation_volume_mcm'], color='orange', alpha=0.6, 
               label='Испарение (ET)', width=1)
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[2].set_ylabel('P, ET\n(млн.м³/день)', fontsize=12)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Остаточный сток
    residual_colors = ['purple' if x < 0 else 'green' for x in df['residual_mcm']]
    axes[3].bar(df['date'], df['residual_mcm'], color=residual_colors, alpha=0.6, width=1)
    axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[3].set_ylabel('Residual\n(млн.м³/день)', fontsize=12)
    axes[3].set_xlabel('Дата', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    
    # Форматирование осей дат
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Сохранение основного графика
    main_plot_path = os.path.join(output_dir, 'water_balance_timeseries.png')
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📊 Временные ряды: {main_plot_path}")
    
    # Дополнительный график - корреляционная матрица
    correlation_plot(df, output_dir)
    
    # Сводный график статистики
    summary_plot(df, output_dir)

def correlation_plot(df, output_dir):
    """Создает корреляционную матрицу компонентов."""
    numeric_cols = ['volume_mcm', 'delta_volume_mcm', 'precipitation_volume_mcm', 
                   'evaporation_volume_mcm', 'residual_mcm']
    
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        print("   ⚠️ Недостаточно данных для корреляционного анализа")
        return
    
    correlation_matrix = df[available_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Добавляем значения корреляции
    for i in range(len(available_cols)):
        for j in range(len(available_cols)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xticks(range(len(available_cols)))
    ax.set_yticks(range(len(available_cols)))
    ax.set_xticklabels(available_cols, rotation=45, ha='right')
    ax.set_yticklabels(available_cols)
    ax.set_title('Корреляционная матрица компонентов водного баланса', 
                fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Коэффициент корреляции')
    plt.tight_layout()
    
    correlation_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   🔗 Корреляционная матрица: {correlation_path}")

def summary_plot(df, output_dir):
    """Создает сводную статистику."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Распределение изменений объёма
    ax1.hist(df['delta_volume_mcm'].dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(df['delta_volume_mcm'].mean(), color='red', linestyle='--', 
               label=f'Среднее: {df["delta_volume_mcm"].mean():.2f}')
    ax1.set_xlabel('ΔS (млн.м³/день)')
    ax1.set_ylabel('Частота')
    ax1.set_title('Распределение изменений объёма')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Сезонность осадков
    if 'precipitation_mm' in df.columns:
        df['month'] = df['date'].dt.month
        monthly_precip = df.groupby('month')['precipitation_mm'].mean()
        ax2.bar(monthly_precip.index, monthly_precip.values, alpha=0.7, color='blue')
        ax2.set_xlabel('Месяц')
        ax2.set_ylabel('Осадки (мм/день)')
        ax2.set_title('Сезонность осадков')
        ax2.set_xticks(range(1, 13))
        ax2.grid(True, alpha=0.3)
    
    # 3. Баланс P vs ET
    if all(col in df.columns for col in ['precipitation_volume_mcm', 'evaporation_volume_mcm']):
        ax3.scatter(df['precipitation_volume_mcm'], df['evaporation_volume_mcm'], 
                   alpha=0.6, color='green')
        max_val = max(df['precipitation_volume_mcm'].max(), df['evaporation_volume_mcm'].max())
        ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label='P = ET')
        ax3.set_xlabel('Осадки (млн.м³/день)')
        ax3.set_ylabel('Испарение (млн.м³/день)')
        ax3.set_title('Осадки vs Испарение')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Кумулятивный баланс
    if 'residual_mcm' in df.columns:
        cumulative_residual = df['residual_mcm'].cumsum()
        ax4.plot(df['date'], cumulative_residual, 'purple', linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Дата')
        ax4.set_ylabel('Кумулятивный остаток (млн.м³)')
        ax4.set_title('Кумулятивный водный баланс')
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, 'water_balance_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📋 Сводная статистика: {summary_path}")

# --- ГЛАВНАЯ ФУНКЦИЯ ---

def main():
    """Главная функция модели водного баланса."""
    print("🚀 Запуск модели водного баланса...\n")
    
    try:
        # Модуль 1: Обработка Sentinel-2
        sentinel_df = process_sentinel_timeseries()
        
        # Модуль 2: Загрузка внешних данных
        gleam_df, imerg_df, curve_df = load_external_data()
        
        # Модуль 3: Конвертация площадь → объём
        area_to_volume_func = area_to_volume_converter(curve_df)
        sentinel_df = convert_areas_to_volumes(sentinel_df, area_to_volume_func)
        
        # Модуль 4: Синхронизация и расчёт
        unified_df = create_unified_timeseries(sentinel_df, gleam_df, imerg_df)
        unified_df = interpolate_missing_values(unified_df)
        balance_df = calculate_water_balance(unified_df)
        
        # Модуль 5: Сохранение и визуализация
        output_path = save_results(balance_df, OUTPUT_DIR)
        create_visualizations(balance_df, OUTPUT_DIR)
        
        print(f"\n🎉 МОДЕЛЬ УСПЕШНО ЗАВЕРШЕНА!")
        print(f"📁 Результаты сохранены в: {OUTPUT_DIR}")
        print(f"📊 Основной файл: {output_path}")
        print(f"📈 Созданы графики и статистика")
        
        # Краткая сводка
        print(f"\n📋 КРАТКАЯ СВОДКА:")
        print(f"   📅 Период: {balance_df['date'].min().strftime('%Y-%m-%d')} — {balance_df['date'].max().strftime('%Y-%m-%d')}")
        print(f"   📊 Дней анализа: {len(balance_df)}")
        print(f"   🌊 Средний объём: {balance_df['volume_mcm'].mean():.2f} млн.м³")
        print(f"   ⚖️ Средний остаток: {balance_df['residual_mcm'].mean():.3f} млн.м³/день")
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Проверяем зависимости
    required_packages = ['pandas', 'numpy', 'matplotlib', 'scipy', 'rasterio']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Недостающие пакеты: {missing_packages}")
        print("💡 Установите их командой:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        main()
