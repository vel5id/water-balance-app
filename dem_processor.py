import os
from pathlib import Path
import glob
import re
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import mapping, Polygon

# --- КОНФИГУРАЦИЯ ---

# 1. Путь к корневой папке с данными Sentinel (по умолчанию — каталог проекта)
ROOT = Path(__file__).resolve().parent
SENTINEL_ROOT_DIR = str(ROOT)

# 2. Путь к файлу DEM (относительно проекта)
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.dirname(os.path.abspath(__file__)))
DEM_PATH = os.path.join(DATA_ROOT, 'Bathymetry', 'Main', 'bathymetry_hh.tif')

# 3. Путь к Copernicus DEM (если доступен)
COPERNICUS_DEM_PATH = os.path.join(DATA_ROOT, 'Hillshade + DEM', 'output_hh.tif')

# 4. Папка для сохранения результатов (кривая и графики)
OUTPUT_DIR = os.path.join(DATA_ROOT, 'processing_output')

# 5. Количество шагов для расчета кривой (чем больше, тем точнее, но дольше)
ELEVATION_STEPS = 200

# Порог мелководья (метры), ниже которого подменяем значениями Copernicus
SHALLOW_THRESHOLD_M = 2.0

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (из ndwi.py) ---

def find_sentinel_files(root_dir):
    """Находит тройки файлов Sentinel-2 (B03, B08 и SCL)."""
    files = glob.glob(os.path.join(root_dir, '**', '*_Sentinel-2_L2A_*.tiff'), recursive=True)
    file_groups = {}
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')
    for f in files:
        match = date_pattern.search(os.path.basename(f))
        if not match: continue
        date_str = match.group(1)
        if date_str not in file_groups: file_groups[date_str] = {}
        if '_B03_' in f: file_groups[date_str]['b03'] = f
        elif '_B08_' in f: file_groups[date_str]['b08'] = f
        elif 'Scene_classification_map' in f: file_groups[date_str]['scl'] = f
    valid_groups = {d: p for d, p in file_groups.items() if 'b03' in p and 'b08' in p and 'scl' in p}
    sorted_dates = sorted(valid_groups.keys())
    return [(d, valid_groups[d]['b03'], valid_groups[d]['b08'], valid_groups[d]['scl']) for d in sorted_dates]

def calculate_test_water_mask(b03_path, b08_path, scl_path, threshold):
    """Тестирует маску воды с заданным порогом NDWI."""
    with rasterio.open(b03_path) as green_src:
        green = green_src.read(1).astype('float32')
        target_shape = green_src.shape

    with rasterio.open(b08_path) as nir_src:
        nir = nir_src.read(1).astype('float32')

    with rasterio.open(scl_path) as scl_src:
        scl_data = scl_src.read(1, out_shape=target_shape, resampling=Resampling.nearest)
    
    cloud_mask = np.isin(scl_data, [3, 8, 9, 10, 11])
    green[cloud_mask] = np.nan
    nir[cloud_mask] = np.nan

    np.seterr(divide='ignore', invalid='ignore')
    ndwi = (green - nir) / (green + nir)
    
    water_mask = np.nan_to_num(ndwi) > threshold
    return water_mask

def calculate_ndwi_and_mask(b03_path, b08_path, scl_path):
    """Рассчитывает NDWI и возвращает бинарную маску воды."""
    with rasterio.open(b03_path) as green_src:
        green = green_src.read(1).astype('float32')
        profile = green_src.profile
        transform = green_src.transform
        crs = green_src.crs
        target_shape = green_src.shape

    with rasterio.open(b08_path) as nir_src:
        nir = nir_src.read(1).astype('float32')

    with rasterio.open(scl_path) as scl_src:
        scl_data = scl_src.read(1, out_shape=target_shape, resampling=Resampling.nearest)
    
    cloud_mask = np.isin(scl_data, [3, 8, 9, 10, 11])
    green[cloud_mask] = np.nan
    nir[cloud_mask] = np.nan

    np.seterr(divide='ignore', invalid='ignore')
    ndwi = (green - nir) / (green + nir)
    
    water_mask = np.nan_to_num(ndwi) > 0.275  # Понижен порог до 0.275
    return water_mask, crs, transform

# --- ОСНОВНОЙ СКРИПТ ---

def get_max_water_mask(sentinel_dir):
    """Находит маску максимального затопления по всем снимкам."""
    print("Шаг 1: Поиск максимальной площади затопления по снимкам Sentinel...")
    file_groups = find_sentinel_files(sentinel_dir)
    if not file_groups:
        raise FileNotFoundError("Не найдено ни одного набора снимков Sentinel.")

    max_area = 0
    max_water_mask = None
    max_mask_crs = None
    max_mask_transform = None
    
    print("ДИАГНОСТИКА МАСКИ ВОДЫ:")
    print("Тестируем разные пороги NDWI...")

    for i, (date, b03, b08, scl) in enumerate(file_groups):
        print(f"  - Анализ снимка от {date} ({i+1}/{len(file_groups)})")
        water_mask, crs, transform = calculate_ndwi_and_mask(b03, b08, scl)
        current_area = np.sum(water_mask)
        
        # Диагностика только для первого снимка
        if i == 0:
            print(f"    Диагностика NDWI для {date}:")
            # Тестируем разные пороги
            for threshold in [0.0, 0.1, 0.2, 0.25, 0.275, 0.3, 0.35, 0.4]:
                test_mask = calculate_test_water_mask(b03, b08, scl, threshold)
                test_area = np.sum(test_mask)
                print(f"      Порог {threshold}: {test_area} пикселей воды")
        
        if current_area > max_area:
            max_area = current_area
            max_water_mask = water_mask
            max_mask_crs = crs
            max_mask_transform = transform
            print(f"    > Новая максимальная площадь найдена: {max_area} пикселей")
    
    print(f"\nИтоговая максимальная маска: {max_area} пикселей")
    
    # Сохраняем максимальную маску NDWI в GeoTIFF
    mask_output_path = os.path.join(OUTPUT_DIR, 'ndwi_mask_0275.tif')
    save_geotiff(
        data=max_water_mask.astype(np.uint8),
        transform=max_mask_transform,
        crs=max_mask_crs,
        output_path=mask_output_path,
        dtype='uint8',
        nodata=0
    )
    print(f"Маска NDWI (порог 0.275) сохранена: {mask_output_path}")
    
    return max_water_mask, max_mask_crs, max_mask_transform

def download_copernicus_dem_instructions():
    """Выводит инструкции по скачиванию Copernicus DEM."""
    print("\n=== ИНСТРУКЦИЯ ПО СКАЧИВАНИЮ COPERNICUS DEM ===")
    print("1. Перейдите на: https://spacedata.copernicus.eu/")
    print("2. Зарегистрируйтесь/войдите в аккаунт")
    print("3. Найдите 'Copernicus Digital Elevation Model'")
    print("4. Выберите разрешение 30м (GLO-30) или 90м (GLO-90)")
    print("5. Скачайте тайл для вашего региона")
    print("6. Сохраните как 'copernicus_dem.tif' в папку с данными")
    print("7. Или используйте Python библиотеки:")
    print("   pip install elevation")
    print("   import elevation")
    print("   elevation.clip(bounds=(lon_min, lat_min, lon_max, lat_max), output='copernicus_dem.tif')")
    print("================================================\n")

def integrate_copernicus_dem(bathymetry_data, mask_transform, mask_crs, water_mask, copernicus_path, shallow_threshold_m: float = 2.0):
    """
    Интегрирует Copernicus DEM с батиметрическими данными для заполнения пробелов.
    Возвращает (integrated_dem, copernicus_reprojected) для диагностики.
    """
    print("\n--- ИНТЕГРАЦИЯ COPERNICUS DEM ---")
    
    if not os.path.exists(copernicus_path):
        print(f"Copernicus DEM не найден: {copernicus_path}")
        download_copernicus_dem_instructions()
        return bathymetry_data, None
    
    try:
        with rasterio.open(copernicus_path) as cop_src:
            print(f"Загружен Copernicus DEM: {cop_src.shape}, CRS: {cop_src.crs}")
            
            # Перепроецируем Copernicus DEM в систему координат маски
            cop_reprojected = np.full(bathymetry_data.shape, np.nan, dtype=np.float32)
            
            reproject(
                source=cop_src.read(1),
                destination=cop_reprojected,
                src_transform=cop_src.transform,
                src_crs=cop_src.crs,
                dst_transform=mask_transform,
                dst_crs=mask_crs,
                resampling=Resampling.bilinear,
                src_nodata=cop_src.nodata,
                dst_nodata=np.nan
            )
            
            print(f"Copernicus DEM перепроецирован: валидных пикселей {np.sum(~np.isnan(cop_reprojected))}")
            
            # Создаем интегрированный DEM
            integrated_dem = bathymetry_data.copy()

            # Оценка уровня воды (береговой отметки) по Copernicus: медиана высоты на границе воды
            h, w = water_mask.shape
            # Граница вне воды (соседние к воде пиксели)
            nb = np.zeros_like(water_mask, dtype=bool)
            nb[1:, :] |= water_mask[:-1, :]
            nb[:-1, :] |= water_mask[1:, :]
            nb[:, 1:] |= water_mask[:, :-1]
            nb[:, :-1] |= water_mask[:, 1:]
            boundary_outside = (~water_mask) & nb
            shoreline_vals = cop_reprojected[boundary_outside]
            shoreline_vals = shoreline_vals[np.isfinite(shoreline_vals)]
            if shoreline_vals.size >= 100:
                shoreline_elev = float(np.nanmedian(shoreline_vals))
            else:
                # Фолбэк: берем высокий перцентиль рельефа вне воды внутри bbox
                ys, xs = np.where(water_mask)
                if ys.size > 0:
                    y0, y1 = max(0, ys.min()-5), min(h, ys.max()+6)
                    x0, x1 = max(0, xs.min()-5), min(w, xs.max()+6)
                    roi = cop_reprojected[y0:y1, x0:x1]
                    roi = roi[np.isfinite(roi) & (~water_mask[y0:y1, x0:x1])]
                    shoreline_elev = float(np.nanpercentile(roi, 95)) if roi.size > 0 else float(np.nanmedian(cop_reprojected[np.isfinite(cop_reprojected)]))
                else:
                    shoreline_elev = float(np.nanmedian(cop_reprojected[np.isfinite(cop_reprojected)]))
            print(f"Оцененная отметка уровня воды (из Copernicus): {shoreline_elev:.2f} м")

            # Натуральная максимальная глубина из исходной батиметрии (по воде)
            bathy_vals = integrated_dem[water_mask]
            bathy_vals = bathy_vals[np.isfinite(bathy_vals)]
            if bathy_vals.size > 0:
                # Если глубины положительные — это глубина; если отрицательные — берем модуль
                native_max_depth = float(max(np.nanmax(bathy_vals), abs(np.nanmin(bathy_vals))))
            else:
                native_max_depth = float(20.0)  # разумный дефолт
            depth_cap = native_max_depth + 2.0
            print(f"Ограничение глубины Copernicus: <= {depth_cap:.2f} м (нативный максимум ~ {native_max_depth:.2f} м)")

            # Относительная глубина из Copernicus: depth = max(0, shoreline_elev - terrain_elev)
            cop_depth_full = shoreline_elev - cop_reprojected
            cop_depth_full = np.where(np.isfinite(cop_depth_full), np.maximum(0.0, cop_depth_full), np.nan)
            cop_depth_full = np.minimum(cop_depth_full, depth_cap)
            
            # Находим области где батиметрия отсутствует, но есть водная маска
            bathymetry_missing = np.isnan(bathymetry_data)
            copernicus_available = ~np.isnan(cop_reprojected)
            water_areas = water_mask
            
            # Области для заполнения: нет батиметрии + есть Copernicus + есть вода
            fill_mask = bathymetry_missing & copernicus_available & water_areas
            
            if np.sum(fill_mask) > 0:
                print(f"Заполняем {np.sum(fill_mask)} пикселей из Copernicus DEM")
                
                # Используем относительную глубину из Copernicus и делаем её отрицательной
                fill_depth = cop_depth_full[fill_mask]
                integrated_dem[fill_mask] = -fill_depth
                print(f"Заполненные глубины (Copernicus, относительные): min={np.nanmin(fill_depth):.2f} м, max={np.nanmax(fill_depth):.2f} м")
            else:
                print("Нет областей для заполнения из Copernicus DEM")

            # Дополнительно: заменяем мелководье (< shallow_threshold_m) значениями Copernicus там где доступно
            # Определяем мелководье по абсолютному значению глубины относительно 0
            shallow_mask = water_areas & copernicus_available
            # Если батиметрия уже отрицательная (глубина), используем |depth|; если положительная, тоже берем |value|
            abs_bathy = np.abs(integrated_dem)
            shallow_mask &= np.isfinite(abs_bathy) & (abs_bathy < float(shallow_threshold_m))
            if np.sum(shallow_mask) > 0:
                print(f"Заменяем мелководье (<{shallow_threshold_m} м) по Copernicus (относительной глубиной): {np.sum(shallow_mask)} пикселей")
                integrated_dem[shallow_mask] = -cop_depth_full[shallow_mask]

            # Нормализуем знак: внутри водной маски глубины должны быть отрицательными
            inside = water_areas & np.isfinite(integrated_dem)
            sign_fixed = np.sum(inside & (integrated_dem > 0))
            if sign_fixed > 0:
                print(f"Исправление знака глубин внутри воды (-> отрицательные): {sign_fixed} пикселей")
                integrated_dem[inside] = -np.abs(integrated_dem[inside])
            
            # Сохраняем интегрированный DEM
            integrated_output_path = os.path.join(OUTPUT_DIR, 'integrated_bathymetry_copernicus.tif')
            save_geotiff(
                data=integrated_dem,
                transform=mask_transform,
                crs=mask_crs,
                output_path=integrated_output_path,
                dtype='float32',
                nodata=np.nan
            )
            print(f"Интегрированная батиметрия сохранена: {integrated_output_path}")
            
            return integrated_dem, cop_reprojected
            
    except Exception as e:
        print(f"Ошибка при обработке Copernicus DEM: {e}")
        return bathymetry_data, None

def save_geotiff(data, transform, crs, output_path, dtype='float32', nodata=None):
    """Сохраняет массив в GeoTIFF файл."""
    profile = {
        'driver': 'GTiff',
        'height': data.shape[0],
        'width': data.shape[1],
        'count': 1,
        'dtype': dtype,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    if nodata is not None:
        profile['nodata'] = nodata
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(dtype), 1)
    
    print(f"GeoTIFF сохранен: {output_path}")

def save_diagnostic_image(dem_data, mask_reprojected, output_dir):
    """Создает диагностическое изображение для анализа пересечения DEM и маски."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # DEM
    im1 = ax1.imshow(dem_data, cmap='terrain', aspect='auto')
    ax1.set_title('DEM (Bathymetry)')
    plt.colorbar(im1, ax=ax1, label='Высота, м')
    
    # Маска
    ax2.imshow(mask_reprojected, cmap='Blues', aspect='auto')
    ax2.set_title('Маска воды (перепроецированная)')
    
    # Наложение
    ax3.imshow(dem_data, cmap='terrain', aspect='auto', alpha=0.7)
    ax3.contour(mask_reprojected, levels=[0.5], colors='red', linewidths=2)
    ax3.set_title('DEM + контур маски')
    
    diag_path = os.path.join(output_dir, 'diagnostic_overlay.png')
    plt.tight_layout()
    plt.savefig(diag_path, dpi=150)
    plt.close()
    print(f"Диагностическое изображение сохранено: {diag_path}")

def create_comprehensive_diagnostic_plot(water_mask, bathymetry_dem, copernicus_dem, integrated_dem, output_dir):
    """Создает подробную диагностическую визуализацию всех компонентов."""
    import matplotlib.pyplot as plt
    
    print("\nСоздание комплексной диагностической визуализации...")
    
    # Создаем фигуру с 6 подграфиками (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Маска воды
    im1 = axes[0,0].imshow(water_mask, cmap='Blues', aspect='auto')
    axes[0,0].set_title(f'Маска воды NDWI\n({np.sum(water_mask)} пикселей)')
    plt.colorbar(im1, ax=axes[0,0], label='Вода (1) / Суша (0)')
    
    # 2. Исходная батиметрия
    valid_bathy = ~np.isnan(bathymetry_dem)
    im2 = axes[0,1].imshow(bathymetry_dem, cmap='terrain_r', aspect='auto', 
                          vmin=np.nanmin(bathymetry_dem), vmax=np.nanmax(bathymetry_dem))
    axes[0,1].set_title(f'Исходная батиметрия\n({np.sum(valid_bathy)} валидных пикселей)')
    plt.colorbar(im2, ax=axes[0,1], label='Глубина, м')
    
    # 3. Copernicus DEM (если доступен)
    if copernicus_dem is not None:
        valid_cop = ~np.isnan(copernicus_dem)
        im3 = axes[0,2].imshow(copernicus_dem, cmap='terrain', aspect='auto',
                              vmin=np.nanmin(copernicus_dem), vmax=np.nanmax(copernicus_dem))
        axes[0,2].set_title(f'Copernicus DEM\n({np.sum(valid_cop)} валидных пикселей)')
        plt.colorbar(im3, ax=axes[0,2], label='Высота, м')
    else:
        axes[0,2].text(0.5, 0.5, 'Copernicus DEM\nне доступен', 
                      ha='center', va='center', transform=axes[0,2].transAxes, fontsize=14)
        axes[0,2].set_title('Copernicus DEM (отсутствует)')
    
    # 4. Интегрированный DEM
    valid_integrated = ~np.isnan(integrated_dem)
    im4 = axes[1,0].imshow(integrated_dem, cmap='terrain_r', aspect='auto',
                          vmin=np.nanmin(integrated_dem), vmax=np.nanmax(integrated_dem))
    axes[1,0].set_title(f'Интегрированный DEM\n({np.sum(valid_integrated)} валидных пикселей)')
    plt.colorbar(im4, ax=axes[1,0], label='Глубина/Высота, м')
    
    # 5. Покрытие данными
    coverage = np.zeros(water_mask.shape, dtype=np.uint8)
    coverage[water_mask] = 1  # Вода = 1
    coverage[valid_bathy] += 2  # + Батиметрия = 3
    if copernicus_dem is not None:
        coverage[valid_cop] += 4  # + Copernicus = 7
    
    coverage_labels = {0: 'Нет данных', 1: 'Только вода', 2: 'Только батиметрия', 
                      3: 'Вода + батиметрия', 4: 'Только Copernicus', 
                      5: 'Вода + Copernicus', 6: 'Батиметрия + Copernicus', 
                      7: 'Все данные'}
    
    im5 = axes[1,1].imshow(coverage, cmap='viridis', aspect='auto')
    axes[1,1].set_title('Покрытие данными')
    cbar5 = plt.colorbar(im5, ax=axes[1,1])
    cbar5.set_label('Типы данных')
    
    # 6. Наложение контуров
    axes[1,2].imshow(integrated_dem, cmap='terrain_r', aspect='auto', alpha=0.8)
    axes[1,2].contour(water_mask, levels=[0.5], colors='blue', linewidths=2, alpha=0.8)
    if np.sum(valid_bathy) > 0:
        axes[1,2].contour(valid_bathy, levels=[0.5], colors='red', linewidths=1, alpha=0.6)
    if copernicus_dem is not None and np.sum(valid_cop) > 0:
        axes[1,2].contour(valid_cop, levels=[0.5], colors='green', linewidths=1, alpha=0.6)
    
    axes[1,2].set_title('Контуры всех данных')
    axes[1,2].legend(['Граница воды', 'Граница батиметрии', 'Граница Copernicus'], 
                    loc='upper right', fontsize=8)
    
    # Убираем оси для всех подграфиков
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Добавляем общую статистику
    stats_text = f"""СТАТИСТИКА ПОКРЫТИЯ:
Всего пикселей: {water_mask.size:,}
Водная маска: {np.sum(water_mask):,} пикселей ({100*np.sum(water_mask)/water_mask.size:.1f}%)
Батиметрия: {np.sum(valid_bathy):,} пикселей ({100*np.sum(valid_bathy)/water_mask.size:.1f}%)"""
    
    if copernicus_dem is not None:
        stats_text += f"\nCopernicus: {np.sum(valid_cop):,} пикселей ({100*np.sum(valid_cop)/water_mask.size:.1f}%)"
    
    stats_text += f"\nИнтегрированный: {np.sum(valid_integrated):,} пикселей ({100*np.sum(valid_integrated)/water_mask.size:.1f}%)"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Сохраняем диагностику
    diagnostic_path = os.path.join(output_dir, 'comprehensive_diagnostic.png')
    plt.savefig(diagnostic_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Комплексная диагностическая визуализация сохранена: {diagnostic_path}")

def create_area_volume_curve(dem_path, water_mask, mask_crs, mask_transform, output_dir):
    """Создает кривую 'Площадь-Объем' на основе DEM и маски."""
    print("\nШаг 2: Построение кривой 'Площадь-Объем'...")
    
    with rasterio.open(dem_path) as dem_src:
        # --- ДИАГНОСТИКА ---
        print("\n--- ДИАГНОСТИКА ---")
        print(f"Маска Sentinel CRS: {mask_crs}")
        print(f"DEM CRS: {dem_src.crs}")
        
        dem_bounds = dem_src.bounds
        print(f"DEM Bounds (L,B,R,T): {dem_bounds.left:.2f}, {dem_bounds.bottom:.2f}, {dem_bounds.right:.2f}, {dem_bounds.top:.2f}")
        print(f"DEM NoData: {dem_src.nodata}")
        
        # Читаем весь DEM для диагностики
        dem_full = dem_src.read(1)
        print(f"DEM размер: {dem_full.shape}")
        print(f"DEM тип данных: {dem_full.dtype}")
        
        # Обработка NoData значений
        if dem_src.nodata is not None:
            dem_full_clean = dem_full.copy()
            dem_full_clean[dem_full == dem_src.nodata] = np.nan
        else:
            dem_full_clean = dem_full.copy()
        
        valid_pixels = ~np.isnan(dem_full_clean)
        print(f"Валидных пикселей в DEM: {np.sum(valid_pixels)} из {dem_full.size}")
        
        if np.sum(valid_pixels) > 0:
            print(f"DEM статистика (min, max, mean): {np.nanmin(dem_full_clean):.2f}, {np.nanmax(dem_full_clean):.2f}, {np.nanmean(dem_full_clean):.2f}")
        else:
            print("DEM содержит только NoData значения!")
        print("---------------------\n")
        # --- КОНЕЦ ДИАГНОСТИКИ ---

        # НОВЫЙ ПОДХОД: Преобразуем DEM в систему координат Sentinel
        print("Преобразование DEM в систему координат Sentinel (EPSG:4326)...")
        
        # Получаем границы области Sentinel в географических координатах
        sentinel_bounds = rasterio.transform.array_bounds(water_mask.shape[0], water_mask.shape[1], mask_transform)
        print(f"Границы Sentinel (L,B,R,T): {sentinel_bounds}")
        
        # Преобразуем границы DEM в географические координаты для сравнения
        dem_bounds_geo = rasterio.warp.transform_bounds(dem_src.crs, mask_crs, *dem_src.bounds)
        print(f"Границы DEM в гео-координатах (L,B,R,T): {dem_bounds_geo}")
        
        # Используем ту же сетку что и у Sentinel (без перепроекции сетки)
        print(f"Размер маски Sentinel: {water_mask.shape}")
        print(f"Трансформация Sentinel: {mask_transform}")
        
        # Перепроецируем DEM напрямую в сетку Sentinel
        dem_reprojected = np.full(water_mask.shape, np.nan, dtype=np.float32)
        
        reproject(
            source=dem_src.read(1),
            destination=dem_reprojected,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=mask_transform,
            dst_crs=mask_crs,
            resampling=Resampling.bilinear,
            src_nodata=dem_src.nodata,
            dst_nodata=np.nan
        )
        
        print(f"Перепроецированный DEM: {dem_reprojected.shape}, валидных пикселей: {np.sum(~np.isnan(dem_reprojected))}")
        
        if np.sum(~np.isnan(dem_reprojected)) > 0:
            print(f"Статистика перепроецированного DEM: {np.nanmin(dem_reprojected):.2f} - {np.nanmax(dem_reprojected):.2f}")
        
        # Сохраняем перепроецированную батиметрию в GeoTIFF
        bathymetry_output_path = os.path.join(output_dir, 'bathymetry_reprojected_epsg4326.tif')
        save_geotiff(
            data=dem_reprojected,
            transform=mask_transform,
            crs=mask_crs,
            output_path=bathymetry_output_path,
            dtype='float32',
            nodata=np.nan
        )
        print(f"Перепроецированная батиметрия сохранена: {bathymetry_output_path}")
        
        # НОВАЯ ФУНКЦИЯ: Интеграция с Copernicus DEM
        integrated_dem, copernicus_reprojected = integrate_copernicus_dem(
            dem_reprojected, mask_transform, mask_crs, water_mask, COPERNICUS_DEM_PATH, shallow_threshold_m=SHALLOW_THRESHOLD_M
        )
        
        # СОЗДАЕМ КОМПЛЕКСНУЮ ДИАГНОСТИЧЕСКУЮ ВИЗУАЛИЗАЦИЮ
        create_comprehensive_diagnostic_plot(
            water_mask=water_mask,
            bathymetry_dem=dem_reprojected,
            copernicus_dem=copernicus_reprojected,
            integrated_dem=integrated_dem,
            output_dir=output_dir
        )
        
        # Используем интегрированный DEM вместо обычной батиметрии
        print(f"\nИспользуем интегрированный DEM для расчета кривой...")
        
        # Теперь применяем маску воды
        print(f"Применяем маску воды ({np.sum(water_mask)} пикселей)...")
        dem_clipped = integrated_dem.copy()
        dem_clipped[~water_mask] = np.nan

        valid_dem_pixels = ~np.isnan(dem_clipped)
        print(f"Валидных пикселей в обрезанном интегрированном DEM: {np.sum(valid_dem_pixels)}")

        if np.sum(valid_dem_pixels) == 0:
            print("\n!!! ОШИБКА: Интегрированный DEM и маска не пересекаются !!!")
            
            # Создаем комплексную диагностическую визуализацию
            create_comprehensive_diagnostic_plot(
                water_mask=water_mask,
                bathymetry_dem=dem_reprojected,
                copernicus_dem=copernicus_reprojected,
                integrated_dem=integrated_dem,
                output_dir=output_dir
            )
            return None

        if np.sum(valid_dem_pixels) < 100:
            print(f"\n!!! ПРЕДУПРЕЖДЕНИЕ: Мало пикселей в пересечении ({np.sum(valid_dem_pixels)}). !!!")
            print("Используем все валидные данные интегрированного DEM...")
            
            # Создаем комплексную диагностическую визуализацию
            create_comprehensive_diagnostic_plot(
                water_mask=water_mask,
                bathymetry_dem=dem_reprojected,
                copernicus_dem=copernicus_reprojected,
                integrated_dem=integrated_dem,
                output_dir=output_dir
            )
            
            # Используем все валидные данные DEM
            dem_clipped = integrated_dem.copy()

        # Расчет площади пикселя в м^2 (правильный расчет для EPSG:4326)
        # Получаем размер пикселя в градусах
        pixel_width_deg = abs(mask_transform[0])  # размер пикселя по X в градусах
        pixel_height_deg = abs(mask_transform[4])  # размер пикселя по Y в градусах
        
        # Примерно в центре области для более точного расчета
        center_lat = (sentinel_bounds[1] + sentinel_bounds[3]) / 2
        
        # Преобразуем градусы в метры (приблизительно)
        meters_per_deg_lon = 111320 * np.cos(np.radians(center_lat))
        meters_per_deg_lat = 111320
        
        pixel_area = pixel_width_deg * meters_per_deg_lon * pixel_height_deg * meters_per_deg_lat
        print(f"Размер пикселя: {pixel_width_deg:.6f}° x {pixel_height_deg:.6f}° = {pixel_area:.2f} м²")
        
        min_elev = np.nanmin(dem_clipped)
        max_elev = np.nanmax(dem_clipped)
        print(f"  - Рельеф дна определен. Высоты: от {min_elev:.2f} м до {max_elev:.2f} м.")
        
        if min_elev == max_elev:
            print("!!! ВНИМАНИЕ: Все высоты в обрезанном DEM одинаковы. Расчет объема невозможен. !!!")
            return None

        # Расчет кривой
        elevation_levels = np.linspace(min_elev, max_elev, ELEVATION_STEPS)
        results = []
        for level in elevation_levels:
            submerged_pixels = dem_clipped[dem_clipped <= level]
            
            # Площадь в км^2
            area_m2 = len(submerged_pixels) * pixel_area
            area_km2 = area_m2 / 1_000_000
            
            # Объем в млн. м^3
            volume_m3 = np.sum(level - submerged_pixels) * pixel_area
            volume_mcm = volume_m3 / 1_000_000
            
            results.append({'elevation_m': level, 'area_km2': area_km2, 'volume_mcm': volume_mcm})
    
    df = pd.DataFrame(results)
    
    # Сохранение
    csv_path = os.path.join(output_dir, 'area_volume_curve.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n-> Кривая 'Площадь-Объем' сохранена в: {csv_path}")
    
    # Визуализация
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    df.plot(x='elevation_m', y='area_km2', ax=ax1, grid=True, legend=False)
    ax1.set_ylabel("Площадь, км²")
    ax1.set_title("Гипсографическая кривая (Высота-Площадь)")
    
    df.plot(x='elevation_m', y='volume_mcm', ax=ax2, grid=True, color='r', legend=False)
    ax2.set_xlabel("Высота уровня воды, м")
    ax2.set_ylabel("Объем, млн. м³")
    ax2.set_title("Батиграфическая кривая (Высота-Объем)")
    
    plot_path = os.path.join(output_dir, 'area_volume_curves.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"-> Графики кривых сохранены в: {plot_path}")
    
    return df

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Создана папка для результатов: {OUTPUT_DIR}")
        
    # Шаг 1
    max_water_mask, crs, transform = get_max_water_mask(SENTINEL_ROOT_DIR)
    
    # Шаг 2
    create_area_volume_curve(DEM_PATH, max_water_mask, crs, transform, OUTPUT_DIR)
    
    print("\nГотово!")

if __name__ == '__main__':
    try:
        import pandas, shapely, matplotlib
    except ImportError:
        print("Необходимые библиотеки не найдены.")
        print("Пожалуйста, установите их, выполнив команду:")
        print("pip install pandas shapely matplotlib rasterio")
    else:
        main()
