import os
import glob
import re
import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage.measure import find_contours

def find_sentinel_files(root_dir):
    """
    Находит тройки файлов Sentinel-2 (B03, B08 и SCL) для каждой даты.
    """
    print(f"Поиск файлов в директории: {root_dir}")
    # Ищем все релевантные файлы TIFF
    files = glob.glob(os.path.join(root_dir, '**', '*_Sentinel-2_L2A_*.tiff'), recursive=True)
    
    file_groups = {}
    # Паттерн для извлечения даты из имени файла
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
            
    # Оставляем только полные наборы (B03, B08, SCL)
    valid_groups = {date: paths for date, paths in file_groups.items() if 'b03' in paths and 'b08' in paths and 'scl' in paths}
    
    # Сортируем по дате
    sorted_dates = sorted(valid_groups.keys())
    
    print(f"Найдено {len(sorted_dates)} полных наборов снимков (B03, B08, SCL).")
    return [(date, valid_groups[date]['b03'], valid_groups[date]['b08'], valid_groups[date]['scl']) for date in sorted_dates]

def create_cloud_mask(scl_path, target_shape):
    """
    Создает маску облаков из файла SCL.
    Значения SCL для маскирования: 3 (тень), 8 (облака ср. вер.), 9 (облака выс. вер.), 10 (тонкие перистые), 11 (снег).
    """
    with rasterio.open(scl_path) as scl_src:
        # Пересэмплируем SCL до разрешения целевого изображения
        scl_data = scl_src.read(
            out_shape=(scl_src.count, target_shape[0], target_shape[1]),
            resampling=Resampling.nearest
        )[0]

        # Значения, которые нужно замаскировать
        cloud_values = [3, 8, 9, 10, 11] 
        mask = np.isin(scl_data, cloud_values)
        return mask

def calculate_ndwi(b03_path, b08_path, scl_path):
    """
    Рассчитывает NDWI с применением маски облаков.
    """
    with rasterio.open(b03_path) as green_src:
        green = green_src.read(1).astype('float32')
        profile = green_src.profile
        target_shape = green_src.shape

    with rasterio.open(b08_path) as nir_src:
        nir = nir_src.read(1).astype('float32')

    # Создаем маску облаков с тем же разрешением, что и у каналов
    cloud_mask = create_cloud_mask(scl_path, target_shape)
    
    # Применяем маску: заменяем облачные пиксели на NaN
    green[cloud_mask] = np.nan
    nir[cloud_mask] = np.nan

    # Рассчитываем NDWI, игнорируя ошибки деления на ноль (NaN / NaN)
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = (green - nir) / (green + nir)
    
    return ndwi, profile

def create_ndwi_visualization(ndwi, date_str, output_path, initial_contour=None):
    """
    Создает и сохраняет визуализацию NDWI с водяной маской и начальным контуром.
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150) # Увеличим DPI для качества
    
    # Показываем NDWI с серой картой цветов, NaN будут прозрачными
    ax.imshow(ndwi, cmap='gray', vmin=-1, vmax=1)
    
    # Накладываем синюю полупрозрачную маску на воду (NDWI > 0.1)
    water_mask = np.nan_to_num(ndwi) > 0.1
    overlay = np.zeros((*water_mask.shape, 4), dtype=np.float32)
    overlay[water_mask] = [0.1, 0.4, 0.8, 0.5]  # Синий цвет с 50% прозрачностью
    ax.imshow(overlay)
    
    # Накладываем контур первого года, если он предоставлен
    if initial_contour:
        for contour in initial_contour:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='r', linestyle='--')

    ax.set_title(f"NDWI - {date_str}", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Сохранена визуализация для {date_str}")

def create_animation(image_files, output_gif):
    """
    Создает анимированный GIF из набора изображений.
    """
    print(f"\nСоздание анимации: {output_gif}")
    images = [imageio.imread(filename) for filename in image_files]
    imageio.mimsave(output_gif, images, duration=0.5) # 0.5 секунды на кадр
    print("Анимация успешно создана!")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.dirname(script_dir)
    
    # Новая папка для результатов
    output_dir = os.path.join(script_dir, 'ndwi_output_masked')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Найти все наборы файлов (B03, B08, SCL)
    file_groups = find_sentinel_files(data_root)
    if not file_groups:
        print("Не найдено полных наборов снимков (B03, B08, SCL). Проверьте структуру папок.")
        return

    # 2. Получаем контур из первого снимка в серии
    print("\nШаг 1: Создание контура по первому снимку...")
    first_date, first_b03, first_b08, first_scl = file_groups[0]
    initial_ndwi, _ = calculate_ndwi(first_b03, first_b08, first_scl)
    # Для поиска контура используем маску воды
    initial_water_mask = np.nan_to_num(initial_ndwi) > 0.1
    initial_contours = find_contours(initial_water_mask, 0.5) # 0.5 - порог для контура
    print(f"Контур для даты {first_date} создан.")

    # 3. Обрабатываем все снимки, добавляя на них начальный контур
    print("\nШаг 2: Генерация кадров для анимации...")
    png_files = []
    for i, (date, b03, b08, scl) in enumerate(file_groups):
        ndwi, _ = calculate_ndwi(b03, b08, scl)
        png_path = os.path.join(output_dir, f'ndwi_{i:03d}_{date}.png')
        # Передаем контур для отрисовки на каждом кадре
        create_ndwi_visualization(ndwi, date, png_path, initial_contour=initial_contours)
        png_files.append(png_path)
        
    # 4. Создаем GIF
    if png_files:
        print("\nШаг 3: Сборка GIF-анимации...")
        output_gif_path = os.path.join(output_dir, 'ndwi_animation_masked_with_contour.gif')
        create_animation(png_files, output_gif_path)
        
        # 5. (Опционально) Очищаем временные PNG
        print("\nШаг 4: Очистка временных файлов...")
        for f in png_files:
            os.remove(f)
        print("Временные PNG файлы удалены.")
    else:
        print("Не было создано ни одного изображения для анимации.")

if __name__ == '__main__':
    # Проверка наличия необходимых библиотек
    try:
        import rasterio, numpy, matplotlib, imageio, skimage
    except ImportError:
        print("Необходимые библиотеки не найдены.")
        print("Пожалуйста, установите их, выполнив команду:")
        print("pip install rasterio numpy matplotlib imageio scikit-image")
    else:
        main()
