import xarray as xr
import os
from shapely.geometry import Polygon
import geopandas as gpd
import numpy as np
from datetime import datetime

# Определение полигона
polygon_coords = [
    [63.075256, 52.918011],
    [62.75116,  52.901862],
    [62.700348, 52.830151],
    [62.819138, 52.747516],
    [63.080063, 52.880734],
    [63.075256, 52.918011]
]

def check_gleam_data():
    # Создаем полигон
    polygon = Polygon(polygon_coords)
    gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon])
    
    # Получаем список всех .nc файлов в текущей директории
    nc_files = [f for f in os.listdir('.') if f.endswith('.nc')]
    
    if not nc_files:
        print("Не найдено .nc файлов в текущей директории")
        return
    
    print(f"Найдено {len(nc_files)} .nc файлов")
    
    for nc_file in nc_files:
        try:
            # Открываем файл
            ds = xr.open_dataset(nc_file)
            
            # Выводим основную информацию о файле
            print(f"\nАнализ файла: {nc_file}")
            print(f"Переменные в файле: {list(ds.data_vars)}")
            print(f"Размерности: {ds.dims}")
            
            # Проверяем наличие данных в области полигона
            lat = ds['lat'].values
            lon = ds['lon'].values
            
            # Определяем индексы для области полигона
            lat_mask = (lat >= min(p[1] for p in polygon_coords)) & (lat <= max(p[1] for p in polygon_coords))
            lon_mask = (lon >= min(p[0] for p in polygon_coords)) & (lon <= max(p[0] for p in polygon_coords))
            
            if np.any(lat_mask) and np.any(lon_mask):
                print("Файл содержит данные в области интереса")
                
                # Выводим статистику по основным переменным
                for var in ds.data_vars:
                    if 'lat' in ds[var].dims and 'lon' in ds[var].dims:
                        data = ds[var].sel(lat=lat[lat_mask], lon=lon[lon_mask])
                        print(f"\nСтатистика для {var}:")
                        print(f"Мин: {data.min().values}")
                        print(f"Макс: {data.max().values}")
                        print(f"Среднее: {data.mean().values}")
            else:
                print("Файл не содержит данных в области интереса!")
            
            ds.close()
            
        except Exception as e:
            print(f"Ошибка при обработке файла {nc_file}: {str(e)}")

if __name__ == "__main__":
    check_gleam_data()
