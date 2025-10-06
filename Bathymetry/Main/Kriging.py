#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автономный скрипт построения ландшафтной карты высот по алгоритму Кригинга.

1. Чтение точек (X, Y, Z) из одного CSV (батиметрия со знаком "-" и/или береговые Z=0).
2. Удаление дублирующихся (X, Y) — усреднение Z.
3. Интерполяция Кригинга по всему набору точек с pseudo-inverse.
4. Сохранение результата в GeoTIFF.
"""

import warnings

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

# === НАСТРОЙКИ ===
CSV_PATH   = 'all_points_cleaned.csv'      # единый CSV с колонками X, Y, Z
OUT_TIF     = 'bathymetry.tif'     # имя выходного GeoTIFF
RES         = 100.0                 # разрешение сетки
OUTPUT_CRS  = CRS.from_epsg(4326)  # здесь укажите нужный EPSG



def load_points(csv_path: str) -> pd.DataFrame:
    """
    Загружает CSV с профилями: столбцы X, Y, Z обязательны.
    Преобразует Z -> отрицательное значение (глубина).
    """
    df = pd.read_csv(csv_path)
    if not {'X', 'Y', 'Z'}.issubset(df.columns):
        raise ValueError("CSV должен содержать столбцы X, Y, Z")
    df['Z'] = -df['Z'].abs()
    return df[['X', 'Y', 'Z']]


def remove_duplicate_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Если в df есть полностью совпадающие X,Y — усредняем Z,
    чтобы матрица вариограмм не вырождалась.
    """
    before = len(df)
    df = df.groupby(['X', 'Y'], as_index=False)['Z'].mean()
    after = len(df)
    if before != after:
        print(f"Удалили дубликаты точек: {before - after}")
    return df


def interpolate_to_grid(df_all: pd.DataFrame, res: float):
    df_clean = remove_duplicate_points(df_all)

    x = df_clean['X'].values
    y = df_clean['Y'].values
    z = df_clean['Z'].values

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xi = np.arange(xmin, xmax + res, res)
    yi = np.arange(ymin, ymax + res, res)
    print(f"Grid size: {len(xi)} x {len(yi)} = {len(xi) * len(yi)} cells")

    ok = OrdinaryKriging(
        x, y, z,
        variogram_model='linear',
        verbose=False,
        enable_plotting=False
        # pseudo_inv по умолчанию False
    )
    zgrid, ss = ok.execute('grid', xi, yi)
    return xi, yi, zgrid



def save_geotiff(xi, yi, zgrid, crs, out_path, res: float):
    """
    Сохраняет zgrid как GeoTIFF с указанным CRS и разрешением.
    """
    transform = from_origin(xi[0], yi[-1], res, res)
    with rasterio.open(
        out_path,
        'w',
        driver='GTiff',
        height=zgrid.shape[0],
        width=zgrid.shape[1],
        count=1,
        dtype=zgrid.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(zgrid, 1)
    print(f"GeoTIFF сохранён: {out_path}")


def main():
    warnings.filterwarnings("ignore")

    print("Загрузка точек из CSV...")
    df_all = load_points(CSV_PATH)
    print(f"Всего точек для интерполяции: {len(df_all)}")

    print("Интерполяция Кригинга...")
    xi, yi, zgrid = interpolate_to_grid(df_all, RES)

    print("Сохранение результата в GeoTIFF...")
    save_geotiff(xi, yi, zgrid, OUTPUT_CRS, OUT_TIF, RES)


if __name__ == '__main__':
    main()
