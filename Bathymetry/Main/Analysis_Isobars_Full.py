#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import Delaunay, KDTree
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter
from matplotlib import pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pyproj import Transformer
from tqdm import tqdm
import rasterio
from rasterio.transform import from_origin

# ---------- SETTINGS (меняйте при необходимости) ----------
# 1) Генерация контура
INPUT_SHP             = "shoreline.shp"            # входной shapefile
NUM_POINTS            = 500                         # число точек на каждый полигон
# 2) Основная обработка
EDGE_CSV              = "edge_points.csv"           # автоматически сгенерируется
INTERIOR_CSV          = "representative_80000_points.csv"
CLEANED_INTERIOR_CSV  = "interior_cleaned.csv"
ALL_POINTS_CSV        = "all_points_cleaned.csv"
OUTPUT_PNG            = "isolines_p80000_k10_mad_25_smth15.png"
THREE_D_PNG           = "3d_grid.png"
# Базовое имя для single-run; при множественных разрешениях будут добавляться суффиксы
OUTPUT_TIF            = "bathymetry_hh.tif"            # Имя выходного GeoTIFF файла (для одиночного запуска)
# 3) Прочие параметры
UTM_EPSG              = "EPSG:32641"
K_NEIGHBORS           = 30
MAD_THRESHOLD         = 3.0
SMOOTH_ITERS          = 3
# Базовый шаг (если не используется список через аргументы командной строки)
GRID_STEP             = 100.0
SIGMA_FACTOR          = 1.5
# ----------------------------------------------------------

def generate_edge_points(shp_path: str, num_points: int) -> pd.DataFrame:
    """
    Читает shapefile, извлекает контуры полигонов и равномерно
    раскладывает по num_points точек на границу каждого.
    Возвращает DataFrame с колонками ['fid','id','Z','Y','X'].
    """
    gdf = gpd.read_file(shp_path)
    records = []

    for fid, geom in enumerate(gdf.geometry):
        boundary = geom.boundary
        if boundary.geom_type == 'MultiLineString':
            segments = list(boundary)
            total_length = sum(seg.length for seg in segments)
            distances = np.linspace(0, total_length, num_points, endpoint=False)
            def point_at(dist):
                acc = 0.0
                for seg in segments:
                    if acc + seg.length >= dist:
                        return seg.interpolate(dist - acc)
                    acc += seg.length
                return segments[-1].interpolate(segments[-1].length)
            points = [point_at(d) for d in distances]
        else:
            length = boundary.length
            distances = np.linspace(0, length, num_points, endpoint=False)
            points = [boundary.interpolate(d) for d in distances]

        for idx, pt in enumerate(points):
            records.append({
                'fid': fid,
                'id' : idx,
                'Z'  : 0,
                'Y'  : pt.y,
                'X'  : pt.x,
            })

    return pd.DataFrame(records, columns=['fid','id','Z','Y','X'])


def load_interior(path: str) -> pd.DataFrame:
    """Загружает csv с точками интерьера."""
    return pd.read_csv(path)


def median_mad_filter(df: pd.DataFrame, k=10, t=3.0) -> pd.DataFrame:
    """MAD-фильтр по соседям в k ближайших точках."""
    coords = df[["X","Y"]].to_numpy()
    z      = df["Z"].to_numpy()
    tree   = KDTree(coords)
    keep   = np.ones(len(df), bool)

    for i in tqdm(range(len(df)), desc="MAD filter"):
        _, idx = tree.query(coords[i], k=k)
        neigh = z[idx]
        med   = np.median(neigh)
        mad   = np.median(np.abs(neigh - med))
        thr   = (2 if mad == 0 else t * mad)
        keep[i] = abs(z[i] - med) <= thr

    return df[keep].reset_index(drop=True)


def invert_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Инвертирует Z := -abs(Z), но сохраняет Z == 0 без изменений."""
    df = df.copy()
    df['Z'] = df['Z'].apply(lambda z: 0 if z == 0 else -abs(z))
    return df


def save_cleaned_points(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} points to '{path}'")


def laplacian_smooth(points: pd.DataFrame,
                     shore_mask: np.ndarray,
                     iters: int = 2) -> np.ndarray:
    coords = points[["X","Y"]].to_numpy()
    z      = points["Z"].to_numpy().copy()
    tri    = Delaunay(coords)

    # строим список соседей
    neigh = [[] for _ in range(len(points))]
    for simplex in tri.simplices:
        for i in range(3):
            a,b = simplex[i], simplex[(i+1)%3]
            neigh[a].append(b); neigh[b].append(a)
    neigh = [np.unique(n).astype(int) for n in neigh]

    for _ in range(iters):
        z_new = z.copy()
        for i in tqdm(range(len(points)), desc="Laplacian smooth", leave=False):
            if shore_mask[i] or len(neigh[i]) == 0:
                continue
            z_new[i] = z[neigh[i]].mean()
        z = z_new

    return z


def grid_and_sanitize(points: pd.DataFrame,
                      shore_poly: Path,
                      step: float = 5.0,
                      sigma_factor: float = 1.5):
    # регулярная сетка
    x_min, x_max = points.X.min(), points.X.max()
    y_min, y_max = points.Y.min(), points.Y.max()
    xi = np.arange(x_min, x_max + step, step)
    yi = np.arange(y_min, y_max + step, step)
    xi, yi = np.meshgrid(xi, yi)

    # линейная интерполяция
    zi = griddata(points[["X","Y"]].to_numpy(),
                  points["Z"].to_numpy(),
                  (xi, yi),
                  method="linear")

    # маска (только внутри полигона)
    inside = shore_poly.contains_points(np.c_[xi.ravel(), yi.ravel()]) \
             .reshape(zi.shape)
    zi[~inside] = np.nan

    # локальный σ → карта
    def win_std(a):
        if np.isnan(a).all(): return 0.0
        return np.nanstd(a)
    sigma_map = generic_filter(zi, win_std, size=3, mode="nearest")
    thresh = sigma_factor * np.nanmedian(sigma_map)

    # «умный» медианный фильтр
    def std_median_filter(vals):
        center = vals[4]
        s = np.nanstd(vals)
        return np.nanmedian(vals) if s > thresh else center

    zi = generic_filter(zi, std_median_filter, size=3, mode="nearest")

    # глобальный клип
    flat = zi[np.isfinite(zi)]
    mu, sd = np.nanmean(flat), np.nanstd(flat)
    zi = np.clip(zi, mu - 3*sd, mu + 3*sd)

    # инверсия глубины
    zi = -zi

    return xi, yi, zi


def plot_and_save_grid(xi, yi, zi, shore_df, out_png, levels=20):
    fig, ax = plt.subplots(figsize=(8,6))
    filled = ax.contourf(xi, yi, zi, levels=levels, cmap='viridis_r')
    ax.scatter(shore_df.X, shore_df.Y, s=6, c="black", label="Shoreline (Z=0)")
    fig.colorbar(filled, ax=ax, label="Z-value")
    ax.set_xlabel("Easting (m, UTM 41 N)")
    ax.set_ylabel("Northing (m, UTM 41 N)")
    ax.set_title("Isolines (TIN smoothed + smart filter)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"Plot saved → {out_png}")
    plt.show()


def plot_3d_grid(xi, yi, zi, out_png=None):
    fig = plt.figure(figsize=(10,7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xi, yi, zi, rstride=1, cstride=1,
                    edgecolor='none', antialiased=True)
    ax.set_xlabel("X (Easting)")
    ax.set_ylabel("Y (Northing)")
    ax.set_zlabel("Z (Elevation)")
    ax.set_title("3D Gridded Surface")
    if out_png:
        fig.savefig(out_png, dpi=300)
        print(f"3D grid plot saved → {out_png}")
    plt.show()


def save_geotiff(xi, yi, zi, crs, out_tif):
    """Сохраняет грид в виде GeoTIFF."""
    height, width = zi.shape
    
    # ДИАГНОСТИКА КООРДИНАТ
    print(f"\n=== ДИАГНОСТИКА GEOTIFF ===")
    print(f"Размер грида: {height} x {width}")
    print(f"xi диапазон: {xi.min():.2f} - {xi.max():.2f}")
    print(f"yi диапазон: {yi.min():.2f} - {yi.max():.2f}")
    print(f"xi[0,0] = {xi[0,0]:.2f}, xi[0,1] = {xi[0,1]:.2f}")
    print(f"yi[0,0] = {yi[0,0]:.2f}, yi[1,0] = {yi[1,0]:.2f}")
    
    # ИСПРАВЛЕНИЕ: отражаем данные по вертикали для правильной ориентации
    print("Применяем вертикальное отражение данных...")
    zi_flipped = np.flipud(zi)  # отражаем массив по вертикали
    
    # Определяем разрешение пикселя
    pixel_width = xi[0,1] - xi[0,0]  # размер пикселя по X
    pixel_height = yi[1,0] - yi[0,0]  # размер пикселя по Y
    
    print(f"Pixel_width: {pixel_width:.2f}")
    print(f"Pixel_height: {pixel_height:.2f}")
    
    # Для правильной геоориентации: верхний левый угол должен быть с максимальной Y
    if pixel_height > 0:
        # Y координаты возрастают вниз в numpy, но должны возрастать вверх в geo
        top_left_x = xi.min()
        top_left_y = yi.max()
        pixel_height_geo = -pixel_height  # отрицательный для правильной ориентации
    else:
        top_left_x = xi.min() 
        top_left_y = yi.min()
        pixel_height_geo = pixel_height
    
    print(f"Верхний левый угол GeoTIFF: ({top_left_x:.2f}, {top_left_y:.2f})")
    print(f"Geo pixel_height: {pixel_height_geo:.2f}")
    
    transform = from_origin(top_left_x, top_left_y, pixel_width, abs(pixel_height_geo))
    print(f"Transform: {transform}")
    print(f"===========================\n")

    with rasterio.open(
        out_tif,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=zi_flipped.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(zi_flipped, 1)

    print(f"GeoTIFF saved → {out_tif}")


def compute_volume_from_grid(zi: np.ndarray, step: float) -> float:
    """Грубый объём (м3) при предположении: shoreline = 0, zi отрицательна как глубина.
    Используется только для сравнения разрешений.
    """
    if zi.size == 0:
        return 0.0
    valid = np.isfinite(zi)
    if not valid.any():
        return 0.0
    # pixel area = step^2 (UTM проекция, квадратная сетка)
    vol_m3 = np.sum((-zi[valid]) * (step * step))
    return float(vol_m3)


def summarize_stats(label: str, zi: np.ndarray, step: float) -> dict:
    valid = np.isfinite(zi)
    if not valid.any():
        return {"label": label, "step": step, "pixels": 0, "min": None, "max": None, "mean": None, "volume_mcm": 0}
    vol_m3 = compute_volume_from_grid(zi, step)
    return {
        "label": label,
        "step": step,
        "pixels": int(valid.sum()),
        "min": float(np.nanmin(zi)),
        "max": float(np.nanmax(zi)),
        "mean": float(np.nanmean(zi)),
        "volume_mcm": vol_m3 / 1e6
    }


def main():
    parser = argparse.ArgumentParser(description="Generate bathymetry grid at one or multiple resolutions.")
    parser.add_argument('--grid_steps', help='Comma separated grid steps (meters), e.g. 100,50,20,10')
    parser.add_argument('--out_prefix', default='bathymetry_hh', help='Base prefix for output rasters')
    parser.add_argument('--stats_csv', default='bathymetry_resolution_stats.csv', help='CSV file for multi-resolution stats')
    parser.add_argument('--no_plots', action='store_true', help='Skip plotting to speed up batch runs')
    args, unknown = parser.parse_known_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 1) Генерация контура
    shore_df = generate_edge_points(INPUT_SHP, NUM_POINTS)

    # при необходимости переводим из lat/lon → UTM
    latlon_mask = shore_df["Y"] < 1_000
    if latlon_mask.any():
        tr = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
        lon = shore_df.loc[latlon_mask, "X"].to_numpy()
        lat = shore_df.loc[latlon_mask, "Y"].to_numpy()
        east, north = tr.transform(lon, lat)
        shore_df.loc[latlon_mask, ["X","Y"]] = np.column_stack([east, north])

    shore_df.to_csv(EDGE_CSV, index=False)
    print(f"Saved {len(shore_df)} shoreline points → '{EDGE_CSV}'")

    # 2) Загрузка и фильтрация interior
    interior = load_interior(INTERIOR_CSV)
    print(f"Loaded shoreline {len(shore_df)} pts, interior {len(interior)} pts")

    interior = median_mad_filter(interior, k=K_NEIGHBORS, t=MAD_THRESHOLD)
    print(f"Interior after MAD-filter: {len(interior)} pts")

    interior = invert_depth(interior)
    save_cleaned_points(interior, CLEANED_INTERIOR_CSV)

    # 3) Объединяем все точки и сохраняем
    all_pts = pd.concat([shore_df, interior], ignore_index=True)
    all_pts = invert_depth(all_pts)  # чтобы Laplacian брал глубины как положительные
    save_cleaned_points(all_pts, ALL_POINTS_CSV)

    # 4) TIN Laplacian-сглаживание
    shore_mask     = all_pts["Z"] == 0
    all_pts["Z"]   = laplacian_smooth(all_pts, shore_mask, iters=SMOOTH_ITERS)

    # 5) Сетка и «умный» пост-обработчик
    shore_poly = Path(shore_df[["X","Y"]].to_numpy())
    # Решаем какие шаги использовать
    if args.grid_steps:
        steps = [float(s.strip()) for s in args.grid_steps.split(',') if s.strip()]
    else:
        steps = [GRID_STEP]

    stats = []
    first_grid = True
    for step in steps:
        print(f"\n=== Building grid step={step} m ===")
        xi, yi, zi = grid_and_sanitize(all_pts, shore_poly,
                                       step=step,
                                       sigma_factor=SIGMA_FACTOR)

        label = f"{args.out_prefix}_{int(step)}m"
        tif_name = f"{label}.tif" if len(steps) > 1 else OUTPUT_TIF
        png_iso  = f"isolines_{int(step)}m.png" if len(steps) > 1 else OUTPUT_PNG
        png_3d   = f"3d_grid_{int(step)}m.png" if len(steps) > 1 else THREE_D_PNG

        # Печать статистики поверхности
        stat = summarize_stats(label, zi, step)
        stats.append(stat)
        print(f"Depth stats: min={stat['min']:.2f} max={stat['max']:.2f} mean={stat['mean']:.2f} volume≈{stat['volume_mcm']:.1f} MCM")

        # Плоты можно пропускать в пакетном режиме
        if not args.no_plots:
            plot_3d_grid(xi, yi, zi, out_png=png_3d)
            plot_and_save_grid(xi, yi, zi, shore_df, png_iso)

        save_geotiff(xi, yi, zi, UTM_EPSG, tif_name)
        first_grid = False

    # Сохранение сводной статистики
    if len(stats) > 1:
        df_stats = pd.DataFrame(stats)
        # Сортируем по шагу
        df_stats = df_stats.sort_values('step')
        df_stats.to_csv(args.stats_csv, index=False)
        print(f"Saved multi-resolution stats → {args.stats_csv}")
        print(df_stats)

    # Завершение, если был только один шаг и старый pipeline ожидает старые имена, они уже созданы.

    # 6) Визуализация и сохранение результатов
if __name__ == "__main__":
    main()
