import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from pykrige.uk import UniversalKriging
from shapely.ops import unary_union
from tqdm import tqdm

# -----------------------------------------------
# Настройка логирования для отладки
# -----------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------
# 1. Параметры входа/выхода
# -----------------------------------------------
CSV_SOUNDINGS = "soundings.csv"   # CSV: X, Y, Z (Z<0)
SHORELINE_SHP = "shoreline.shp"   # полигон берега
OUTPUT_TIF    = "bathymetry.tif"
RESOLUTION    = 5                   # шаг сетки в метрах

logger.debug("Параметры: CSV=%s, SHP=%s, OutTIF=%s, Res=%sm",
             CSV_SOUNDINGS, SHORELINE_SHP, OUTPUT_TIF, RESOLUTION)

# -----------------------------------------------
# 2. Чтение и валидация эхолота
# -----------------------------------------------
df = pd.read_csv(CSV_SOUNDINGS).dropna(subset=["X", "Y", "Z"])
logger.debug("Саундингов прочитано: %d", df.shape[0])
if df.shape[0] < 10:
    logger.error("Недостаточно саундингов: %d < 10", df.shape[0])
    raise RuntimeError(f"Нужно ≥10 саундингов, получено {df.shape[0]}")

soundings = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.X, df.Y),
    crs=None
)

# -----------------------------------------------
# 3. Береговая линия: чтение и правка геометрии
# -----------------------------------------------
shore = gpd.read_file(SHORELINE_SHP)
logger.debug("Полигоны берега прочитаны: %d", len(shore))
shore["geometry"] = shore.buffer(0)
shore = shore.explode(index_parts=False).reset_index(drop=True)
if not shore.is_valid.all():
    logger.error("Найдены невалидные геометрии в %s", SHORELINE_SHP)
    raise RuntimeError("shoreline.shp содержит невалидные геометрии")

# -----------------------------------------------
# 4. Проекция в метрический CRS (UTM)
# -----------------------------------------------
# Если уже метрический CRS, можно пропустить to_crs
target_crs = shore.estimate_utm_crs()
logger.debug("Выбран CRS: %s", target_crs)
shore      = shore.to_crs(target_crs)
soundings  = soundings.set_crs(shore.crs, allow_override=True).to_crs(target_crs)

# -----------------------------------------------
# 5. Генерация береговых точек Z=0
# -----------------------------------------------
boundary  = unary_union(shore.geometry).boundary
perim_len = boundary.length
n_shore   = int(np.ceil(perim_len / RESOLUTION))
logger.debug("Периметр контура: %.2f м, генерируем %d береговых точек", perim_len, n_shore)

dists = np.linspace(0, perim_len, n_shore)
shore_pts_list = []
for d in tqdm(dists, desc="Генерация береговых точек", unit="pt"):
    shore_pts_list.append(boundary.interpolate(d))

shore_pts = gpd.GeoDataFrame(
    {"Z": np.zeros(len(shore_pts_list))},
    geometry=shore_pts_list,
    crs=target_crs
)

# -----------------------------------------------
# 6. Объединение точек
# -----------------------------------------------
soundings = soundings[["geometry", "Z"]]
pts_all   = pd.concat([soundings, shore_pts], ignore_index=True)
pts       = gpd.GeoDataFrame(pts_all, geometry="geometry", crs=target_crs)
logger.debug("Всего точек для кригинга: %d (эхо + берег)", len(pts))

# -----------------------------------------------
# 7. Настройка растра и маски воды
# -----------------------------------------------
minx, miny, maxx, maxy = shore.total_bounds
width  = int(np.ceil((maxx - minx) / RESOLUTION))
height = int(np.ceil((maxy - miny) / RESOLUTION))
logger.debug("Grid size: %d cols x %d rows", width, height)
if width < 1 or height < 1:
    logger.error("Неверные размеры сетки: %d x %d", width, height)
    raise RuntimeError(f"Неверный grid: {width}×{height}")

transform = from_origin(minx, maxy, RESOLUTION, RESOLUTION)
mask      = geometry_mask(
    shore.geometry,
    out_shape=(height, width),
    transform=transform,
    invert=True
)
dist_map = distance_transform_edt(~mask) * RESOLUTION

# -----------------------------------------------
# 8. Расчет d_shore и фильтрация
# -----------------------------------------------
xs_coords = np.array([pt.x for pt in pts.geometry])
ys_coords = np.array([pt.y for pt in pts.geometry])

xs = np.clip(np.floor((xs_coords - minx) / RESOLUTION).astype(int), 0, width - 1)
ys = np.clip(np.floor((maxy - ys_coords) / RESOLUTION).astype(int), 0, height - 1)

pts["d_shore"] = dist_map[ys, xs]
pts_before = len(pts)
pts = pts.dropna(subset=["d_shore", "Z"]).reset_index(drop=True)
logger.debug("Точек после фильтрации NaN: %d (было %d)", len(pts), pts_before)
if len(pts) < 5:
    logger.error("Слишком мало точек после фильтрации: %d", len(pts))
    raise RuntimeError(f"После фильтрации осталось только {len(pts)} точек")

# -----------------------------------------------
# 9. Подготовка gridx/gridy и инициализация UK
# -----------------------------------------------
gridx = np.linspace(minx, maxx, width)
gridy = np.linspace(miny, maxy, height)
logger.info("Подготовлены gridx (%d) и gridy (%d)", len(gridx), len(gridy))

logger.info("Инициализация Universal Kriging модели с %d точками...", len(pts))
try:
    UK = UniversalKriging(
        pts.geometry.x.values,
        pts.geometry.y.values,
        pts.Z.values,
        variogram_model="spherical",
        drift_terms=["specified"],
        specified_drift=[pts["d_shore"].values]
    )
    logger.info("Модель Universal Kriging инициализирована")
except Exception as e:
    logger.exception("Ошибка инициализации Universal Kriging: %s", e)
    raise

# -----------------------------------------------
# 10. Кригинг по строкам с прогресс-баром
# -----------------------------------------------
logger.info("Начало криговой интерполяции по строкам...")
zgrid = np.zeros((height, width), dtype=float)
for j, y in enumerate(tqdm(gridy, desc="Кригинг строк", unit="стр")):
    try:
        z_row, _ = UK.execute("grid", gridx, [y])
        zgrid[j, :] = z_row
    except Exception as e:
        logger.exception("Ошибка кригинга на строке %d (y=%.2f): %s", j, y, e)
        zgrid[j, :] = np.nan

zgrid = np.flipud(zgrid)
logger.info("Кригинг завершён")

# -----------------------------------------------
# 11. Сохранение GeoTIFF
# -----------------------------------------------
profile = {
    "driver": "GTiff",
    "dtype": "float32",
    "count": 1,
    "height": height,
    "width": width,
    "crs": target_crs.to_string(),
    "transform": transform
}
with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
    dst.write(zgrid.astype("float32"), 1)
logger.info("Готово: батиметрическая модель сохранена в %s", OUTPUT_TIF)
