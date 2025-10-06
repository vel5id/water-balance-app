#!/usr/bin/env python3
"""
Isolines with TIN-smoothing + smart post-processing + tqdm progress bars.
"""

# ---------- SETTINGS ----------
EDGE_CSV               = "edge_points.csv"
INTERIOR_CSV           = "representative_80000_points.csv"
CLEANED_INTERIOR_CSV   = "interior_cleaned.csv"        # ← новый файл
ALL_POINTS_CSV         = "all_points_cleaned.csv"      # ← (опционально)
OUTPUT_PNG             = "isolines_p80000_k10_mad.png.png"

UTM_EPSG               = "EPSG:32641"

K_NEIGHBORS            = 10
MAD_THRESHOLD          = 3.0

SMOOTH_ITERS           = 2
GRID_STEP              = 100.0
SIGMA_FACTOR           = 1.5
# ------------------------------

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, KDTree
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter
from matplotlib.path import Path
from pyproj import Transformer
from tqdm import tqdm

# ---------- loading helpers ----------
def load_shore(path: str) -> pd.DataFrame:
    shore = pd.read_csv(path)
    latlon = shore["Y"] < 1_000
    if latlon.any():
        tr = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
        lon = shore.loc[latlon, "X"].to_numpy()
        lat = shore.loc[latlon, "Y"].to_numpy()
        east, north = tr.transform(lon, lat)
        shore.loc[latlon, ["X", "Y"]] = np.column_stack([east, north])
    return shore

def load_interior(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
# -------------------------------------

# ---------- filters ----------
def median_mad_filter(df: pd.DataFrame, k=10, t=3.0) -> pd.DataFrame:
    coords = df[["X", "Y"]].to_numpy()
    z = df["Z"].to_numpy()
    tree = KDTree(coords)
    keep = np.ones(len(df), bool)

    for i in tqdm(range(len(df)), desc="MAD filter"):
        _, idx = tree.query(coords[i], k=k)
        neigh = z[idx]
        med = np.median(neigh)
        mad = np.median(np.abs(neigh - med))
        thr = 2 if mad == 0 else t * mad
        keep[i] = abs(z[i] - med) <= thr
    return df[keep]
# ------------------------------------

# ---------- utility: save cleaned points ----------
def save_cleaned_points(df: pd.DataFrame, path: str):
    """
    Сохраняет отфильтрованные/обработанные точки в CSV
    и выводит сообщение о количестве точек.
    """
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} points to '{path}'")
# -----------------------------------------

# ---------- Laplacian smoothing ----------
def laplacian_smooth(points: pd.DataFrame,
                     shore_mask: np.ndarray,
                     iters: int = 2) -> np.ndarray:
    coords = points[["X", "Y"]].to_numpy()
    z      = points["Z"].to_numpy().copy()
    tri    = Delaunay(coords)

    # список соседей
    neigh = [[] for _ in range(len(points))]
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
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
# -----------------------------------------

# ---------- Grid & smart post ------------
def grid_and_sanitize(points: pd.DataFrame,
                      shore_poly: Path,
                      step=5.0,
                      sigma_factor=1.5):
    x_min, x_max = points.X.min(), points.X.max()
    y_min, y_max = points.Y.min(), points.Y.max()
    xi = np.arange(x_min, x_max + step, step)
    yi = np.arange(y_min, y_max + step, step)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata(points[["X", "Y"]].to_numpy(),
                  points["Z"].to_numpy(),
                  (xi, yi),
                  method="linear")

    # маска озера
    inside = shore_poly.contains_points(np.c_[xi.ravel(), yi.ravel()]) \
                             .reshape(zi.shape)
    zi[~inside] = np.nan

    # --- локальная σ 3×3 ---
    def win_std(a):
        if np.isnan(a).all():
            return 0.0
        return np.nanstd(a)

    sigma_map = generic_filter(zi, win_std, size=3, mode="nearest")
    thresh = sigma_factor * np.nanmedian(sigma_map)

    def std_median_filter(values):
        center = values[4]
        s = np.nanstd(values)
        if s > thresh:
            return np.nanmedian(values)
        return center

    zi = generic_filter(zi, std_median_filter, size=3, mode="nearest")

    # --- глобальный клип ---
    z_flat = zi[np.isfinite(zi)]
    mu, sigma = np.nanmean(z_flat), np.nanstd(z_flat)
    zi = np.clip(zi, mu - 3 * sigma, mu + 3 * sigma)

    return xi, yi, zi
# -----------------------------------------

# ---------- plotting ----------
def plot_and_save_grid(xi, yi, zi, shore_df, out_png, levels=20):
    fig, ax = plt.subplots(figsize=(8, 6))
    filled = ax.contourf(xi, yi, zi, levels=levels)

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
# ---------------------------------

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    shore    = load_shore(EDGE_CSV)
    interior = load_interior(INTERIOR_CSV)
    print(f"Loaded shoreline {len(shore)} pts, interior {len(interior)} pts")

    # --- MAD-фильтр и сохранение результата ---
    interior = median_mad_filter(interior, k=K_NEIGHBORS, t=MAD_THRESHOLD)
    print(f"Interior after MAD-filter: {len(interior)} pts")
    save_cleaned_points(interior, CLEANED_INTERIOR_CSV)

    # --- объединяем все точки и сохраняем (опционально) ---
    all_pts = pd.concat([shore, interior], ignore_index=True)
    save_cleaned_points(all_pts, ALL_POINTS_CSV)

    # --- Laplacian TIN smoothing ---
    shore_mask   = all_pts["Z"] == 0
    all_pts["Z"] = laplacian_smooth(all_pts, shore_mask, iters=SMOOTH_ITERS)

    # --- regular grid + smart post ---
    shore_poly = Path(shore[["X", "Y"]].to_numpy())
    xi, yi, zi = grid_and_sanitize(all_pts, shore_poly,
                                   step=GRID_STEP,
                                   sigma_factor=SIGMA_FACTOR)

    plot_and_save_grid(xi, yi, zi, shore, OUTPUT_PNG)

if __name__ == "__main__":
    main()
