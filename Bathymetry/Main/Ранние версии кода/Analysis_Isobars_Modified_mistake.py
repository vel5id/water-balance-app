#!/usr/bin/env python3
"""
Isolines with TIN-smoothing only — no MAD filter, no post-processing.
"""

# ---------- SETTINGS ----------
EDGE_CSV               = "edge_points.csv"
INTERIOR_CSV           = "representative_20000_points.csv"
CLEANED_INTERIOR_CSV   = "interior_unfiltered.csv"
ALL_POINTS_CSV         = "all_points_unfiltered.csv"
OUTPUT_PNG             = "isolines_raw.png"

UTM_EPSG               = "EPSG:32641"

SMOOTH_ITERS           = 2
GRID_STEP              = 100.0
# ------------------------------

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
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

def save_cleaned_points(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} points to '{path}'")

# ---------- Laplacian smoothing ----------
def laplacian_smooth(points: pd.DataFrame,
                     shore_mask: np.ndarray,
                     iters: int = 2) -> np.ndarray:
    coords = points[["X", "Y"]].to_numpy()
    z      = points["Z"].to_numpy().copy()
    tri    = Delaunay(coords)

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

# ---------- Raw grid (NO post-filter) ----------
def grid_raw(points: pd.DataFrame,
             shore_poly: Path,
             step=100.0):
    x_min, x_max = points.X.min(), points.X.max()
    y_min, y_max = points.Y.min(), points.Y.max()
    xi = np.arange(x_min, x_max + step, step)
    yi = np.arange(y_min, y_max + step, step)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata(points[["X", "Y"]].to_numpy(),
                  points["Z"].to_numpy(),
                  (xi, yi),
                  method="linear")

    inside = shore_poly.contains_points(np.c_[xi.ravel(), yi.ravel()]) \
                             .reshape(zi.shape)
    zi[~inside] = np.nan

    return xi, yi, zi
# -----------------------------------------------

# ---------- plotting ----------
def plot_and_save_grid(xi, yi, zi, shore_df, out_png, levels=20):
    fig, ax = plt.subplots(figsize=(8, 6))
    filled = ax.contourf(xi, yi, zi, levels=levels)

    ax.scatter(shore_df.X, shore_df.Y, s=6, c="black", label="Shoreline (Z=0)")
    fig.colorbar(filled, ax=ax, label="Z-value")

    ax.set_xlabel("Easting (m, UTM 41 N)")
    ax.set_ylabel("Northing (m, UTM 41 N)")
    ax.set_title("Isolines (raw: no MAD, no post-processing)")
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

    # No filtering at all
    save_cleaned_points(interior, CLEANED_INTERIOR_CSV)

    all_pts = pd.concat([shore, interior], ignore_index=True)
    save_cleaned_points(all_pts, ALL_POINTS_CSV)

    shore_mask   = all_pts["Z"] == 0
    all_pts["Z"] = laplacian_smooth(all_pts, shore_mask, iters=SMOOTH_ITERS)

    shore_poly = Path(shore[["X", "Y"]].to_numpy())
    xi, yi, zi = grid_raw(all_pts, shore_poly, step=GRID_STEP)

    plot_and_save_grid(xi, yi, zi, shore, OUTPUT_PNG)

if __name__ == "__main__":
    main()
