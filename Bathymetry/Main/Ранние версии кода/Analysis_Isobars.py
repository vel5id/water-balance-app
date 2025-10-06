#!/usr/bin/env python3
"""
Изолинии для водоёма из двух файлов:
    • edge_points.csv              – берег (Z = 0)
    • representative_1000_points.csv  – глубины внутри

Береговые точки остаются, внутренние чистятся медиан-MAD-фильтром,
затем рисуем и сохраняем карту изолиний.
"""

# ------------- SETTINGS -------------
EDGE_CSV       = "edge_points.csv"
INTERIOR_CSV   = "representative_20k_points_full.csv"
OUTPUT_PNG     = "isolines4_k30.png"   # куда сохранить картинку
K_NEIGHBORS    = 400               # соседей для фильтра
MAD_THRESHOLD  = 3.0
UTM_EPSG       = "EPSG:32641"     # UTM zone 41 N
# ------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from pyproj import Transformer


def load_shore(path: str) -> pd.DataFrame:
    """Чтение + LL→UTM при необходимости."""
    shore = pd.read_csv(path)
    latlon = shore["Y"] < 1_000          # маленькие значения → широта/долгота

    if latlon.any():
        tr = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
        lon = shore.loc[latlon, "X"].to_numpy()
        lat = shore.loc[latlon, "Y"].to_numpy()
        east, north = tr.transform(lon, lat)
        shore.loc[latlon, ["X", "Y"]] = np.column_stack([east, north])

    return shore


def load_interior(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def median_mad_filter(df: pd.DataFrame, k: int = 10, t: float = 3.0) -> pd.DataFrame:
    """Удаляет выбросы по критерию |ΔZ| > t·MAD относительно k ближайших соседей."""
    coords = df[["X", "Y"]].to_numpy()
    z = df["Z"].to_numpy()
    tree = KDTree(coords)

    keep = np.ones(len(df), dtype=bool)
    for i, p in enumerate(coords):
        _, idx = tree.query(p, k=k)
        neigh_z = z[idx]
        med = np.median(neigh_z)
        mad = np.median(np.abs(neigh_z - med))
        thr = 2 if mad == 0 else t * mad
        keep[i] = abs(z[i] - med) <= thr
    return df[keep]


def plot_and_save(clean: pd.DataFrame, shore: pd.DataFrame, out_png: str, levels: int = 15):
    """Строит изолинии, показывает и сохраняет PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))

    filled = ax.tricontourf(clean["X"], clean["Y"], clean["Z"], levels=levels)
    ax.tricontour(clean["X"], clean["Y"], clean["Z"], levels=levels, linewidths=0.5)

    ax.scatter(shore["X"], shore["Y"], s=6, c="black", label="Shoreline (Z=0)")
    fig.colorbar(filled, ax=ax, label="Z-value")

    ax.set_xlabel("Easting (m, UTM 41 N)")
    ax.set_ylabel("Northing (m, UTM 41 N)")
    ax.set_title("Isolines with shoreline kept")
    ax.legend()
    plt.tight_layout()

    # --- Сохранение ---
    fig.savefig(out_png, dpi=300)
    print(f"Plot saved → {out_png}")

    plt.show()


def main():
    shore = load_shore(EDGE_CSV)
    interior = load_interior(INTERIOR_CSV)

    print(f"Shoreline: {len(shore)} pts | Interior: {len(interior)} pts")

    filtered_int = median_mad_filter(interior, k=K_NEIGHBORS, t=MAD_THRESHOLD)
    removed = len(interior) - len(filtered_int)
    print(f"Removed {removed} outliers ({removed/len(interior):.1%})")

    clean = pd.concat([filtered_int, shore], ignore_index=True)
    plot_and_save(clean, shore, OUTPUT_PNG)


if __name__ == "__main__":
    main()
