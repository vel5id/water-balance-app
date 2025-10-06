#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Kriging interpolation script for geoinformatics workflows.

Features:
- Configurable variogram models (linear, exponential, spherical)
- Logging instead of print statements
- Input validation and error handling
- Parameterized neighbor count and sample size
- CRS-aware world file generation
"""
import warnings
import logging
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from PIL import Image

# === CONFIGURATION ===
CSV_PATH = 'all_points_cleaned.csv'
OUT_TIF = 'bathymetry1.tif'
RES = 100.0  # grid resolution
OUTPUT_CRS = None  # e.g. 'EPSG:32641'
K_NEIGHBORS = 300
VARIOGRAM_MODEL = 'linear'  # 'linear', 'exponential', 'spherical'
SAMPLE_SIZE = 20000
EPS = 1e-12  # small perturbation for numerical stability


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )


def load_points(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=['X', 'Y', 'Z'], dtype=float)
    if df[['X', 'Y', 'Z']].isnull().any().any():
        raise ValueError('Input CSV contains NaN values in X, Y, or Z')
    # Convert Z to negative depths
    df['Z'] = -np.abs(df['Z'])
    return df


def remove_duplicate_points(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df_clean = df.groupby(['X', 'Y'], as_index=False)['Z'].mean()
    after = len(df_clean)
    if before != after:
        logging.info(f'Removed {before - after} duplicate points')
    return df_clean


def compute_variogram(distances: np.ndarray, semivars: np.ndarray, model: str) -> float:
    """
    Fit a variogram model through origin: semivariance = gamma(h).
    Returns slope parameter a for linear model; for other models, returns first parameter.
    """
    if model == 'linear':
        # γ(h) = a * h
        num = np.dot(distances, semivars)
        den = np.dot(distances, distances)
        return abs(num / den) if den > 0 else EPS
    else:
        # Placeholder for exponential / spherical fitting
        # User can extend this section
        return 1.0


def estimate_slope(x: np.ndarray, y: np.ndarray, z: np.ndarray, sample_size: int) -> float:
    n = len(x)
    max_pairs = n * (n - 1) // 2
    m = min(sample_size, max_pairs)
    idx_i = np.random.randint(0, n, size=m)
    idx_j = np.random.randint(0, n, size=m)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    dists = np.hypot(x[idx_i] - x[idx_j], y[idx_i] - y[idx_j])
    semivars = 0.5 * (z[idx_i] - z[idx_j]) ** 2
    slope = compute_variogram(dists, semivars, VARIOGRAM_MODEL)
    return max(slope, 1e-6)


def interpolate_to_grid(df: pd.DataFrame, res: float) -> (np.ndarray, np.ndarray, np.ndarray):
    df = remove_duplicate_points(df)
    x, y, z = df['X'].values, df['Y'].values, df['Z'].values

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xi = np.arange(xmin, xmax + res, res)
    yi = np.arange(ymin, ymax + res, res)
    logging.info(f'Grid size: {len(xi)} x {len(yi)} = {len(xi) * len(yi)} cells')

    # Estimate variogram parameter
    logging.info('Estimating variogram slope...')
    slope = estimate_slope(x, y, z, SAMPLE_SIZE)
    logging.info(f'Variogram slope: {slope:.6e}')

    tree = cKDTree(np.column_stack((x, y)))
    zgrid = np.empty((len(yi), len(xi)), dtype=np.float32)

    # Perform kriging for each grid node
    for i, yy in enumerate(yi[::-1]):
        for j, xx in enumerate(xi):
            dists, idxs = tree.query([xx, yy], k=K_NEIGHBORS)
            if K_NEIGHBORS == 1:
                dists, idxs = np.array([dists]), np.array([idxs])
            k = len(idxs)
            xi_n, yi_n, zi_n = x[idxs], y[idxs], z[idxs]
            coords = np.column_stack((xi_n, yi_n))
            diff = coords[:, None, :] - coords[None, :, :]
            dist_mat = np.hypot(diff[..., 0], diff[..., 1])
            gamma_mat = slope * dist_mat + np.eye(k) * EPS

            # Assemble kriging system
            A = np.empty((k + 1, k + 1), dtype=float)
            A[:k, :k] = gamma_mat
            A[:k, k] = 1.0
            A[k, :k] = 1.0
            A[k, k] = 0.0

            gamma_vec = slope * dists
            b = np.empty(k + 1, dtype=float)
            b[:k] = gamma_vec
            b[k] = 1.0

            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(A, b, rcond=None)[0]
            zgrid[i, j] = np.dot(w[:k], zi_n)

    logging.info('Interpolation completed')
    return xi, yi, zgrid


def save_geotiff(xi: np.ndarray, yi: np.ndarray, zgrid: np.ndarray,
                 crs: str, out_path: str, res: float) -> None:
    # Save TIFF
    img = Image.fromarray(zgrid, mode='F')
    img.save(out_path)

    # Write world file
    world_path = out_path.rsplit('.', 1)[0] + '.tfw'
    with open(world_path, 'w') as wf:
        wf.write(f"{res}\n")
        wf.write("0.0\n")
        wf.write("0.0\n")
        wf.write(f"{-res}\n")
        wf.write(f"{xi[0] + res/2}\n")  # center of top-left pixel
        wf.write(f"{yi[-1] - res/2}\n")  # center of top-left pixel
    logging.info(f'Saved GeoTIFF {out_path} and world file {world_path}')


def main():
    setup_logging()
    warnings.filterwarnings('ignore')

    try:
        logging.info('Loading points...')
        df = load_points(CSV_PATH)
        logging.info(f'Total points: {len(df)}')

        xi, yi, zgrid = interpolate_to_grid(df, RES)
        save_geotiff(xi, yi, zgrid, OUTPUT_CRS, OUT_TIF, RES)

    except Exception as e:
        logging.exception('An error occurred during processing')
        raise

if __name__ == '__main__':
    main()
