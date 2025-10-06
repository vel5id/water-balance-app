import os
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
try:
    from pykrige.ok import OrdinaryKriging
    KRIGE_AVAILABLE = True
except ImportError:
    KRIGE_AVAILABLE = False

import rasterio
from rasterio.transform import from_origin
from matplotlib import colormaps
from PIL import Image

# -------------------------------------------------------------------
# User settings: simply change paths/params below
INPUT_CSV    = "./all_points_cleaned.csv"  # CSV with columns X, Y, Z
OUTPUT_FILE  = "./dem400.png"                 # .tif, .png, .jpg
RESOLUTION   = 100.0                           # grid resolution
METHOD       = "nearest"                    # 'nearest','linear','cubic','kriging'
CRS          = "EPSG:32641"                 # CRS for GeoTIFF, or None
# -------------------------------------------------------------------


def read_csv(path):
    # only load needed cols, with float32 dtype for speed
    df = pd.read_csv(path, usecols=["X","Y","Z"], dtype={"X":"float32","Y":"float32","Z":"float32"})
    return df


def make_grid(xs, ys, resolution):
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    xi = np.arange(xmin, xmax + resolution, resolution, dtype='float32')
    yi = np.arange(ymin, ymax + resolution, resolution, dtype='float32')
    X, Y = np.meshgrid(xi, yi)
    return X, Y, xmin, ymax


def interpolate_points(xs, ys, zs, X, Y, method):
    xs = xs.astype('float32'); ys = ys.astype('float32'); zs = zs.astype('float32')
    pts = np.column_stack((xs, ys))
    if method == 'nearest':
        tree = cKDTree(pts)
        grid_pts = np.column_stack((X.ravel(), Y.ravel()))
        _, idx = tree.query(grid_pts, k=1)
        Z = zs[idx].reshape(X.shape)
        return Z
    elif method == 'kriging':
        if not KRIGE_AVAILABLE:
            raise ImportError('pykrige not installed')
        try:
            ok = OrdinaryKriging(xs, ys, zs,
                                 variogram_model='spherical', verbose=False, enable_plotting=False)
            Z, _ = ok.execute('grid', X[0,:], Y[:,0])
            return Z.astype('float32')
        except KeyboardInterrupt:
            print("Kriging interrupted by user. Falling back to linear interpolation.")
            return griddata((xs, ys), zs, (X, Y), method='linear').astype('float32')
    else:
        Z = griddata((xs, ys), zs, (X, Y), method=method)
        return Z.astype('float32')


def write_output(path, Z, xmin, ymax, resolution, crs=None):
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    if ext in ['.tif', '.tiff']:
        height, width = Z.shape
        transform = from_origin(xmin, ymax, resolution, resolution)
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'float32',
            'transform': transform
        }
        if crs:
            profile['crs'] = crs
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(Z, 1)

    elif ext in ['.png', '.jpg', '.jpeg']:
        # normalize to 0-255 and apply colormap
        Zmin, Zmax = np.nanmin(Z), np.nanmax(Z)
        norm = (Z - Zmin) / (Zmax - Zmin + 1e-8)
        cmap = colormaps['terrain']
        rgba = cmap(norm, bytes=True)
        rgb = rgba[...,:3]
        img = Image.fromarray(rgb)
        img.save(path)

    else:
        raise ValueError(f"Unsupported format: {ext}")


def generate_dem(input_file, output_file, resolution, method, crs):
    df = read_csv(input_file)
    xs, ys, zs = df['X'].values, df['Y'].values, df['Z'].values
    X, Y, xmin, ymax = make_grid(xs, ys, resolution)
    Z = interpolate_points(xs, ys, zs, X, Y, method)
    write_output(output_file, Z, xmin, ymax, resolution, crs)
    print(f"DEM saved: {output_file}")


if __name__ == '__main__':
    generate_dem(
        INPUT_CSV,
        OUTPUT_FILE,
        RESOLUTION,
        METHOD,
        CRS
    )
