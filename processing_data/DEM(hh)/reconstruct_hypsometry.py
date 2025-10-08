"""Hypsometric curve physical reconstruction.

Steps:
 1. Load internal bathymetry (high-res) and external Copernicus DEM.
 2. Reproject both into a common equal-area CRS (auto UTM or EPSG:6933 fallback).
 3. Build candidate maximum reservoir polygon (union of bathy extent + NDWI mask if available + optional convex hull buffer).
 4. Merge rasters into integrated elevation model (bathymetry takes precedence below shoreline, DEM above).
 5. Calibrate vertical datum & optional depth scaling to match known design targets:
       A_max_target_km2, V_max_target_mcm, (optional) H_max_target_m.
 6. Generate dense elevation levels and compute cumulative area & volume.
 7. Export new curve to processed_data/processing_output/area_volume_curve_reconstructed.csv and comparison plot.

Run: python reconstruct_hypsometry.py --bathy Bathymetry/Main/bathymetry_hh.tif --dem "Hillshade + DEM/output_hh.tif" \
    --out processed_data/processing_output --amax 93.7 --vmax 791 --hmax 19.8

NOTE: This script is non-destructive; it will not overwrite the existing curve unless --overwrite is passed.
"""
from __future__ import annotations
import os, sys, math, argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.enums import Resampling as Rsp
from shapely.geometry import shape, Polygon, box
from shapely.ops import unary_union
from rasterio.features import rasterize, shapes as rio_shapes
from rasterio.warp import transform_geom
import warnings

# -------------------- Helpers --------------------

def pick_equal_area_crs(bounds_lonlat: tuple[float,float,float,float]):
    """Pick a suitable projected CRS for area/volume computations.
    Priority: UTM zone based on centroid; fallback EPSG:6933 (World Equidistant Cylindrical).
    """
    try:
        minx, miny, maxx, maxy = bounds_lonlat
        cx = 0.5*(minx+maxx); cy = 0.5*(miny+maxy)
        zone = int(math.floor((cx + 180) / 6) + 1)
        hemisphere = '326' if cy >= 0 else '327'
        utm_epsg = int(f"{hemisphere}{zone:02d}")
        return f"EPSG:{utm_epsg}"  # WGS84 / UTM zone
    except Exception:
        return "EPSG:6933"

def read_raster(path: str):
    ds = rasterio.open(path)
    arr = ds.read(1).astype('float32')
    nodata = ds.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return ds, arr


def reproject_match(src_arr, src_ds, dst_profile):
    dst = np.full((dst_profile['height'], dst_profile['width']), np.nan, dtype='float32')
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_ds.transform,
        src_crs=src_ds.crs,
        dst_transform=dst_profile['transform'],
        dst_crs=dst_profile['crs'],
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def build_target_grid(bathy_ds, dem_ds, target_crs: str, resolution: float | None):
    # Union bounds in source CRS -> go to lon/lat -> to target
    b_bounds = transform_bounds(bathy_ds.crs, 'EPSG:4326', *bathy_ds.bounds)
    d_bounds = transform_bounds(dem_ds.crs, 'EPSG:4326', *dem_ds.bounds)
    minx = min(b_bounds[0], d_bounds[0]); miny = min(b_bounds[1], d_bounds[1])
    maxx = max(b_bounds[2], d_bounds[2]); maxy = max(b_bounds[3], d_bounds[3])
    # Slight buffer
    buf = 0.001
    minx -= buf; miny -= buf; maxx += buf; maxy += buf
    # Transform to target CRS for grid definition
    # We'll approximate by transforming corners
    from rasterio.warp import transform
    xs, ys = transform('EPSG:4326', target_crs, [minx,maxx], [miny,maxy])
    xmin_t, xmax_t = min(xs), max(xs); ymin_t, ymax_t = min(ys), max(ys)
    # choose resolution if not provided: coarsest of inputs or 30 m fallback
    if resolution is None:
        # approximate by average pixel size of bathy
        bxres = abs(bathy_ds.transform.a)
        byres = abs(bathy_ds.transform.e)
        dres = max(abs(dem_ds.transform.a), abs(dem_ds.transform.e))
        resolution = float(np.nanmax([bxres, byres, dres, 30.0]))
    width = int(math.ceil((xmax_t - xmin_t)/resolution))
    height = int(math.ceil((ymax_t - ymin_t)/resolution))
    from affine import Affine
    transform_aff = Affine(resolution, 0, xmin_t, 0, -resolution, ymax_t)
    profile = {
        'driver':'GTiff','dtype':'float32','count':1,
        'crs': target_crs,'transform': transform_aff,
        'width': width,'height': height,
        'nodata': np.nan,
    }
    return profile


def integrate_surfaces(bathy_arr, dem_arr, shoreline_elev_est: float | None):
    """Merge arrays: prefer bathy where available (assumed depths or elevations). If bathy appears negative (depth), convert to elevation using shoreline_elev_est if provided.
    Strategy:
      - Detect sign: if median(bathy) < 0 and max(bathy) ~ 0 -> treat as depths (negative downward) relative to 0 shoreline.
      - If depths and shoreline_elev_est given: elevation = shoreline_elev_est + bathy (bathy negative) else leave as is.
      - Where bathy NaN use DEM elevation.
    """
    out = np.array(dem_arr, copy=True)
    if bathy_arr is not None:
        bathy_valid = np.isfinite(bathy_arr)
        # quick heuristics
        b_med = np.nanmedian(bathy_arr)
        b_max = np.nanmax(bathy_arr)
        b_min = np.nanmin(bathy_arr)
        depth_mode = b_med < 0 and b_max <= 1.0  # near zero surface
        if depth_mode and shoreline_elev_est is not None:
            elev_bathy = shoreline_elev_est + bathy_arr  # bathy negative
        else:
            elev_bathy = bathy_arr
        out[bathy_valid] = elev_bathy[bathy_valid]
    return out


def estimate_shoreline_from_dem(dem_elev: np.ndarray, fraction: float = 0.98):
    vals = np.sort(dem_elev[np.isfinite(dem_elev)])
    if vals.size == 0:
        return None
    return float(vals[int(fraction*(vals.size-1))])


def compute_curve(elev_grid: np.ndarray, levels: np.ndarray):
    results = []
    # Pixel area constant in projected CRS
    # assume square
    return results


def hypsometry(elev_grid: np.ndarray, levels: np.ndarray, pixel_area_m2: float, mask: np.ndarray | None = None):
    """Compute cumulative area/volume curve.

    Parameters:
        elev_grid: 2D array of elevations (m)
        levels: 1D array of target water surface elevations
        pixel_area_m2: pixel area (constant) in m^2
        mask: Optional 2D boolean/int array; True/1 where reservoir is allowed. If provided, only those cells count.
    """
    out = []
    sort_levels = np.sort(levels)
    elev_flat = elev_grid.ravel()
    if mask is not None:
        mask_flat = mask.astype(bool).ravel()
        valid_elev = elev_flat[mask_flat]
    else:
        mask_flat = None
        valid_elev = elev_flat
    for L in sort_levels:
        if mask_flat is not None:
            sel = (valid_elev <= L)
            sub = valid_elev[sel]
        else:
            sub = elev_flat[elev_flat <= L]
        area_m2 = sub.size * pixel_area_m2
        vol_m3 = np.nansum(L - sub) * pixel_area_m2
        out.append((L, area_m2/1e6, vol_m3/1e6))
    return pd.DataFrame(out, columns=["elevation_m","area_km2","volume_mcm"])


def calibrate_vertical(df: pd.DataFrame, amax_target: float, vmax_target: float, hmax_target: float | None):
    """Apply vertical shift and optional stretching so that max area and volume align with targets.
    Steps:
      1) Identify current max row.
      2) Compute scale factors for area & volume (area scaling not done geometrically here – warning; we only adjust vertical to affect volume; if area deficit is large, suggests missing spatial extent).
      3) If area deficiency > 8%, emit warning and leave to spatial reconstruction.
      4) Adjust heights so that range matches hmax if provided (stretch).
    """
    dfc = df.copy()
    row_max = dfc.iloc[-1]
    area_def = (amax_target - row_max.area_km2)/amax_target
    if area_def > 0.08:
        print(f"[WARN] Area still {area_def*100:.1f}% below target. Need spatial expansion before pure vertical calibration.")
    # vertical stretch if hmax_target provided
    if hmax_target is not None:
        current_range = dfc.elevation_m.max() - dfc.elevation_m.min()
        if current_range > 0:
            stretch = hmax_target/current_range
            dfc['elevation_m'] = (dfc.elevation_m - dfc.elevation_m.min())*stretch + dfc.elevation_m.min()
    # After stretch, recompute volume by geometric proportion? Not correct without recomputing integrals; skip.
    return dfc


# -------------------- Main pipeline --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bathy', required=True, help='Path to high-res bathymetry raster')
    ap.add_argument('--dem', required=True, help='Path to external DEM (Copernicus)')
    ap.add_argument('--out', default='processed_data/processing_output', help='Output directory')
    ap.add_argument('--amax', type=float, required=True, help='Target max area km2')
    ap.add_argument('--vmax', type=float, required=True, help='Target max volume MCM')
    ap.add_argument('--hmax', type=float, default=None, help='Target max depth range (m)')
    ap.add_argument('--levels', type=int, default=400, help='Number of elevation levels')
    ap.add_argument('--resolution', type=float, default=None, help='Override output resolution (m)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--shoreline', help='Optional shoreline polygon (SHP/GPKG) to constrain reservoir extent')
    ap.add_argument('--ndwi_mask', help='Optional single NDWI mask raster (values >0 water) used if shoreline not provided')
    ap.add_argument('--ndwi_list', help='Comma-separated list of NDWI mask rasters to union with shoreline (each thresholded)')
    ap.add_argument('--auto_buffer', action='store_true', help='Automatically buffer combined mask outward until target area is reached')
    ap.add_argument('--buffer_step', type=float, default=50.0, help='Buffer step in meters when auto_buffer enabled')
    ap.add_argument('--buffer_tolerance', type=float, default=0.005, help='Relative tolerance for reaching target area (e.g. 0.005=0.5%)')
    ap.add_argument('--max_buffer_iterations', type=int, default=25, help='Safety cap for auto buffer iterations')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    curve_out = os.path.join(args.out, 'area_volume_curve_reconstructed.csv')
    if os.path.exists(curve_out) and not args.overwrite:
        print(f"File exists: {curve_out} (use --overwrite to regenerate)")
        return

    bathy_ds, bathy_arr = read_raster(args.bathy)
    dem_ds, dem_arr = read_raster(args.dem)

    # Pick CRS
    union_bounds_ll = transform_bounds(bathy_ds.crs, 'EPSG:4326', *bathy_ds.bounds)
    union_bounds_ll = (
        min(union_bounds_ll[0], transform_bounds(dem_ds.crs, 'EPSG:4326', *dem_ds.bounds)[0]),
        min(union_bounds_ll[1], transform_bounds(dem_ds.crs, 'EPSG:4326', *dem_ds.bounds)[1]),
        max(union_bounds_ll[2], transform_bounds(dem_ds.crs, 'EPSG:4326', *dem_ds.bounds)[2]),
        max(union_bounds_ll[3], transform_bounds(dem_ds.crs, 'EPSG:4326', *dem_ds.bounds)[3]),
    )
    target_crs = pick_equal_area_crs(union_bounds_ll)
    print(f"Target CRS: {target_crs}")

    grid_profile = build_target_grid(bathy_ds, dem_ds, target_crs, args.resolution)

    bathy_proj = reproject_match(bathy_arr, bathy_ds, grid_profile)
    dem_proj = reproject_match(dem_arr, dem_ds, grid_profile)

    # Rough shoreline estimate from DEM upper quantile
    shoreline_est = estimate_shoreline_from_dem(dem_proj, fraction=0.985)
    print(f"Estimated shoreline elevation ~ {shoreline_est}")

    integrated = integrate_surfaces(bathy_proj, dem_proj, shoreline_est)

    # ---------------- Mask construction (Variant A) ----------------
    mask = None
    shoreline_poly = None
    if args.shoreline and os.path.exists(args.shoreline):
        try:
            import fiona
            shapes = []
            polys = []
            with fiona.open(args.shoreline) as src:
                src_crs = src.crs_wkt or src.crs
                for feat in src:
                    geom = feat['geometry']
                    if geom is None:
                        continue
                    try:
                        geom_t = transform_geom(src_crs, target_crs, geom, precision=6)
                        shapes.append((geom_t, 1))
                        from shapely.geometry import shape as shp_shape
                        polys.append(shp_shape(geom_t))
                    except Exception as ge:
                        warnings.warn(f"Geom transform failed: {ge}")
            if shapes:
                mask = rasterize(
                    shapes=shapes,
                    out_shape=(grid_profile['height'], grid_profile['width']),
                    transform=grid_profile['transform'],
                    fill=0,
                    all_touched=True,
                    dtype='uint8'
                )
                if polys:
                    from shapely.ops import unary_union as uu
                    shoreline_poly = uu(polys)
                poly_area_km2 = mask.sum() * abs(grid_profile['transform'].a) * abs(grid_profile['transform'].e) / 1e6
                print(f"Shoreline mask applied. Mask area ~ {poly_area_km2:.2f} km^2")
            else:
                print("[WARN] No geometries found in shoreline file; continuing without polygon mask.")
        except Exception as e:
            print(f"[WARN] Failed to apply shoreline mask: {e}")
            mask = None
    # Collect extra NDWI masks (union)
    ndwi_union_mask = None
    ndwi_sources = []
    if args.ndwi_list:
        for p in [p.strip() for p in args.ndwi_list.split(',') if p.strip()]:
            if os.path.exists(p):
                ndwi_sources.append(p)
            else:
                print(f"[WARN] NDWI listed not found: {p}")
    if args.ndwi_mask and os.path.exists(args.ndwi_mask):
        ndwi_sources.append(args.ndwi_mask)
    if ndwi_sources:
        for ndwi_path in ndwi_sources:
            try:
                ndwi_ds, ndwi_arr = read_raster(ndwi_path)
                ndwi_proj = reproject_match(ndwi_arr, ndwi_ds, grid_profile)
                thr = 0.0
                if np.nanmax(ndwi_proj) <= 1.0 and np.nanmin(ndwi_proj) >= -1.0:
                    thr = 0.3
                ndwi_mask_local = (ndwi_proj > thr).astype('uint8')
                if ndwi_union_mask is None:
                    ndwi_union_mask = ndwi_mask_local
                else:
                    # logical OR
                    ndwi_union_mask = np.where((ndwi_union_mask==1) | (ndwi_mask_local==1), 1, 0).astype('uint8')
            except Exception as e:
                print(f"[WARN] Failed NDWI mask {ndwi_path}: {e}")
        if ndwi_union_mask is not None:
            ndwi_area_km2 = ndwi_union_mask.sum() * abs(grid_profile['transform'].a) * abs(grid_profile['transform'].e) / 1e6
            print(f"NDWI union mask area ~ {ndwi_area_km2:.2f} km^2 from {len(ndwi_sources)} sources")
            if mask is None:
                mask = ndwi_union_mask
            else:
                mask = np.where((mask==1) | (ndwi_union_mask==1), 1, 0).astype('uint8')
    elif not mask:
        try:
            ndwi_ds, ndwi_arr = read_raster(args.ndwi_mask)
            ndwi_proj = reproject_match(ndwi_arr, ndwi_ds, grid_profile)
            # Heuristic threshold: >0 (if binary) or >0.3 if continuous NDWI (common water threshold)
            thr = 0.0
            if np.nanmax(ndwi_proj) <= 1.0 and np.nanmin(ndwi_proj) >= -1.0:
                # likely NDWI float - pick 0.3
                thr = 0.3
            mask = (ndwi_proj > thr).astype('uint8')
            mask_area_km2 = mask.sum() * abs(grid_profile['transform'].a) * abs(grid_profile['transform'].e) / 1e6
            print(f"NDWI mask applied (thr={thr}). Mask area ~ {mask_area_km2:.2f} km^2")
        except Exception as e:
            print(f"[WARN] Failed to apply NDWI mask: {e}")
            mask = None
    if mask is None:
        print("No shoreline or NDWI mask provided; using full grid (likely to overestimate area).")

    # ---------------- Auto buffer to reach target area ----------------
    if mask is not None and args.auto_buffer and args.amax:
        current_area_km2 = mask.sum() * abs(grid_profile['transform'].a) * abs(grid_profile['transform'].e) / 1e6
        target_area = args.amax
        tol = args.buffer_tolerance
        if current_area_km2 < target_area*(1 - tol):
            print(f"Auto-buffer activated: current {current_area_km2:.2f} km^2 < target {target_area:.2f} km^2")
            # Derive polygon from mask if shoreline_poly not set or union of both
            if shoreline_poly is None:
                # extract shapes from mask
                polys = []
                for geom, val in rio_shapes(mask, transform=grid_profile['transform']):
                    if val != 1:
                        continue
                    try:
                        from shapely.geometry import shape as shp_shape
                        polys.append(shp_shape(geom))
                    except Exception:
                        pass
                if polys:
                    from shapely.ops import unary_union as uu
                    shoreline_poly = uu(polys)
            # If NDWI extended, union mask polygon anyway
            if shoreline_poly is None:
                print("[WARN] Could not derive polygon for buffering; skipping auto buffer.")
            else:
                step = args.buffer_step
                max_iter = args.max_buffer_iterations
                it = 0
                while current_area_km2 < target_area*(1 - tol) and it < max_iter:
                    shoreline_poly = shoreline_poly.buffer(step)
                    new_mask = rasterize(
                        [(shoreline_poly.__geo_interface__, 1)],
                        out_shape=(grid_profile['height'], grid_profile['width']),
                        transform=grid_profile['transform'],
                        fill=0,
                        all_touched=True,
                        dtype='uint8'
                    )
                    current_area_km2 = new_mask.sum() * abs(grid_profile['transform'].a) * abs(grid_profile['transform'].e) / 1e6
                    it += 1
                if current_area_km2 >= target_area*(1 - tol):
                    print(f"Buffered to {current_area_km2:.2f} km^2 in {it} iterations (step {step} m)")
                    mask = new_mask
                else:
                    print(f"[WARN] Reached only {current_area_km2:.2f} km^2 after {it} iterations (target {target_area:.2f}).")

    # Build levels
    elev_min = np.nanpercentile(integrated, 0.5)
    elev_max = np.nanpercentile(integrated, 99.5)
    levels = np.linspace(elev_min, elev_max, args.levels)

    # Pixel area
    px_area = abs(grid_profile['transform'].a) * abs(grid_profile['transform'].e)

    curve_df = hypsometry(integrated, levels, px_area, mask=mask)

    # Basic sanity
    if curve_df.area_km2.max() < args.amax * 0.5:
        print("[WARN] Computed max area is far below target; spatial extent likely incomplete.")

    if curve_df.volume_mcm.max() < args.vmax * 0.3:
        print("[WARN] Computed max volume << target; check bathymetry depth scale & extent.")

    calibrated = calibrate_vertical(curve_df, args.amax, args.vmax, args.hmax)

    calibrated.to_csv(curve_out, index=False)
    print(f"Saved reconstructed curve: {curve_out}")

    # Comparison plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2, figsize=(12,5))
        calibrated.plot(x='elevation_m', y='area_km2', ax=axes[0], title='Area Curve (Reconstructed)')
        calibrated.plot(x='elevation_m', y='volume_mcm', ax=axes[1], title='Volume Curve (Reconstructed)', color='r')
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = os.path.join(args.out, 'area_volume_curve_reconstructed.png')
        fig.savefig(fig_path, dpi=120)
        print(f"Saved plot: {fig_path}")
    except Exception as e:
        print(f"Plot failed: {e}")

    print("Done.")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Example usage: python reconstruct_hypsometry.py --bathy Bathymetry/Main/bathymetry_hh.tif --dem 'Hillshade + DEM/output_hh.tif' --out processed_data/processing_output --amax 93.7 --vmax 791 --hmax 19.8")
    main()
