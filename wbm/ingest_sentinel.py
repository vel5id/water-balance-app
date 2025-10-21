from __future__ import annotations

"""
Sentinel-2 NDWI ingestion: compute daily water area and convert to volume.

Inputs (auto-detected by default):
- Searches recursively for triplets per date: B03 (Green), B08 (NIR), SCL (scene classification)
  under these roots (in order):
    1) <DATA_ROOT>/Santinel
    2) <DATA_ROOT>/raw_data/Santinel
    3) <DATA_ROOT>

Algorithm:
- Read B03 grid as target grid; reproject B08 and SCL to B03 grid.
- Cloud mask via SCL classes: {3, 8, 9, 10, 11} (cloud/shadows/high prob).
- NDWI = (Green - NIR) / (Green + NIR); threshold > ndwi_threshold for water.
- Compute water area (km^2) using pixel area from raster transform and CRS.
- Convert area -> volume via processed area_volume_curve.csv (linear, clipped).

Outputs:
- processed_data/water_balance_output/sentinel_area_volume.csv
   columns: date, area_km2, volume_mcm, source
- processed_data/water_balance_output/sentinel_validation.txt
   quick comparison vs reference: Area 93.7 km^2, Volume 791 MCM, Useful 562 MCM, Depth 19.8 m

CLI:
  python -m wbm.ingest_sentinel --root <SEARCH_DIR> \
      --curve processed_data/processing_output/area_volume_curve.csv \
      --out processed_data/water_balance_output/sentinel_area_volume.csv \
      --ndwi-threshold 0.275

Notes:
- This module avoids any dependency on IMERG/GLEAM; it's a minimal ingest for area+volume.
"""

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd


DATE_RX = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass
class Triplet:
    date: str
    b03: str
    b08: str
    scl: str


def _pixel_area_km2(transform, crs, bounds) -> float:
    """Compute pixel area (km^2) from raster transform/CRS.

    - For projected CRS (meters), use |a*e - b*d|.
    - For geographic CRS (degrees), convert degree sizes to meters at center lat.
    """
    # Affine components
    a = transform.a
    b = transform.b
    d_ = transform.d
    e = transform.e
    if crs is not None and getattr(crs, "is_geographic", False) is False:
        pixel_area_m2 = abs(a * e - b * d_)
        return float(pixel_area_m2) / 1_000_000.0
    # Geographic degrees → meters with latitude correction
    pixel_width_deg = abs(a)
    pixel_height_deg = abs(e)
    center_lat = (bounds.bottom + bounds.top) / 2.0
    m_per_deg_lat = 111_132.954
    m_per_deg_lon = 111_320.0 * float(np.cos(np.radians(center_lat)))
    pixel_area_m2 = (pixel_width_deg * m_per_deg_lon) * (pixel_height_deg * m_per_deg_lat)
    return float(pixel_area_m2) / 1_000_000.0


def find_triplets(search_root: Path) -> List[Triplet]:
    """Find all dates that have B03, B08, and SCL files.

    Matches any file containing 'Sentinel-2_L2A' and extracts first YYYY-MM-DD from basename.
    """
    files = list(search_root.rglob("*Sentinel-2_L2A*"))
    groups: Dict[str, Dict[str, str]] = {}
    for p in files:
        name = p.name
        m = DATE_RX.search(name)
        if not m:
            continue
        date = m.group(1)
        d = groups.setdefault(date, {})
        low = name.lower()
        if "_b03" in low:
            d["b03"] = str(p)
        elif "_b08" in low:
            d["b08"] = str(p)
        elif "scene_classification_map" in low or "_scl" in low:
            d["scl"] = str(p)
    out: List[Triplet] = []
    for date, g in groups.items():
        if {"b03", "b08", "scl"}.issubset(g.keys()):
            out.append(Triplet(date=date, b03=g["b03"], b08=g["b08"], scl=g["scl"]))
    out.sort(key=lambda t: t.date)
    return out


def find_triplets_fallback_legacy(search_root: Path) -> List[Triplet]:
    """Fallback: use legacy raw_data/Santinel/ndwi.py finder if present."""
    ndwi_py = search_root / "ndwi.py"
    if not ndwi_py.exists():
        # try parent
        ndwi_py = search_root.parent / "ndwi.py"
        if not ndwi_py.exists():
            return []
    import importlib.util
    spec = importlib.util.spec_from_file_location("legacy_ndwi", str(ndwi_py))
    if spec is None or spec.loader is None:
        return []
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)  # type: ignore[attr-defined]
    try:
        groups = legacy.find_sentinel_files(str(search_root))  # returns list of tuples
    except Exception:
        return []
    out: List[Triplet] = []
    for item in groups:
        if len(item) >= 4:
            date, b03, b08, scl = item[0], item[1], item[2], item[3]
            out.append(Triplet(date=date, b03=b03, b08=b08, scl=scl))
    out.sort(key=lambda t: t.date)
    return out


def compute_area_km2(
    b03_path: str,
    b08_path: str,
    scl_path: str,
    ndwi_threshold: float,
    aoi_wgs84: Optional[dict] = None,
    want_aoi_area: bool = False,
) -> Tuple[float, float, Optional[float]]:
    """Return (area_km2, pixel_area_km2, aoi_area_km2?). If want_aoi_area is False, last is None."""
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_geom
    from rasterio.features import geometry_mask

    # Read target grid (B03)
    with rasterio.open(b03_path) as gsrc:
        green = gsrc.read(1).astype("float32")
        profile = gsrc.profile
        target_shape = gsrc.shape
        px_area_km2 = _pixel_area_km2(gsrc.transform, gsrc.crs, gsrc.bounds)
        transform = profile["transform"]
        crs = profile.get("crs")

    # Reproject B08 to B03 grid
    with rasterio.open(b08_path) as nsrc:
        nir = np.empty(target_shape, dtype="float32")
        reproject(
            source=nsrc.read(1),
            destination=nir,
            src_transform=nsrc.transform,
            src_crs=nsrc.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
        )

    # Reproject SCL to B03 grid
    with rasterio.open(scl_path) as ssrc:
        scl = np.empty(target_shape, dtype="uint8")
        reproject(
            source=ssrc.read(1),
            destination=scl,
            src_transform=ssrc.transform,
            src_crs=ssrc.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )

    # Cloud/shadow classes to mask
    cloud_mask = np.isin(scl, [3, 8, 9, 10, 11])
    green = green.copy()
    nir = nir.copy()
    green[cloud_mask] = np.nan
    nir[cloud_mask] = np.nan

    # NDWI and thresholding
    np.seterr(divide="ignore", invalid="ignore")
    ndwi = (green - nir) / (green + nir)
    water_mask = np.nan_to_num(ndwi) > ndwi_threshold

    # Optional AOI clipping
    aoi_area_km2: Optional[float] = None
    if aoi_wgs84 is not None:
        try:
            aoi_proj = transform_geom("EPSG:4326", crs, aoi_wgs84, precision=6)
            # geometry_mask with invert=True yields True INSIDE the geometry
            aoi_mask = geometry_mask(
                [aoi_proj],
                transform=transform,
                invert=True,
                out_shape=target_shape,
            )
            # aoi_mask is True-inside; combine with water
            water_mask = np.logical_and(water_mask, aoi_mask)
            if want_aoi_area:
                aoi_pixels = int(aoi_mask.sum())
                aoi_area_km2 = aoi_pixels * px_area_km2
        except Exception:
            pass
    water_pixels = int(water_mask.sum())
    area_km2 = water_pixels * px_area_km2
    return float(area_km2), float(px_area_km2), (float(aoi_area_km2) if aoi_area_km2 is not None else None)


def _build_aoi_mask(aoi_wgs84: dict, transform, crs, shape) -> Optional[np.ndarray]:
    """Build a boolean mask True-inside AOI on the target grid. Returns None if fails."""
    try:
        from rasterio.warp import transform_geom
        from rasterio.features import geometry_mask

        aoi_proj = transform_geom("EPSG:4326", crs, aoi_wgs84, precision=6)
        mask = geometry_mask([aoi_proj], transform=transform, invert=True, out_shape=shape)
        return mask
    except Exception:
        return None


def _reproject_dem_to_target(dem_path: Path, dst_transform, dst_crs, dst_shape, *, src_crs_override: Optional[str] = None) -> np.ndarray:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS

    with rasterio.open(str(dem_path)) as src:
        out = np.empty(dst_shape, dtype="float32")
        src_crs = src.crs
        if src_crs is None:
            if src_crs_override:
                try:
                    src_crs = CRS.from_string(src_crs_override)
                except Exception:
                    src_crs = dst_crs
            else:
                src_crs = dst_crs
        band = src.read(1)
        src_nodata = src.nodata
        # Ensure destination initialized to NaN so uncovered areas stay NaN
        out.fill(np.nan)
        reproject(
            source=band,
            destination=out,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )
    return out


def _px_area_m2_from_transform(transform, crs, bounds) -> float:
    # reuse km2, convert to m2
    km2 = _pixel_area_km2(transform, crs, bounds)
    return km2 * 1_000_000.0


def _area_at_level(bottom: np.ndarray, level: float, px_area_m2: float, roi_mask: Optional[np.ndarray]) -> float:
    # Treat NaNs as invalid (dry)
    valid = np.isfinite(bottom)
    wet = np.logical_and(bottom < level, valid)
    if roi_mask is not None:
        wet = np.logical_and(wet, roi_mask)
    return float(wet.sum()) * px_area_m2 / 1_000_000.0  # km^2


def _volume_at_level(bottom: np.ndarray, level: float, px_area_m2: float, roi_mask: Optional[np.ndarray]) -> float:
    # depth valid only where bottom is finite
    valid = np.isfinite(bottom)
    depth = np.where(valid, level - bottom, 0.0)
    wet = depth > 0
    if roi_mask is not None:
        wet = np.logical_and(wet, roi_mask)
    vol_m3 = float(np.where(wet, depth, 0.0).sum() * px_area_m2)
    return vol_m3 / 1_000_000.0  # MCM


def _solve_level_for_area(bottom: np.ndarray, target_area_km2: float, px_area_m2: float, roi_mask: Optional[np.ndarray]) -> float:
    # Bracket: use min/max of bottom within ROI (finite only), add margin
    if roi_mask is not None:
        valid = bottom[roi_mask]
    else:
        valid = bottom
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("nan")
    vmin = float(np.nanmin(valid))
    vmax = float(np.nanmax(valid))
    lo = vmin
    hi = vmax + 50.0  # margin meters
    # Binary search
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        a_mid = _area_at_level(bottom, mid, px_area_m2, roi_mask)
        if a_mid < target_area_km2:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _find_integrated_dem(data_root: Path) -> Optional[Path]:
    """Try to locate an integrated bathymetry DEM produced by the DEM processor.
    Returns the path if found, else None.
    """
    candidates = [
        data_root / "processed_data" / "processing_output" / "integrated_bathymetry_copernicus.tif",
        data_root / "processed_data" / "processing_output" / "bathymetry_reprojected_epsg4326.tif",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _parse_kml_polygon(kml_path: Path) -> Optional[dict]:
    """Parse first Polygon coordinates from a KML file, return GeoJSON-like WGS84 geometry dict."""
    try:
        tree = ET.parse(str(kml_path))
        root = tree.getroot()
        ns = {"kml": "http://www.opengis.net/kml/2.2", "gx": "http://www.google.com/kml/ext/2.2"}
        # Find first coordinates under any Polygon/outerBoundaryIs/LinearRing/coordinates
        coords_el = None
        for coords in root.findall(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns):
            coords_el = coords
            break
        if coords_el is None:
            return None
        text = coords_el.text or ""
        parts = [p.strip() for p in text.replace("\n", " ").split() if p.strip()]
        ring = []
        for p in parts:
            # KML order: lon,lat[,alt]
            toks = p.split(",")
            if len(toks) >= 2:
                lon = float(toks[0]); lat = float(toks[1])
                ring.append((lon, lat))
        if len(ring) < 3:
            return None
        # Ensure closure
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        geom = {"type": "Polygon", "coordinates": [ring]}
        return geom
    except Exception:
        return None


def load_curve(curve_path: Path):
    from scipy.interpolate import interp1d
    df = pd.read_csv(curve_path)
    if not {"area_km2", "volume_mcm"}.issubset(df.columns):
        raise ValueError(f"Curve file missing required columns: {curve_path}")
    df = df.sort_values("area_km2").reset_index(drop=True)
    areas = df["area_km2"].values
    vols = df["volume_mcm"].values
    f = interp1d(
        areas,
        vols,
        kind="linear",
        bounds_error=False,
        fill_value=(float(vols[0]), float(vols[-1])),
    )
    return f, float(areas.min()), float(areas.max()), float(vols.min()), float(vols.max())


def resolve_default_roots(data_root: Path) -> List[Path]:
    return [
        data_root / "Santinel",
        data_root / "raw_data" / "Santinel",
        data_root,
    ]


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Ingest Sentinel-2 area via NDWI and convert to volume")
    ap.add_argument("--root", help="Search root for Sentinel-2 files (recursive)")
    # Prefer reconstructed curve if available
    default_curve_reco = Path("processed_data/processing_output/area_volume_curve_reconstructed.csv")
    default_curve_base = Path("processed_data/processing_output/area_volume_curve.csv")
    default_curve = default_curve_reco if default_curve_reco.exists() else default_curve_base
    ap.add_argument("--curve", default=str(default_curve), help="Path to area_volume_curve.csv (reconstructed preferred)")
    ap.add_argument("--out", default=str(Path("processed_data/water_balance_output/sentinel_area_volume.csv")), help="Output CSV path")
    ap.add_argument("--ndwi-threshold", type=float, default=0.275, help="NDWI threshold for water mask")
    ap.add_argument("--aoi-kml", type=str, default=None, help="Optional KML polygon to clip water area to reservoir polygon")
    ap.add_argument("--year", type=int, default=0, help="Optional year filter, process only dates in this YYYY year")
    ap.add_argument("--volume-mode", choices=["curve", "dem", "auto"], default="dem", help="How to derive volume: 'dem' (default), 'curve', or 'auto' (prefer DEM, fallback to curve)")
    ap.add_argument("--bathymetry-dem", type=str, default=None, help="Path to bathymetry DEM raster. If omitted and --volume-mode dem, will try to use processed integrated DEM if available.")
    ap.add_argument("--dem-crs", type=str, default=None, help="Optional CRS string for DEM if missing (e.g. 'EPSG:32642')")
    ap.add_argument("--max", type=int, default=0, help="Optional cap on number of dates to process (0 = all)")
    ap.add_argument("--min-area-km2", type=float, default=40.0, help="Skip dates with computed water area below this threshold (km^2)")
    ap.add_argument("--allow-extrapolation", action="store_true", help="Allow volume extrapolation/clipping outside curve area range. If not set, such dates are skipped.")
    args = ap.parse_args(argv)

    data_root = Path(os.environ.get("DATA_ROOT", ".")).resolve()
    curve_path = Path(args.curve).resolve()

    # Load curve (always available; used for 'curve' mode)
    vol_fn, area_min, area_max, vol_min, vol_max = load_curve(curve_path)

    # Resolve search roots
    search_roots: List[Path] = []
    if args.root:
        search_roots = [Path(args.root).resolve()]
    else:
        for cand in resolve_default_roots(data_root):
            if cand.exists():
                search_roots.append(cand)
    if not search_roots:
        raise SystemExit("No valid search roots found for Sentinel-2 files. Provide --root.")

    # Find triplets from first root that yields results
    triplets: List[Triplet] = []
    chosen_root: Optional[Path] = None
    for root in search_roots:
        t = find_triplets(root)
        if t:
            triplets = t
            chosen_root = root
            break
    # Fallback: try legacy finder
    if not triplets:
        for root in search_roots:
            t = find_triplets_fallback_legacy(root)
            if t:
                triplets = t
                chosen_root = root
                break
    if not triplets:
        raise SystemExit("No Sentinel-2 triplets (B03/B08/SCL) found under provided roots.")

    print(f"[Sentinel] Root: {chosen_root}")
    aoi_geom = None
    if args.aoi_kml:
        aoi_geom = _parse_kml_polygon(Path(args.aoi_kml))
        if aoi_geom is None:
            print(f"[WARN] Failed to parse AOI from KML: {args.aoi_kml}")
        else:
            print(f"[Sentinel] AOI KML loaded: {args.aoi_kml}")
    # Prepare DEM if dem mode
    dem_array = None
    dem_px_area_m2 = None
    aoi_mask = None
    dem_ready = False
    # Prepare DEM when requested or when in auto mode (prefer DEM)
    if args.volume_mode in ("dem", "auto"):
        dem_path: Optional[Path]
        if args.bathymetry_dem:
            dem_path = Path(args.bathymetry_dem)
        else:
            dem_path = _find_integrated_dem(data_root)
            if dem_path:
                print(f"[Sentinel] Using integrated DEM: {dem_path}")
        if not dem_path or not dem_path.exists():
            if args.volume_mode == "dem":
                raise SystemExit("--volume-mode dem requires a DEM. Provide --bathymetry-dem or generate an integrated DEM under processed_data/processing_output.")
            else:
                print("[Sentinel] DEM not found; 'auto' mode will fallback to curve.")
                dem_path = None
        # Reproject DEM to the grid of the first available B03 to lock grid
        if not triplets:
            raise SystemExit("No triplets to derive target grid for DEM")
        import rasterio
        with rasterio.open(triplets[0].b03) as gsrc:
            target_shape = gsrc.shape
            dst_transform = gsrc.transform
            dst_crs = gsrc.crs
            dem_px_area_m2 = _px_area_m2_from_transform(dst_transform, dst_crs, gsrc.bounds)
        if dem_path is not None:
            dem_array = _reproject_dem_to_target(
                dem_path, dst_transform, dst_crs, target_shape, src_crs_override=args.dem_crs
            )
        if aoi_geom is not None:
            aoi_mask = _build_aoi_mask(aoi_geom, dst_transform, dst_crs, target_shape)
        # Diagnostics: count valid DEM pixels inside AOI (or overall)
        if dem_array is not None:
            valid_dem = np.isfinite(dem_array)
            if aoi_mask is not None:
                valid_count = int(np.logical_and(valid_dem, aoi_mask).sum())
                print(f"[DEM] Valid pixels within AOI: {valid_count}")
                if valid_count == 0:
                    print("[WARN] DEM has no valid pixels within AOI on the Sentinel grid; volumes may be zero. Check DEM coverage/CRS.")
            else:
                print(f"[DEM] Valid pixels on Sentinel grid: {int(valid_dem.sum())}")
            dem_ready = True
        else:
            dem_ready = False
    print(f"[Sentinel] Dates with complete triplets: {len(triplets)} (from {triplets[0].date} to {triplets[-1].date})")
    # Year filter
    if args.year and args.year > 0:
        y = f"{args.year:04d}-"
        triplets = [t for t in triplets if t.date.startswith(y)]
        if not triplets:
            raise SystemExit(f"No triplets found for year {args.year}")
        print(f"[Sentinel] Year filter: {args.year} → {len(triplets)} dates")
    if args.max and args.max > 0:
        triplets = triplets[: args.max]
        print(f"[Sentinel] Processing capped to first {len(triplets)} dates via --max")

    # Process dates sequentially
    rows: List[Dict[str, object]] = []
    px_area_cache: Optional[float] = None
    aoi_area_est_km2: Optional[float] = None
    skipped_small_area = 0
    skipped_curve_range = 0
    for i, t in enumerate(triplets, 1):
        try:
            want_aoi_area = aoi_area_est_km2 is None and aoi_geom is not None
            area_km2, px_area, aoi_area = compute_area_km2(
                t.b03, t.b08, t.scl, args.ndwi_threshold, aoi_wgs84=aoi_geom, want_aoi_area=want_aoi_area
            )
            if px_area_cache is None:
                px_area_cache = px_area
            if aoi_area is not None:
                aoi_area_est_km2 = aoi_area
            # Enforce minimum area filter
            if area_km2 < float(args.min_area_km2):
                skipped_small_area += 1
                print(f" [SKIP] {t.date}: area {area_km2:.2f} km^2 < min-area {args.min_area_km2:.1f}")
                continue
            volume_mcm: Optional[float] = None
            volume_source = ""
            # DEM preferred in 'dem' or 'auto' (when available)
            if args.volume_mode in ("dem", "auto") and dem_ready and dem_array is not None and dem_px_area_m2 is not None:
                level = _solve_level_for_area(dem_array, area_km2, dem_px_area_m2, aoi_mask)
                volume_mcm = _volume_at_level(dem_array, level, dem_px_area_m2, aoi_mask)
                volume_source = "DEM"
                # If DEM produced NaN/0 suspiciously and we're in auto, try curve as fallback within range
                if args.volume_mode == "auto" and (not np.isfinite(volume_mcm) or volume_mcm <= 0):
                    volume_mcm = None
            # Fallback to curve if DEM not used or produced invalid
            if volume_mcm is None:
                if not args.allow_extrapolation and (area_km2 < area_min or area_km2 > area_max):
                    skipped_curve_range += 1
                    print(f" [SKIP] {t.date}: area {area_km2:.2f} km^2 outside curve range [{area_min:.2f}, {area_max:.2f}] km^2")
                    continue
                volume_mcm = float(vol_fn(area_km2))
                volume_source = "curve"
            rows.append({
                "date": pd.to_datetime(t.date),
                "area_km2": area_km2,
                "volume_mcm": volume_mcm,
                "source": "Sentinel-2/NDWI",
                "volume_source": volume_source,
            })
            print(f" {i:4d}/{len(triplets)} {t.date} → area {area_km2:.2f} km^2, volume {volume_mcm:.1f} MCM [{volume_source}]")
        except Exception as e:
            print(f" [WARN] {t.date}: failed -> {e}")

    if not rows:
        raise SystemExit("No rows produced (all failed)")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write CSV with all columns in consistent order
    cols_order = ['date', 'area_km2', 'volume_mcm', 'volume_source', 'source']
    cols_present = [c for c in cols_order if c in df.columns]
    df[cols_present].to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(df)} rows → {out_path}")

    # Validation summary
    ref_area_km2 = 93.7
    ref_vol_mcm = 791.0
    ref_useful_mcm = 562.0
    ref_depth_m = 19.8

    max_area = float(df["area_km2"].max())
    max_vol = float(df["volume_mcm"].max())
    # Implied mean depth at observed max state (rough): V/A (m)
    implied_depth_m = (max_vol * 1e6) / (max_area * 1e6) if max_area > 0 else np.nan

    def pct_diff(obs: float, ref: float) -> float:
        return (obs - ref) / ref * 100.0 if ref != 0 else np.nan

    lines = []
    lines.append("SENTINEL AREA/VOLUME VALIDATION\n")
    lines.append(f"Triplets processed (after filters): {len(df)}\n")
    lines.append(f"Observed period: {df['date'].min().date()} .. {df['date'].max().date()}\n")
    if skipped_small_area or skipped_curve_range:
        lines.append("Skipped dates summary:\n")
        if skipped_small_area:
            lines.append(f" - Small area < {args.min_area_km2:.1f} km^2: {skipped_small_area}\n")
        if skipped_curve_range:
            lines.append(f" - Outside curve range [{area_min:.2f}, {area_max:.2f}] km^2: {skipped_curve_range}\n")
    if aoi_area_est_km2 is not None:
        lines.append(f"AOI (from KML) area estimate on grid: {aoi_area_est_km2:.2f} km^2\n")
    if args.aoi_kml:
        lines.append(f"AOI source: {args.aoi_kml}\n")
    lines.append("")
    lines.append("Reference (Wikipedia or external):\n")
    lines.append(f" - Area: {ref_area_km2} km^2\n")
    lines.append(f" - Volume (total): {ref_vol_mcm} MCM\n")
    lines.append(f" - Useful volume: {ref_useful_mcm} MCM\n")
    lines.append(f" - Depth (mean or listed): {ref_depth_m} m\n")
    lines.append("")
    lines.append("Observed (from Sentinel-derived timeseries):\n")
    lines.append(f" - Max area: {max_area:.2f} km^2 (Δ% vs ref area: {pct_diff(max_area, ref_area_km2):+.1f}%)\n")
    lines.append(f" - Max volume: {max_vol:.1f} MCM (Δ% vs ref vol: {pct_diff(max_vol, ref_vol_mcm):+.1f}%)\n")
    if max_area > 0:
        lines.append(f" - Implied mean depth at max (V/A): {implied_depth_m:.2f} m (compare to listed {ref_depth_m} m)\n")
    lines.append("")
    lines.append("Notes:\n")
    lines.append(" - Area/volume here reflect available clear-sky Sentinel dates and NDWI thresholding; \n")
    lines.append("   they may not coincide with the exact hydrometric maximum.\n")
    lines.append(" - Implied mean depth = V/A is a rough check; listed depth may refer to mean/max at full supply.\n")

    val_path = out_path.parent / "sentinel_validation.txt"
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Validation report → {val_path}")


if __name__ == "__main__":
    main()
