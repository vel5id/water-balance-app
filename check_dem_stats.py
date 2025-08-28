import rasterio
import numpy as np
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
DEM_PATH = ROOT / "processing_output" / "bathymetry_reprojected_epsg4326.tif"
OUT_DIR = ROOT / "processing_output"
OUT_TXT = OUT_DIR / "dem_stats.txt"
OUT_JSON = OUT_DIR / "dem_stats.json"

if not DEM_PATH.exists():
    msg = f"DEM not found: {DEM_PATH}"
    print(msg)
    try:
        OUT_TXT.write_text(msg, encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(1)

with rasterio.open(DEM_PATH) as ds:
    arr = ds.read(1).astype("float32")
    nodata = ds.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    vmean = float(np.nanmean(arr))
    info = {
        "path": str(DEM_PATH),
        "crs": str(ds.crs),
        "shape": [int(ds.height), int(ds.width)],
        "bounds": [float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)],
        "nodata": None if nodata is None else float(nodata),
        "min": vmin,
        "max": vmax,
        "mean": vmean,
    }
    # Print to console
    print(json.dumps(info, indent=2))
    # Persist to files for retrieval
    try:
        OUT_TXT.write_text(
            "\n".join([
                f"CRS: {info['crs']}",
                f"Shape: {info['shape'][0]} x {info['shape'][1]}",
                f"Bounds: {tuple(info['bounds'])}",
                f"NoData: {info['nodata']}",
                f"Min: {info['min']}",
                f"Max: {info['max']}",
                f"Mean: {info['mean']}",
            ]),
            encoding="utf-8",
        )
    except Exception:
        pass
    try:
        OUT_JSON.write_text(json.dumps(info, indent=2), encoding="utf-8")
    except Exception:
        pass
