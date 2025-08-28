# Deployment (Server)

This app is portable. Provide data paths via the DATA_ROOT environment variable.

## Prerequisites
- Python 3.10+
- GDAL/GEOS runtime present (rasterio wheels typically include binaries on Windows/macOS; Linux servers may need system packages).

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```bash
export DATA_ROOT=/srv/wbm-data   # Windows PowerShell: $env:DATA_ROOT='C:\\wbm-data'
streamlit run app.py --server.port 8501 --server.headless true
```

## Required data under $DATA_ROOT
- processing_output/area_volume_curve.csv
- processing_output/integrated_bathymetry_copernicus.tif (preferred) or bathymetry_reprojected_epsg4326.tif
- processing_output/ndwi_mask_0275.tif
- GLEAM/processed/gleam_summary_all_years.csv
- precipitation_timeseries.csv
- water_balance_output/water_balance_final.csv (optional; scenarios work with climatology)

## Notes
- The app will display which DEM is used and show DEM stats on the map.
- Switch “Water mask mode” to “Depth & NDWI” for physically consistent water extent.