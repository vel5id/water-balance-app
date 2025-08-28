This folder collects auxiliary scripts not required to run the Streamlit app on a server.

Included (examples from your workspace):
- dem_processor.py — builds DEM + area_volume_curve from bathymetry and Copernicus DEM.
- water_balance_model.py — builds baseline water balance CSV.
- predict_2025_gap.py — forecast filler for missing periods.
- list_small_area_scenes.py — diagnostics for small water areas.
- check_dem_stats.py — quick DEM stats.

Keep only what you want to share with frontend/backend; large raw data should not be versioned.
