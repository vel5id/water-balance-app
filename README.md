# Water Balance Model (Streamlit)

Interactive reservoir water-balance model using Sentinel-2 surface area, an area–volume (and elevation) curve, GLEAM (evaporation), and IMERG (precipitation). Includes a dynamic map overlay using bathymetry/elevation.

## Contents
- `app.py` — Streamlit interactive app.
- `wbm/` — utilities (curve, data, simulate, plots, analysis).
- `dem_processor.py` — builds DEM artifacts and the area–volume curve from bathymetry + Copernicus DEM.
- `water_balance_model.py` — builds baseline daily balance from external drivers (optional).
- `requirements.txt` — runtime dependencies.
- `DEPLOYMENT.md` — server deployment notes.

## Prerequisites
- Python 3.10+
- pip
- For Linux servers: rasterio may require GDAL runtime (see Troubleshooting below).

## Data layout (DATA_ROOT)
The app reads inputs under a base directory, configurable via the `DATA_ROOT` environment variable. Default is the current repo folder path used in development: `C:\\Users\\vladi\\Downloads\\Data`.

Required files/folders under `%DATA_ROOT%`:

```
processing_output/
	area_volume_curve.csv
	integrated_bathymetry_copernicus.tif   # preferred DEM for the app map
	ndwi_mask_0275.tif                      # NDWI water mask (1=water)

GLEAM/processed/
	gleam_summary_all_years.csv             # evaporation_mm column (or E)

precipitation_timeseries.csv              # precipitation_mm (or mean_precip_mm_per_h*24)

water_balance_output/
	water_balance_final.csv                 # optional baseline (app can run without it)
```

Tip: If `integrated_bathymetry_copernicus.tif` is missing, the app uses `bathymetry_reprojected_epsg4326.tif` if present.

## Install

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS (bash):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure paths

Set `DATA_ROOT` to point to your data directory.

Windows PowerShell:

```powershell
$env:DATA_ROOT = 'C:\\wbm-data'
```

Linux/macOS (bash):

```bash
export DATA_ROOT=/srv/wbm-data
```

If not set, the app will default to the development path used in this repo.

## Prepare artifacts (optional but recommended)

If you don’t yet have `processing_output/area_volume_curve.csv` and the integrated DEM, build them locally using `dem_processor.py`. Update the constants at the top if your input paths differ (e.g., Copernicus DEM path).

Windows PowerShell:

```powershell
python dem_processor.py
```

This will create:
- `processing_output/integrated_bathymetry_copernicus.tif`
- `processing_output/area_volume_curve.csv`

To build a baseline daily balance (optional), prepare GLEAM/IMERG and run:

```powershell
python water_balance_model.py
```

This will create `water_balance_output/water_balance_final.csv` used as the baseline line on the chart.

## Run the Streamlit app

Windows PowerShell:

```powershell
streamlit run app.py --server.port 8501 --server.headless true
```

Linux/macOS (bash):

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

Open the Local URL in your browser.

## Using the app
- Sidebar controls set P/ET scaling, inflow/outflow, start date, and horizon.
- Forecast drivers:
	- Monthly mean (all years)
	- Seasonal climatology (DOY mean)
	- Seasonal + trend (full-history linear trend + seasonality)
- Map:
	- DEM view: grayscale or hillshade.
	- Water mask mode:
		- Depth & NDWI: water where DEM depth < 0 AND NDWI=water.
		- Simulated level: threshold using volume→elevation (requires elevation DEM).
- Download CSV of the simulated scenario.

## Automation and server notes (for ML/AI ops style deployments)
- Configure `DATA_ROOT` via environment variable to swap datasets per environment.
- Run headless on a fixed port; frontends can proxy to the Streamlit server.
- For CI or containerized testing, prepare a minimal data bundle under `/data` and set `DATA_ROOT=/data`.
- Health-check: poll the `/` root page and look for the app title.

### Optional: Docker (example)
Create a simple `Dockerfile` (Linux base shown; ensure GDAL is available):

```
FROM python:3.10-slim
RUN apt-get update && apt-get install -y gdal-bin libgdal-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV DATA_ROOT=/data
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]
```

Build and run:

```bash
docker build -t wbm-app .
docker run --rm -p 8501:8501 -e DATA_ROOT=/data -v /host/wbm-data:/data wbm-app
```

## Troubleshooting
- Missing files: the app will show an error if `area_volume_curve.csv` is absent. Run `dem_processor.py` first.
- Precipitation volumes are zero: IMERG may be missing or in different units; switch to “Monthly mean (all years)” or ensure `precipitation_mm` exists (or `mean_precip_mm_per_h` × 24).
- RasterIO/GDAL errors on Linux: install GDAL runtime (`apt-get install gdal-bin libgdal-dev`) or use a base image with GDAL.
- Map water extent looks inverted: switch DEM view or ensure you’re in “Depth & NDWI” mode (the integrated DEM stores depths as negative values inside water).

## Backtesting / Validation

Для проверки прогностической состоятельности модели добавлен документ `URGENT_BACKTEST_2024.md`.

Содержит:
- Процедуру построения прогноза 2024 года по данным 2022-2023
- Метрики оценки (MAE, RMSE, Bias, MAPE, NSE)
- Чеклист и критерии успешности
- Рекомендации по визуализации и отчёту

Быстрый старт:
1. Подготовьте очищенный набор 2022-01-01 .. 2023-12-31
2. Запустите deterministic прогноз осадков и испарения на 2024 через `build_robust_season_trend_series`
3. Прогоните модель объёма, сравните с фактами 2024
4. Сведите метрики и графики в `backtest_2024_report.md`

(Опционально) Создайте скрипт автоматизации `backtest_2024.py` как описано в документе.

## License
Proprietary/internal use. Update as appropriate for your organization.
