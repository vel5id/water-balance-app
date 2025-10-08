"""Incremental ERA5 (hourly) -> daily SQLite processor.

Ключевые отличия рефакторинга:
1. Пофайловая обработка: каждый *.nc (на самом деле zip с вложенным data_0.nc) открывается отдельно.
2. Никакого объединения всех месяцев в памяти – сразу агрегируем в суточные значения и пишем (UPSERT) в таблицы.
3. Поддержка разных имен координаты времени (time | valid_time) и лишних измерений (number, expver) – они сбрасываются (isel(0)).
4. Разные правила агрегации:
   - t2m        (K)  -> суточное среднее, преобразовать в °C
   - tp         (m)  -> суточная сумма, *1000 -> precip_mm
   - e          (m)  -> суточная сумма, *(-1000) -> evap_mm (делаем положительной испарение)
   - ro         (m)  -> суточная сумма, *1000 -> runoff_mm
   - sde        (m)  -> суточное среднее -> snow_depth_m
5. Идемпотентность: повторный запуск перезапишет (UPSERT) те же даты.

Использование:
    python era5_process.py --input raw_nc --output-db processing_output/era5_daily.sqlite \
        --lat-range 55 56 --lon-range 37 38

Создаваемые таблицы (date TEXT PRIMARY KEY, value REAL):
    era5_t2m_c, era5_precip_mm, era5_evap_mm, era5_runoff_mm, era5_snow_depth_m

Примечание: Если в файлах отсутствует переменная – строки не добавляются (таблица останется частично заполненной другими датами или пустой)."""
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Optional
import pandas as pd
import numpy as np
import xarray as xr
import zipfile
import tempfile
import re

VAR_ALIASES: Dict[str, list[str]] = {
    "t2m": ["t2m", "2m_temperature", "temperature_2m"],
    "runoff": ["ro", "runoff"],
    "snow_depth": ["sde", "sd", "snow_depth"],
    "evap": ["e", "evaporation", "total_evaporation"],
    "precip": ["tp", "precip", "total_precipitation", "precipitation"],
}

# Target units:
# t2m: Kelvin -> Celsius (K - 273.15)
# runoff/evap/precip: meters -> mm ( *1000 )
# snow_depth: meters (leave as is)


def find_var(ds: xr.Dataset, logical_name: str) -> Optional[str]:
    for cand in VAR_ALIASES[logical_name]:
        if cand in ds.variables:
            return cand
    return None


def iter_files(input_dir: str) -> Iterable[Path]:
    for p in sorted(Path(input_dir).glob("*.nc")):
        yield p


def open_inner_dataset(path: Path) -> xr.Dataset:
    """Open possibly-zipped ERA5 file and return xarray Dataset.
    Preference order of engines left implicit except we try h5netcdf then netcdf4.
    """
    with open(path, 'rb') as fh:
        sig = fh.read(4)
    candidate = path
    tmpdir = None
    if sig.startswith(b'PK'):
        with zipfile.ZipFile(path, 'r') as zf:
            nc_members = [n for n in zf.namelist() if n.lower().endswith('.nc')]
            if not nc_members:
                raise RuntimeError(f"Zip container {path} has no .nc members")
            inner = nc_members[0]
            tmpdir = tempfile.mkdtemp(prefix='era5_zip_')
            out_nc = Path(tmpdir) / Path(inner).name
            with zf.open(inner) as src, open(out_nc, 'wb') as dst:
                dst.write(src.read())
            candidate = out_nc
    # Try engines
    last_err = None
    for eng in ["h5netcdf", "netcdf4", "scipy"]:
        try:
            ds = xr.open_dataset(candidate, engine=eng)
            return ds
        except Exception as e:  # noqa
            last_err = e
            continue
    raise RuntimeError(f"Failed to open {path}: {last_err}")


def spatial_subset(ds: xr.Dataset, lat_range, lon_range) -> xr.Dataset:
    if lat_range or lon_range:
        lat_name = None
        for cand in ["latitude", "lat", "Latitude"]:
            if cand in ds.coords:
                lat_name = cand
                break
        lon_name = None
        for cand in ["longitude", "lon", "Longitude"]:
            if cand in ds.coords:
                lon_name = cand
                break
        if lat_name and lon_name:
            lat0, lat1 = lat_range
            lon0, lon1 = lon_range
            sub = ds.sel({lat_name: slice(min(lat0, lat1), max(lat0, lat1)),
                          lon_name: slice(min(lon0, lon1), max(lon0, lon1))})
            return sub
    return ds


def reduce_time_dim(da: xr.DataArray, time_name: str) -> xr.DataArray:
    # remove ensemble / expver dims by taking first index
    drop_dims = [d for d in da.dims if d not in (time_name, 'latitude', 'longitude', 'lat', 'lon')]
    for d in drop_dims:
        da = da.isel({d: 0})
    return da


def aggregate_hourly_to_daily(da: xr.DataArray, time_name: str, how: str) -> pd.Series:
    da = da.rename({time_name: 'time'}) if time_name != 'time' else da
    s = da.to_series()
    # ensure DateTimeIndex
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    if how == 'mean':
        return s.resample('D').mean()
    elif how == 'sum':
        return s.resample('D').sum()
    else:
        raise ValueError(how)


TABLE_SPECS = {
    't2m': ('era5_t2m_c', 'mean', 'K_TO_C'),
    'tp': ('era5_precip_mm', 'sum', 'M_TO_MM'),
    'e': ('era5_evap_mm', 'sum', 'M_TO_MM_NEG'),
    'ro': ('era5_runoff_mm', 'sum', 'M_TO_MM'),
    'sde': ('era5_snow_depth_m', 'mean', 'IDENT'),
}


def convert_units(values: pd.Series, mode: str) -> pd.Series:
    if mode == 'K_TO_C':
        return values - 273.15
    if mode == 'M_TO_MM':
        return values * 1000.0
    if mode == 'M_TO_MM_NEG':  # evaporation: stored отрицательное накопление
        return (-values) * 1000.0
    if mode == 'IDENT':
        return values
    raise ValueError(mode)


def ensure_tables(conn: sqlite3.Connection):
    conn.execute("CREATE TABLE IF NOT EXISTS era5_meta (key TEXT PRIMARY KEY, value TEXT)")
    for tbl in ['era5_t2m_c','era5_precip_mm','era5_evap_mm','era5_runoff_mm','era5_snow_depth_m']:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {tbl} (date TEXT PRIMARY KEY, value REAL)")
    conn.execute("INSERT OR REPLACE INTO era5_meta(key,value) VALUES (?,?)", ("schema_version", "2"))


def upsert_series(conn: sqlite3.Connection, table: str, series: pd.Series):
    cur = conn.cursor()
    for dt, val in series.items():
        cur.execute(f"INSERT OR REPLACE INTO {table}(date,value) VALUES (?,?)", (dt.strftime('%Y-%m-%d'), None if pd.isna(val) else float(val)))
    conn.commit()


def process_file(path: Path, conn: sqlite3.Connection, lat_range, lon_range):
    ds = open_inner_dataset(path)
    # detect time coord
    time_name = 'time' if 'time' in ds.coords else ('valid_time' if 'valid_time' in ds.coords else None)
    if time_name is None:
        print(f"Skip {path.name}: no time coordinate")
        ds.close()
        return
    sub = spatial_subset(ds, lat_range, lon_range)
    produced = []
    for logical, aliases in VAR_ALIASES.items():
        found = find_var(sub, logical)
        if not found:
            continue
        base_name = found
        spec_key = base_name
        # Map base var to our TABLE_SPECS key (strip possible variants)
        if base_name not in TABLE_SPECS:
            # heuristic mapping using alias groups
            for canon in ['t2m','tp','e','ro','sde']:
                if base_name.startswith(canon) or base_name == canon:
                    spec_key = canon
                    break
        if spec_key not in TABLE_SPECS:
            continue
        tbl, how, unit_mode = TABLE_SPECS[spec_key]
        try:
            da = sub[found]
            da = reduce_time_dim(da, time_name)
            # collapse spatial dims -> mean
            spatial_dims = [d for d in da.dims if d not in (time_name,)]
            if spatial_dims:
                da = da.mean(dim=spatial_dims, skipna=True)
            daily = aggregate_hourly_to_daily(da, time_name, how)
            daily = convert_units(daily, unit_mode)
            upsert_series(conn, tbl, daily)
            produced.append(tbl)
        except Exception as e:  # noqa
            print(f"Failed {path.name} var {found}: {e}")
    ds.close()
    if produced:
        print(f"{path.name}: updated {', '.join(produced)}")
    else:
        print(f"{path.name}: no target variables found")


def main():
    ap = argparse.ArgumentParser(description="Incrementally aggregate ERA5 hourly (zipped) into daily SQLite")
    ap.add_argument("--input", required=True, help="Directory with monthly .nc (zip) files")
    ap.add_argument("--output-db", required=True, help="SQLite DB path")
    ap.add_argument("--lat-range", nargs=2, type=float, default=None, help="Latitude min max")
    ap.add_argument("--lon-range", nargs=2, type=float, default=None, help="Longitude min max")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N files (debug)")
    args = ap.parse_args()

    lat_range = tuple(args.lat_range) if args.lat_range else None
    lon_range = tuple(args.lon_range) if args.lon_range else None
    out_db = args.output_db
    os.makedirs(os.path.dirname(out_db), exist_ok=True)

    files = list(iter_files(args.input))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit("No input files")

    with sqlite3.connect(out_db) as conn:
        ensure_tables(conn)
        for i, p in enumerate(files, 1):
            process_file(p, conn, lat_range, lon_range)
            if i % 25 == 0:
                print(f"Progress: {i}/{len(files)} files")
        # vacuum optional
        try:
            conn.execute('VACUUM')
        except Exception:
            pass
    print(f"Done. Processed {len(files)} files -> {out_db}")


if __name__ == "__main__":
    main()
