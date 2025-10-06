"""Incremental ERA5 hourly -> daily CSV exporter.

Создаёт (или дополняет) 5 файлов в выходной директории:
  era5_t2m_c.csv
  era5_precip_mm.csv
  era5_evap_mm.csv
  era5_runoff_mm.csv
  era5_snow_depth_m.csv

Формат каждого CSV: date,value (без заголовков индекса). При повторном запуске даты UPSERT-ятся: мы читаем существующий CSV (если есть),
обновляем / добавляем значения и сохраняем обратно.

Использование:
  python era5_process_csv.py --input raw_nc --output-dir processing_output/era5_csv [--lat-range 55 56 --lon-range 37 38]
"""
from __future__ import annotations
import argparse, os, zipfile, tempfile, sqlite3
from pathlib import Path
from typing import Dict, Iterable, Optional
import pandas as pd
import xarray as xr

VAR_ALIASES: Dict[str, list[str]] = {
    "t2m": ["t2m", "2m_temperature", "temperature_2m"],
    "runoff": ["ro", "runoff"],
    "snow_depth": ["sde", "sd", "snow_depth"],
    "evap": ["e", "evaporation", "total_evaporation"],
    "precip": ["tp", "precip", "total_precipitation", "precipitation"],
}

TABLE_SPECS = {
    't2m': ('era5_t2m_c', 'mean', 'K_TO_C'),
    'tp': ('era5_precip_mm', 'sum', 'M_TO_MM'),
    'e': ('era5_evap_mm', 'sum', 'M_TO_MM_NEG'),
    'ro': ('era5_runoff_mm', 'sum', 'M_TO_MM'),
    'sde': ('era5_snow_depth_m', 'mean', 'IDENT'),
}

CSV_FILES = {spec[0]: spec[0] + '.csv' for spec in TABLE_SPECS.values()}


def find_var(ds: xr.Dataset, logical_name: str) -> Optional[str]:
    for cand in VAR_ALIASES[logical_name]:
        if cand in ds.variables:
            return cand
    return None


def iter_files(input_dir: str):
    for p in sorted(Path(input_dir).glob('*.nc')):
        yield p


def open_inner_dataset(path: Path) -> xr.Dataset:
    with open(path, 'rb') as fh:
        sig = fh.read(4)
    candidate = path
    if sig.startswith(b'PK'):
        with zipfile.ZipFile(path,'r') as zf:
            nc_members = [n for n in zf.namelist() if n.lower().endswith('.nc')]
            if not nc_members:
                raise RuntimeError(f'No nc inside {path}')
            inner = nc_members[0]
            tmpdir = tempfile.mkdtemp(prefix='era5_zip_')
            out_nc = Path(tmpdir)/Path(inner).name
            with zf.open(inner) as src, open(out_nc,'wb') as dst:
                dst.write(src.read())
            candidate = out_nc
    last_err=None
    for eng in ['h5netcdf','netcdf4','scipy']:
        try:
            return xr.open_dataset(candidate, engine=eng)
        except Exception as e:
            last_err=e
    raise RuntimeError(f'Failed open {path}: {last_err}')


def spatial_subset(ds: xr.Dataset, lat_range, lon_range) -> xr.Dataset:
    if lat_range or lon_range:
        lat_name=None
        for cand in ['latitude','lat','Latitude']:
            if cand in ds.coords:
                lat_name=cand; break
        lon_name=None
        for cand in ['longitude','lon','Longitude']:
            if cand in ds.coords:
                lon_name=cand; break
        if lat_name and lon_name:
            lat0,lat1=lat_range
            lon0,lon1=lon_range
            ds = ds.sel({lat_name: slice(min(lat0,lat1), max(lat0,lat1)),
                         lon_name: slice(min(lon0,lon1), max(lon0,lon1))})
    return ds


def reduce_time_dim(da: xr.DataArray, time_name: str) -> xr.DataArray:
    drop_dims=[d for d in da.dims if d not in (time_name,'latitude','longitude','lat','lon')]
    for d in drop_dims:
        da=da.isel({d:0})
    return da


def hourly_to_daily(da: xr.DataArray, time_name: str, how: str) -> pd.Series:
    if time_name!='time':
        da=da.rename({time_name:'time'})
    s=da.to_series()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index=pd.to_datetime(s.index)
    if how=='mean':
        return s.resample('D').mean()
    if how=='sum':
        return s.resample('D').sum()
    raise ValueError(how)


def convert(series: pd.Series, mode: str) -> pd.Series:
    if mode=='K_TO_C':
        return series-273.15
    if mode=='M_TO_MM':
        return series*1000.0
    if mode=='M_TO_MM_NEG':
        return (-series)*1000.0
    if mode=='IDENT':
        return series
    raise ValueError(mode)


def load_existing(csv_path: Path) -> pd.Series:
    if not csv_path.exists():
        return pd.Series(dtype=float)
    df=pd.read_csv(csv_path)
    if df.empty:
        return pd.Series(dtype=float)
    return pd.Series(df['value'].values, index=pd.to_datetime(df['date']))


def save_series(series: pd.Series, path: Path):
    # Ensure datetime index if possible
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            dt_index = pd.to_datetime(series.index, errors='coerce')
            series.index = dt_index
        except Exception:
            pass
    series = series[~series.index.isna()]
    series = series.sort_index()
    if isinstance(series.index, pd.DatetimeIndex):
        date_col = series.index.strftime('%Y-%m-%d')
    else:
        date_col = series.index.astype(str)
    out = pd.DataFrame({'date': date_col, 'value': series.values})
    out.to_csv(path, index=False)


def merge_upsert(existing: pd.Series, new: pd.Series) -> pd.Series:
    if existing.empty:
        return new
    # Normalize indexes to datetime where possible
    if not isinstance(existing.index, pd.DatetimeIndex):
        try:
            existing.index = pd.to_datetime(existing.index, errors='coerce')
        except Exception:
            pass
    if not isinstance(new.index, pd.DatetimeIndex):
        try:
            new.index = pd.to_datetime(new.index, errors='coerce')
        except Exception:
            pass
    combined = existing.copy()
    for dt, val in new.items():
        combined[dt] = val
    return combined


def process_file(path: Path, lat_range, lon_range, accumulators: dict[str,pd.Series]):
    ds=open_inner_dataset(path)
    time_name='time' if 'time' in ds.coords else ('valid_time' if 'valid_time' in ds.coords else None)
    if time_name is None:
        print(f'Skip {path.name}: no time')
        ds.close(); return
    sub=spatial_subset(ds, lat_range, lon_range)
    produced=[]
    for logical, aliases in VAR_ALIASES.items():
        found=find_var(sub, logical)
        if not found:
            continue
        base=found
        spec_key=base
        if spec_key not in TABLE_SPECS:
            for canon in ['t2m','tp','e','ro','sde']:
                if base.startswith(canon) or base==canon:
                    spec_key=canon; break
        if spec_key not in TABLE_SPECS:
            continue
        csv_name, how, conv = TABLE_SPECS[spec_key]
        try:
            da=sub[found]
            da=reduce_time_dim(da, time_name)
            spatial_dims=[d for d in da.dims if d not in (time_name,)]
            if spatial_dims:
                da=da.mean(dim=spatial_dims, skipna=True)
            daily=hourly_to_daily(da, time_name, how)
            daily=convert(daily, conv)
            acc=accumulators.get(csv_name, pd.Series(dtype=float))
            merged=merge_upsert(acc, daily)
            accumulators[csv_name]=merged
            produced.append(csv_name)
        except Exception as e:
            print(f'Failed {path.name} var {found}: {e}')
    ds.close()
    if produced:
        print(f'{path.name}: updated {", ".join(produced)}')
    else:
        print(f'{path.name}: no variables')


def main():
    ap=argparse.ArgumentParser(description='Incremental ERA5 -> daily CSV exporter')
    ap.add_argument('--input', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--lat-range', nargs=2, type=float, default=None)
    ap.add_argument('--lon-range', nargs=2, type=float, default=None)
    ap.add_argument('--limit', type=int, default=None)
    args=ap.parse_args()

    out_dir=Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lat_range=tuple(args.lat_range) if args.lat_range else None
    lon_range=tuple(args.lon_range) if args.lon_range else None

    # Load existing CSVs into accumulators
    accumulators: dict[str,pd.Series]={}
    for csv_stem in CSV_FILES.values():
        stem=csv_stem.replace('.csv','')
        path=out_dir/csv_stem
        accumulators[stem]=load_existing(path)

    files=list(iter_files(args.input))
    if args.limit:
        files=files[:args.limit]
    if not files:
        raise SystemExit('No input files')

    for i,p in enumerate(files,1):
        process_file(p, lat_range, lon_range, accumulators)
        if i % 25 == 0:
            print(f'Progress: {i}/{len(files)} files')

    # Save all accumulators
    for stem, series in accumulators.items():
        save_series(series, out_dir/f'{stem}.csv')
    print(f'Done. Wrote CSVs to {out_dir}')

if __name__=='__main__':
    main()
