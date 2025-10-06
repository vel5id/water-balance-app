"""Process ERA5 total precipitation (tp) files into daily sums.

Outputs in the same folder:
  era5_precip_mm.csv          (date,value in mm/day)
  era5_precip_mm.sqlite (table era5_precip_mm PRIMARY KEY date)

Usage:
  python process_precipitation.py --input raw_nc/total_precipitation
"""
from __future__ import annotations
import argparse, zipfile, tempfile, sqlite3
from pathlib import Path
import pandas as pd
import xarray as xr

ALIASES=["tp","total_precipitation","precip","precipitation"]
TABLE_NAME='era5_precip_mm'
CSV_NAME=TABLE_NAME+'.csv'
DB_NAME=TABLE_NAME+'.sqlite'
AGG_MODE='sum'
CONV_MODE='M_TO_MM'


def open_inner_dataset(path: Path) -> xr.Dataset:
    with open(path,'rb') as fh:
        sig=fh.read(4)
    candidate=path
    if sig.startswith(b'PK'):
        with zipfile.ZipFile(path,'r') as zf:
            nc_members=[n for n in zf.namelist() if n.lower().endswith('.nc')]
            if not nc_members:
                raise RuntimeError(f'Zip without nc: {path}')
            inner=nc_members[0]
            tmpdir=tempfile.mkdtemp(prefix='era5_zip_')
            out_nc=Path(tmpdir)/Path(inner).name
            with zf.open(inner) as src, open(out_nc,'wb') as dst:
                dst.write(src.read())
            candidate=out_nc
    last_err=None
    for eng in ['h5netcdf','netcdf4','scipy']:
        try:
            return xr.open_dataset(candidate, engine=eng)
        except Exception as e:
            last_err=e
    raise RuntimeError(f'Failed open {path}: {last_err}')


def find_var(ds: xr.Dataset):
    for cand in ALIASES:
        if cand in ds.variables:
            return cand
    return None


def spatial_subset(ds: xr.Dataset, lat_range, lon_range):
    if not (lat_range or lon_range):
        return ds
    lat_name=None
    for cand in ['latitude','lat','Latitude']:
        if cand in ds.coords:
            lat_name=cand; break
    lon_name=None
    for cand in ['longitude','lon','Longitude']:
        if cand in ds.coords:
            lon_name=cand; break
    if lat_name and lon_name and lat_range and lon_range:
        lat0,lat1=lat_range; lon0,lon1=lon_range
        ds = ds.sel({lat_name: slice(min(lat0,lat1), max(lat0,lat1)),
                     lon_name: slice(min(lon0,lon1), max(lon0,lon1))})
    return ds


def reduce_time_dim(da: xr.DataArray, time_name: str):
    drop=[d for d in da.dims if d not in (time_name,'latitude','longitude','lat','lon')]
    for d in drop:
        da=da.isel({d:0})
    return da


def convert(series: pd.Series, mode: str):
    if mode=='M_TO_MM':
        return series*1000.0
    return series


def hourly_to_daily(series: pd.Series, how: str):
    if how=='sum':
        return series.resample('D').sum()
    if how=='mean':
        return series.resample('D').mean()
    raise ValueError(how)


def load_existing(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    df=pd.read_csv(path)
    if df.empty:
        return pd.Series(dtype=float)
    return pd.Series(df['value'].values, index=pd.to_datetime(df['date']))


def save_csv(series: pd.Series, path: Path):
    series=series.sort_index()
    df=pd.DataFrame({'date': series.index.strftime('%Y-%m-%d'), 'value': series.values})
    df.to_csv(path, index=False)


def upsert_db(series: pd.Series, db_path: Path):
    conn=sqlite3.connect(db_path)
    cur=conn.cursor()
    cur.execute(f'CREATE TABLE IF NOT EXISTS {TABLE_NAME} (date TEXT PRIMARY KEY, value REAL)')
    rows=[(d.strftime('%Y-%m-%d'), float(v) if pd.notna(v) else None) for d,v in series.items()]
    cur.executemany(f'INSERT OR REPLACE INTO {TABLE_NAME} (date,value) VALUES (?,?)', rows)
    conn.commit(); conn.close()


def process_file(path: Path, lat_range, lon_range):
    ds=open_inner_dataset(path)
    time_name='time' if 'time' in ds.coords else ('valid_time' if 'valid_time' in ds.coords else None)
    if time_name is None:
        ds.close(); return pd.Series(dtype=float)
    v=find_var(ds)
    if not v:
        ds.close(); return pd.Series(dtype=float)
    sub=spatial_subset(ds, lat_range, lon_range)
    da=sub[v]
    da=reduce_time_dim(da, time_name)
    spatial=[d for d in da.dims if d!=time_name]
    if spatial:
        da=da.mean(dim=spatial, skipna=True)
    if time_name!='time':
        da=da.rename({time_name:'time'})
    s=da.to_series()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index=pd.to_datetime(s.index, errors='coerce')
    s=s[~s.index.isna()]
    daily=hourly_to_daily(s, AGG_MODE)
    daily=convert(daily, CONV_MODE)
    ds.close(); return daily


def main():
    ap=argparse.ArgumentParser(description='Process ERA5 total precipitation into daily mm')
    ap.add_argument('--input', required=True)
    ap.add_argument('--lat-range', nargs=2, type=float, default=None)
    ap.add_argument('--lon-range', nargs=2, type=float, default=None)
    ap.add_argument('--limit', type=int, default=None)
    args=ap.parse_args()

    folder=Path(args.input)
    lat_range=tuple(args.lat_range) if args.lat_range else None
    lon_range=tuple(args.lon_range) if args.lon_range else None
    existing=load_existing(folder/CSV_NAME)

    files=sorted(folder.glob('*.nc'))
    if args.limit:
        files=files[:args.limit]
    if not files:
        raise SystemExit('No files')

    for i,p in enumerate(files,1):
        d=process_file(p, lat_range, lon_range)
        if not d.empty:
            for dt,val in d.items():
                existing[dt]=val
            print(f'{p.name}: {len(d)} days')
        else:
            print(f'{p.name}: skipped')
        if i % 25==0:
            print(f'Progress {i}/{len(files)}')

    if existing.empty:
        print('No data aggregated.'); return
    if not isinstance(existing.index, pd.DatetimeIndex):
        existing.index=pd.to_datetime(existing.index, errors='coerce')
    existing=existing[~existing.index.isna()].sort_index()

    save_csv(existing, folder/CSV_NAME)
    upsert_db(existing, folder/DB_NAME)
    print(f'Done. Wrote {CSV_NAME} and {DB_NAME}')

if __name__=='__main__':
    main()
