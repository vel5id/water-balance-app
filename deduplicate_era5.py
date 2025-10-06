import sqlite3, argparse, os, pandas as pd

TABLES = [
    ('era5_t2m_c','first_non_null'),
    ('era5_precip_mm','first_non_null'),
    ('era5_evap_mm','first_non_null'),
    ('era5_runoff_mm','first_non_null'),
    ('era5_snow_depth_m','first_non_null'),
]

def pick(series: pd.Series, mode: str):
    if mode == 'first_non_null':
        for v in series:
            if pd.notna(v):
                return v
        return None
    raise ValueError(mode)

def dedupe(db_path: str):
    if not os.path.exists(db_path):
        raise SystemExit(f'DB not found: {db_path}')
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for tbl, mode in TABLES:
        print(f'Processing {tbl} ...')
        df = pd.read_sql(f'SELECT date, value FROM {tbl}', conn)
        before = len(df)
        grouped = df.groupby('date')['value'].apply(lambda s: pick(s, mode)).reset_index()
        after = len(grouped)
        cur.execute(f'DROP TABLE {tbl}')
        cur.execute(f'CREATE TABLE {tbl} (date TEXT PRIMARY KEY, value REAL)')
        grouped.to_sql(tbl, conn, if_exists='append', index=False)
        print(f'  {before}->{after} rows, nulls remaining: {grouped.value.isna().sum()}')
    conn.execute('VACUUM')
    conn.close()
    print('Done.')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='processing_output/era5_daily.sqlite')
    args = ap.parse_args()
    dedupe(args.db)
