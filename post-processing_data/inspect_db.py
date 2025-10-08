import sqlite3, pandas as pd, json, argparse, os

def inspect(db_path: str):
    if not os.path.exists(db_path):
        raise SystemExit(f"DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    report = {}
    for t in sorted(tables):
        try:
            cur.execute(f'PRAGMA table_info({t})')
            schema = cur.fetchall()
            cur.execute(f'SELECT COUNT(*) FROM {t}')
            total = cur.fetchone()[0]
            cols = [c[1] for c in schema]
            if 'date' in cols:
                cur.execute(f'SELECT COUNT(DISTINCT date) FROM {t}')
                distinct = cur.fetchone()[0]
                cur.execute(f'SELECT MIN(date), MAX(date) FROM {t}')
                mn, mx = cur.fetchone()
                sample = pd.read_sql(f'SELECT * FROM {t} ORDER BY date LIMIT 5', conn)
                dupes = total - distinct
            else:
                distinct = mn = mx = dupes = None
                sample = pd.read_sql(f'SELECT * FROM {t} LIMIT 5', conn)
            report[t] = {
                'total_rows': total,
                'distinct_dates': distinct,
                'duplicate_rows': dupes,
                'min_date': mn,
                'max_date': mx,
                'schema': schema,
                'sample': sample.to_dict(orient='records')
            }
        except Exception as e:  # noqa
            report[t] = {'error': str(e)}
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='processing_output/era5_daily.sqlite')
    args = ap.parse_args()
    inspect(args.db)
