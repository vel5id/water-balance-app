import os
import pandas as pd
from water_balance_model import SENTINEL_ROOT_DIR, find_sentinel_files, calculate_water_area

OUT_CSV = os.path.join(os.path.dirname(__file__), 'water_balance_output', 'small_area_scenes.csv')

file_groups = find_sentinel_files(SENTINEL_ROOT_DIR)
rows = []
cache = None
for (date_str, b03, b08, scl) in file_groups:
    area_km2, cache = calculate_water_area(b03, b08, scl, cache)
    if area_km2 is not None and area_km2 < 5.0:
        rows.append({
            'date': date_str,
            'area_km2': float(area_km2),
            'b03_path': b03,
            'b08_path': b08,
            'scl_path': scl,
        })

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f'SAVED {OUT_CSV} with {len(rows)} rows')
