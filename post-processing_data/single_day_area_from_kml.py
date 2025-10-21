from __future__ import annotations

"""
Compute Sentinel-2 NDWI water area for a specific date within a provided KML polygon (AOI).

Usage:
  python post-processing_data/single_day_area_from_kml.py \
    --root raw_data/Santinel \
    --kml Untitled\ map.kml \
    --date 2024-04-20 \
    --thresholds 0.10 0.05 0.00

Outputs: prints a small report and writes CSV under processed_data/water_balance_output/single_day_area_<date>.csv
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _ensure_import_path():
    import sys
    here = Path(__file__).resolve()
    project_root = here.parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main(argv: Optional[List[str]] = None) -> None:
    import sys
    _ensure_import_path()
    from wbm.ingest_sentinel import (
        find_triplets,
        find_triplets_fallback_legacy,
        compute_area_km2,
        _parse_kml_polygon,
    )

    ap = argparse.ArgumentParser(description="Single-day NDWI area within AOI KML")
    ap.add_argument("--root", required=True, help="Search root for Sentinel-2 files (recursive)")
    ap.add_argument("--kml", required=True, help="KML polygon file (AOI)")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD (exact or nearest)")
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.10, 0.05, 0.00], help="NDWI thresholds to compute")
    args = ap.parse_args(argv)

    aoi_geom = _parse_kml_polygon(Path(args.kml))
    if aoi_geom is None:
        raise SystemExit(f"Failed to parse AOI from KML: {args.kml}")

    root = Path(args.root).resolve()
    triplets = find_triplets(root)
    if not triplets:
        triplets = find_triplets_fallback_legacy(root)
    if not triplets:
        raise SystemExit(f"No Sentinel-2 triplets found under {root}")

    target = _parse_date(args.date)
    # pick exact match else nearest by absolute delta
    def to_dt(s: str) -> datetime:
        return _parse_date(s)

    chosen = None
    best_delta = None
    for t in triplets:
        dt = to_dt(t.date)
        delta = abs((dt - target).days)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            chosen = t
            if delta == 0:
                break

    if chosen is None:
        raise SystemExit("Internal: could not choose a date")

    print(f"Chosen date: {chosen.date} (requested {args.date}, delta={best_delta} days)")

    rows = []
    aoi_area_km2 = None
    for thr in args.thresholds:
        area_km2, px_area, aoi_area = compute_area_km2(
            chosen.b03, chosen.b08, chosen.scl, thr, aoi_wgs84=aoi_geom, want_aoi_area=(aoi_area_km2 is None)
        )
        if aoi_area is not None:
            aoi_area_km2 = aoi_area
        rows.append({
            "date": pd.to_datetime(chosen.date),
            "ndwi_threshold": float(thr),
            "area_km2": float(area_km2),
            "pixel_area_km2": float(px_area),
            "aoi_area_km2": float(aoi_area_km2) if aoi_area_km2 is not None else None,
        })
        print(f" NDWI>{thr:.3f} → area {area_km2:.2f} km^2")

    # Ensure we write under the project root, not under post-processing_data
    project_root = Path(__file__).resolve().parents[1]
    out_dir = (project_root / "processed_data" / "water_balance_output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"single_day_area_{chosen.date}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Wrote results → {out_csv}")


if __name__ == "__main__":
    main()
