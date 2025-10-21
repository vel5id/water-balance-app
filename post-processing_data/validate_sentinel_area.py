from __future__ import annotations

"""
Validate Sentinel-derived areas against target 93.7 km^2 by sweeping NDWI thresholds.

Outputs:
- processed_data/water_balance_output/sentinel_area_validation_thresholds.txt
- processed_data/water_balance_output/sentinel_area_threshold_sweep.csv (optional detailed rows)
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _ensure_import_path():
    """Ensure project root (containing 'wbm') is on sys.path."""
    import sys
    here = Path(__file__).resolve()
    project_root = here.parents[1]  # .../Data
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))


def _sweep(root: Path, thresholds: List[float], limit: int | None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_import_path()
    from wbm.ingest_sentinel import find_triplets, find_triplets_fallback_legacy, compute_area_km2

    # find triplets
    triplets = find_triplets(root)
    if not triplets:
        triplets = find_triplets_fallback_legacy(root)
    if not triplets:
        raise SystemExit(f"No Sentinel triplets found under {root}")

    if limit is not None and limit > 0:
        triplets = triplets[:limit]

    rows = []
    for thr in thresholds:
        for t in triplets:
            try:
                area_km2, _ = compute_area_km2(t.b03, t.b08, t.scl, thr)
            except Exception as e:
                # skip failures in sweep quietly
                continue
            rows.append({
                "date": pd.to_datetime(t.date),
                "ndwi_threshold": thr,
                "area_km2": float(area_km2),
            })
    if not rows:
        raise SystemExit("No areas computed in sweep")
    df = pd.DataFrame(rows)
    best = df.groupby("ndwi_threshold").agg(
        max_area_km2=("area_km2", "max"),
        mean_top5_km2=("area_km2", lambda s: float(np.mean(sorted(s.values, reverse=True)[:5])))
    ).reset_index().sort_values("ndwi_threshold")
    return df, best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Sweep NDWI thresholds and validate observed areas against 93.7 km^2")
    ap.add_argument("--root", required=True, help="Root directory containing Sentinel triplets (e.g., raw_data/Santinel)")
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.275, 0.30, 0.35], help="NDWI thresholds to test")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on number of dates (0 = all)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    limit = args.limit if args.limit and args.limit > 0 else None
    detailed, summary = _sweep(root, list(args.thresholds), limit)

    out_dir = Path("processed_data/water_balance_output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = out_dir / "sentinel_area_threshold_sweep.csv"
    summary_path = out_dir / "sentinel_area_validation_thresholds.txt"
    detailed.sort_values(["ndwi_threshold", "date"]).to_csv(detailed_path, index=False)

    lines = []
    lines.append("SENTINEL AREA VALIDATION — NDWI THRESHOLD SWEEP\n")
    lines.append(f"Roots: {root}\n")
    lines.append(f"Thresholds tested: {', '.join(str(x) for x in args.thresholds)}\n\n")
    lines.append("Results by threshold:\n")
    for _, row in summary.iterrows():
        lines.append(f" - thr={row['ndwi_threshold']:.3f}: max_area={row['max_area_km2']:.2f} km^2; mean_top5={row['mean_top5_km2']:.2f} km^2\n")
    lines.append("\nReference area: 93.7 km^2; acceptance check: any max >= 88 km^2?\n")
    ok = (summary["max_area_km2"] >= 88.0).any()
    lines.append(f"Meets >=88 km^2 at least once: {'YES' if ok else 'NO'}\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"[OK] Wrote sweep details → {detailed_path}")
    print(f"[OK] Wrote summary → {summary_path}")


if __name__ == "__main__":
    main()
