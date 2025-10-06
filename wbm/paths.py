"""Minimal path utilities required by wbm.db.

This reconstruction provides only the functions referenced in db.py:
 - project_root(): return repo root (directory containing this file's parent)
 - ensure_dirs(): create expected storage subdirectories if missing
 - era5_csv_read_path(var): optional override via environment variables

Environment variable overrides:
  ERA5_CSV_PRECIPITATION, ERA5_CSV_EVAPORATION, ERA5_CSV_RUNOFF,
  ERA5_CSV_TEMPERATURE, ERA5_CSV_SWE, ERA5_CSV_SNOW_DEPTH
Return None if no override present.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def project_root() -> Path:
    # Assume this file is at <root>/wbm/paths.py
    return Path(__file__).resolve().parent.parent


def ensure_dirs() -> None:
    root = project_root()
    storage_base = root / "wbm" / "storage"
    for sub in [storage_base / "meta", storage_base / "sentinel"]:
        sub.mkdir(parents=True, exist_ok=True)


_ERA5_ENV_MAP = {
    "precipitation": "ERA5_CSV_PRECIPITATION",
    "evaporation": "ERA5_CSV_EVAPORATION",
    "runoff": "ERA5_CSV_RUNOFF",
    "temperature": "ERA5_CSV_TEMPERATURE",
    "swe": "ERA5_CSV_SWE",
    "snow_depth": "ERA5_CSV_SNOW_DEPTH",
}


def era5_csv_read_path(var: str) -> Optional[str]:  # pragma: no cover - straightforward lookup
    env = _ERA5_ENV_MAP.get(var)
    if not env:
        return None
    return os.environ.get(env)


__all__ = [
    "project_root",
    "ensure_dirs",
    "era5_csv_read_path",
]
