from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from wbm.paths import era5_csv_read_path, ensure_dirs, project_root
from wbm.ui.data_loader import _read_csv_safe, load_all_data

from wbm.data import partition_for_table, resolve_sqlite_partitions, resolve_sqlite_settings


@dataclass(frozen=True)
class TableAsset:
    name: str
    source_path: Path
    description: str = ""


def default_data_root() -> Path:
    env = os.environ.get("DATA_ROOT")
    if env:
        return Path(env).expanduser()
    return project_root()


def default_db_path(data_root: Path) -> Path:
    return data_root / "wbm" / "storage" / "water_balance.db"


def _resolve_era5_path(var: str, data_root: Path) -> Path:
    mapping = {
        "precipitation": "era5_land_total_precipitation_daily.csv",
        "evaporation": "era5_land_total_evaporation_daily.csv",
        "runoff": "era5_land_runoff_daily.csv",
        "temperature": "era5_land_temperature_daily.csv",
        "swe": "era5_land_swe_daily.csv",
        "snow_depth": "era5_land_snow_depth_daily.csv",
    }
    path = era5_csv_read_path(var)  # type: ignore[arg-type]
    if path is not None:
        return Path(path)
    return data_root / "wbm" / "storage" / mapping[var]


def _load_core_tables(
    data_root: Path,
    *,
    use_sqlite: bool,
    sqlite_paths: Mapping[str, Path] | None,
) -> dict[str, tuple[pd.DataFrame, TableAsset]]:
    ensure_dirs()

    output_dir = data_root / "water_balance_output"
    gleam_path = data_root / "GLEAM" / "processed" / "gleam_summary_all_years.csv"
    imerg_path = data_root / "precipitation_timeseries.csv"
    # Updated processed data path
    area_volume_path = data_root / "processed_data" / "processing_output" / "area_volume_curve.csv"

    era5_tp_path = _resolve_era5_path("precipitation", data_root)
    era5_e_path = _resolve_era5_path("evaporation", data_root)
    era5_swe_path = _resolve_era5_path("swe", data_root)
    era5_ro_path = _resolve_era5_path("runoff", data_root)
    era5_t2m_path = _resolve_era5_path("temperature", data_root)

    (
        balance_df,
        gleam_df,
        imerg_df,
        curve_df,
        era5_tp_df,
        era5_e_df,
        era5_swe_df,
        era5_ro_df,
        era5_t2m_df,
    ) = load_all_data(
        str(output_dir),
        str(gleam_path),
        str(imerg_path),
        str(area_volume_path),
        str(era5_tp_path),
        str(era5_e_path),
        str(era5_swe_path),
        str(era5_ro_path),
        str(era5_t2m_path),
        use_sqlite=use_sqlite,
        sqlite_paths={k: str(v) for k, v in (sqlite_paths or {}).items()},
    )

    core: dict[str, tuple[pd.DataFrame, TableAsset]] = {
        "balance": (
            balance_df,
            TableAsset(
                name="balance",
                source_path=output_dir / "water_balance_final.csv",
                description="Historical reservoir water balance (observed timeseries)",
            ),
        ),
        "gleam_evaporation": (
            gleam_df,
            TableAsset(
                name="gleam_evaporation",
                source_path=gleam_path,
                description="GLEAM evaporation time series (mm/day)",
            ),
        ),
        "imerg_precipitation": (
            imerg_df,
            TableAsset(
                name="imerg_precipitation",
                source_path=imerg_path,
                description="IMERG precipitation time series (mm/day)",
            ),
        ),
        "area_volume_curve": (
            curve_df,
            TableAsset(
                name="area_volume_curve",
                source_path=area_volume_path,
                description="Bathymetric area-volume relationship",
            ),
        ),
        "era5_precipitation": (
            era5_tp_df,
            TableAsset(
                name="era5_precipitation",
                source_path=era5_tp_path,
                description="ERA5-Land precipitation drivers (mm/day)",
            ),
        ),
        "era5_evaporation": (
            era5_e_df,
            TableAsset(
                name="era5_evaporation",
                source_path=era5_e_path,
                description="ERA5-Land evaporation drivers (mm/day)",
            ),
        ),
        "era5_swe": (
            era5_swe_df,
            TableAsset(
                name="era5_swe",
                source_path=era5_swe_path,
                description="ERA5-Land snow water equivalent (mm)",
            ),
        ),
        "era5_runoff": (
            era5_ro_df,
            TableAsset(
                name="era5_runoff",
                source_path=era5_ro_path,
                description="ERA5-Land runoff estimates (mm/day)",
            ),
        ),
        "era5_temperature": (
            era5_t2m_df,
            TableAsset(
                name="era5_temperature",
                source_path=era5_t2m_path,
                description="ERA5-Land 2m air temperature (°C)",
            ),
        ),
    }

    filled_path = output_dir / "water_balance_final_filled.csv"
    filled_df = _read_csv_safe(str(filled_path))
    if not filled_df.empty:
        core["balance_filled"] = (
            filled_df,
            TableAsset(
                name="balance_filled",
                source_path=filled_path,
                description="Gap-filled balance series (backfilled missing volumes)",
            ),
        )

    return core


def _discover_extra_csvs(data_root: Path, skip_tables: Iterable[str]) -> dict[str, tuple[pd.DataFrame, TableAsset]]:
    skip = set(skip_tables)
    extra: dict[str, tuple[pd.DataFrame, TableAsset]] = {}
    search_targets = [
        ("out_", data_root / "water_balance_output"),
    ("proc_", data_root / "processed_data" / "processing_output"),
    ]

    for prefix, directory in search_targets:
        if not directory.exists():
            continue
        for csv_path in sorted(directory.glob("*.csv")):
            stem = csv_path.stem.lower()
            table_name = f"{prefix}{stem}"
            if stem in skip or table_name in skip:
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:  # pragma: no cover - log-only branch
                logging.getLogger(__name__).warning("Skipping CSV %s: %s", csv_path, exc)
                continue
            for col in df.columns:
                if "date" in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    except Exception:
                        pass
            extra[table_name] = (
                df,
                TableAsset(
                    name=table_name,
                    source_path=csv_path,
                    description=f"Auto-ingested CSV from {directory.name}",
                ),
            )
    return extra


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    prepared = df.copy()
    for col in prepared.columns:
        series = prepared[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            prepared[col] = series.dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_timedelta64_dtype(series):
            prepared[col] = series.dt.total_seconds()
    return prepared.where(pd.notnull(prepared), None)


def build_database(
    db_path: Path,
    tables: dict[str, tuple[pd.DataFrame, TableAsset]],
    *,
    dry_run: bool = False,
) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    if dry_run:
        for name, (df, asset) in tables.items():
            records.append(
                {
                    "table_name": name,
                    "source_path": str(asset.source_path),
                    "description": asset.description,
                    "row_count": int(len(df)),
                    "ingested_at_utc": timestamp,
                }
            )
        return records

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF;")
        for name, (df, asset) in tables.items():
            row_count = int(len(df))
            if row_count == 0:
                logging.getLogger(__name__).info("Skipping empty table %s (source=%s)", name, asset.source_path)
            else:
                prepared = _prepare_dataframe(df)
                prepared.to_sql(name, conn, if_exists="replace", index=False)
                logging.getLogger(__name__).info("Wrote table %s (%d rows)", name, row_count)
            records.append(
                {
                    "table_name": name,
                    "source_path": str(asset.source_path),
                    "description": asset.description,
                    "row_count": row_count,
                    "ingested_at_utc": timestamp,
                    "column_types_json": json.dumps({col: str(dtype) for col, dtype in df.dtypes.items()}),
                }
            )
        metadata_df = pd.DataFrame(records)
        metadata_df.to_sql("_ingestion_log", conn, if_exists="replace", index=False)
    return records


def build_partitioned_databases(
    partition_paths: Mapping[str, Path],
    tables: dict[str, tuple[pd.DataFrame, TableAsset]],
    *,
    dry_run: bool = False,
) -> list[dict[str, str | int]]:
    grouped: dict[str, dict[str, tuple[pd.DataFrame, TableAsset]]] = {}
    for table_name, payload in tables.items():
        partition = partition_for_table(table_name)
        grouped.setdefault(partition, {})[table_name] = payload

    timestamp = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, str | int]] = []

    def _target_for(partition: str) -> Path:
        if partition in partition_paths:
            return Path(partition_paths[partition])
        if "core" in partition_paths:
            return Path(partition_paths["core"])
        raise KeyError(f"No target path configured for partition '{partition}'")

    if dry_run:
        for partition, part_tables in grouped.items():
            target_path = _target_for(partition)
            for name, (df, asset) in part_tables.items():
                records.append(
                    {
                        "table_name": name,
                        "source_path": str(asset.source_path),
                        "description": asset.description,
                        "row_count": int(len(df)),
                        "ingested_at_utc": timestamp,
                        "partition": partition,
                        "target_path": str(target_path),
                    }
                )
        return records

    for partition, part_tables in grouped.items():
        target_path = _target_for(partition)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        partition_records: list[dict[str, str | int]] = []

        with sqlite3.connect(target_path) as conn:
            conn.execute("PRAGMA foreign_keys = OFF;")
            for name, (df, asset) in part_tables.items():
                row_count = int(len(df))
                if row_count == 0:
                    logging.getLogger(__name__).info(
                        "Skipping empty table %s for partition %s (source=%s)",
                        name,
                        partition,
                        asset.source_path,
                    )
                else:
                    prepared = _prepare_dataframe(df)
                    prepared.to_sql(name, conn, if_exists="replace", index=False)
                    logging.getLogger(__name__).info(
                        "Wrote table %s to %s (%d rows)", name, target_path, row_count
                    )

                record: dict[str, str | int] = {
                    "table_name": name,
                    "source_path": str(asset.source_path),
                    "description": asset.description,
                    "row_count": row_count,
                    "ingested_at_utc": timestamp,
                    "partition": partition,
                    "target_path": str(target_path),
                    "column_types_json": json.dumps({col: str(dtype) for col, dtype in df.dtypes.items()}),
                }
                partition_records.append(record)

            metadata_df = pd.DataFrame(
                [{k: v for k, v in rec.items() if k != "target_path"} for rec in partition_records]
            )
            metadata_df.to_sql("_ingestion_log", conn, if_exists="replace", index=False)

        records.extend(partition_records)

    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a SQLite database from project CSV artifacts.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override data root (defaults to DATA_ROOT or project root)")
    parser.add_argument("--db", type=Path, default=None, help="Destination SQLite database path")
    parser.add_argument("--skip-extra", action="store_true", help="Skip auto-ingestion of additional CSVs")
    parser.add_argument("--dry-run", action="store_true", help="Load data and report table stats without writing the database")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity")
    parser.add_argument(
        "--input-source",
        default="auto",
        choices=["auto", "csv", "sqlite"],
        help="Select data source for ingestion. 'auto' prefers SQLite when available.",
    )
    parser.add_argument(
        "--input-sqlite-path",
        type=Path,
        default=None,
        help="Explicit path to the source SQLite database (defaults to detected location).",
    )
    parser.add_argument(
        "--partitioned-output",
        action="store_true",
        help="Write tables into partition-specific SQLite files (core/era5/outputs). Overrides --db destination.",
    )
    parser.add_argument(
        "--partition-dir",
        type=Path,
        default=None,
        help="Target directory for partitioned SQLite outputs (defaults to detected storage directory).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    data_root = args.data_root if args.data_root is not None else default_data_root()
    db_path = args.db if args.db is not None else default_db_path(data_root)

    auto_use_sqlite, auto_sqlite_paths = resolve_sqlite_settings(str(data_root))
    auto_sqlite_path_map = {k: Path(v) for k, v in auto_sqlite_paths.items()}
    _, partition_defaults_raw = resolve_sqlite_partitions(str(data_root), fallback_to_core=False)
    if args.input_sqlite_path is not None:
        inferred_sqlite_paths: Mapping[str, Path] = {"core": args.input_sqlite_path}
    else:
        inferred_sqlite_paths = auto_sqlite_path_map

    if args.input_source == "csv":
        source_use_sqlite = False
    elif args.input_source == "sqlite":
        source_use_sqlite = True
    else:
        source_use_sqlite = auto_use_sqlite

    core_tables = _load_core_tables(
        data_root,
        use_sqlite=source_use_sqlite,
        sqlite_paths=inferred_sqlite_paths if source_use_sqlite else None,
    )

    if args.skip_extra:
        tables = core_tables
    else:
        extra_tables = _discover_extra_csvs(data_root, skip_tables=core_tables.keys())
        tables = {**core_tables, **extra_tables}

    if args.partitioned_output:
        partition_dir = args.partition_dir
        if partition_dir is None and args.db is not None:
            partition_dir = args.db if args.db.suffix == "" else args.db.parent

        base_mapping = partition_defaults_raw or auto_sqlite_paths

        if partition_dir is not None:
            partition_dir = partition_dir.expanduser().resolve()
            partition_paths = {
                name: partition_dir / Path(path).name for name, path in base_mapping.items()
            }
        else:
            partition_paths = {name: Path(path) for name, path in base_mapping.items()}

        summary = build_partitioned_databases(partition_paths, tables, dry_run=args.dry_run)

        if args.dry_run:
            print("Dry-run summary (partitioned):")
            for record in summary:
                print(
                    f"- [{record['partition']}] {record['table_name']}: {record['row_count']} rows "
                    f"(source={record['source_path']}) -> {record['target_path']}"
                )
        else:
            partition_table_counts: dict[str, int] = {}
            partition_targets: dict[str, str] = {}
            for record in summary:
                partition = str(record["partition"])
                partition_table_counts[partition] = partition_table_counts.get(partition, 0) + 1
                partition_targets[partition] = str(record["target_path"])

            print("Partitioned databases written:")
            for partition in sorted(partition_table_counts):
                path = partition_targets.get(partition, "?")
                count = partition_table_counts[partition]
                print(f"- {partition}: {path} ({count} tables)")
        return

    summary = build_database(db_path, tables, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry-run summary:")
        for record in summary:
            print(f"- {record['table_name']}: {record['row_count']} rows (source={record['source_path']})")
    else:
        print(f"Database written to {db_path} ({len(summary)} tables including _ingestion_log).")


if __name__ == "__main__":  # pragma: no cover
    main()