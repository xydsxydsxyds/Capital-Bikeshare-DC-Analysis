"""Read, clean, enrich, and export Capital Bikeshare trip data.

The pipeline is intentionally chunked because the complete source data is much
larger than memory on many student machines.  It retains trips with missing
station fields for network/time and grid analysis, while exposing explicit
eligibility flags for station, OD, and distance analyses.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, MutableSet

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .config import (
    REQUIRED_COLUMNS,
    SOURCE_COLUMN_MAP,
    STRING_COLUMNS,
    CleaningConfig,
)
from .paths import DATA_DIR, PROCESSED_DIR, discover_trip_files, month_from_trip_file


EARTH_RADIUS_KM = 6371.0088


def canonicalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize Capital Bikeshare column names to the assignment schema."""

    frame = raw.copy()
    rename_map = {
        source: target
        for source, target in SOURCE_COLUMN_MAP.items()
        if source in frame.columns and target not in frame.columns
    }
    frame = frame.rename(columns=rename_map)

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            "输入 CSV 缺少必要字段: " + ", ".join(missing)
        )

    if "rideable_type" not in frame.columns:
        frame["rideable_type"] = pd.Series(pd.NA, index=frame.index, dtype="string")

    # Preserve a supplied duration for audit, but calculate the authoritative
    # duration from parsed timestamps below.
    if "trip_duration" in frame.columns and "trip_duration_source_s" not in frame.columns:
        frame = frame.rename(columns={"trip_duration": "trip_duration_source_s"})

    return frame


def _clean_string_columns(frame: pd.DataFrame) -> None:
    for column in STRING_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column].astype("string").str.strip()
        frame[column] = values.mask(values.eq(""), pd.NA)


def haversine_km(
    start_lon: pd.Series,
    start_lat: pd.Series,
    end_lon: pd.Series,
    end_lat: pd.Series,
) -> pd.Series:
    """Vectorized great-circle displacement in kilometres."""

    lon1 = np.radians(start_lon.to_numpy(dtype="float64"))
    lat1 = np.radians(start_lat.to_numpy(dtype="float64"))
    lon2 = np.radians(end_lon.to_numpy(dtype="float64"))
    lat2 = np.radians(end_lat.to_numpy(dtype="float64"))

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    distance = 2.0 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return pd.Series(distance, index=start_lon.index, dtype="float64")


def _coordinate_quality(
    frame: pd.DataFrame,
    config: CleaningConfig,
) -> dict[str, pd.Series]:
    for column in ("start_lat", "start_lon", "end_lat", "end_lon"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    start_missing = frame[["start_lat", "start_lon"]].isna().any(axis=1)
    end_missing = frame[["end_lat", "end_lon"]].isna().any(axis=1)

    start_out_of_bounds = (~start_missing) & (
        ~frame["start_lat"].between(config.min_latitude, config.max_latitude)
        | ~frame["start_lon"].between(config.min_longitude, config.max_longitude)
    )
    end_out_of_bounds = (~end_missing) & (
        ~frame["end_lat"].between(config.min_latitude, config.max_latitude)
        | ~frame["end_lon"].between(config.min_longitude, config.max_longitude)
    )

    # Do not let clearly wrong values enter any downstream map or distance
    # calculation. Missing/invalid coordinates remain auditable through flags.
    frame.loc[start_out_of_bounds, ["start_lat", "start_lon"]] = np.nan
    frame.loc[end_out_of_bounds, ["end_lat", "end_lon"]] = np.nan

    return {
        "start_coords_missing": start_missing,
        "start_coords_out_of_bounds": start_out_of_bounds,
        "end_coords_missing": end_missing,
        "end_coords_out_of_bounds": end_out_of_bounds,
        "has_valid_start_coords": ~(start_missing | start_out_of_bounds),
        "has_valid_end_coords": ~(end_missing | end_out_of_bounds),
    }


def _add_calendar_features(frame: pd.DataFrame, config: CleaningConfig) -> None:
    start_time = frame["start_time"]
    frame["start_date"] = start_time.dt.normalize()
    frame["start_year"] = start_time.dt.year.astype("int16")
    frame["start_month"] = start_time.dt.month.astype("int8")
    frame["start_hour"] = start_time.dt.hour.astype("int8")
    frame["start_weekday"] = start_time.dt.weekday.astype("int8")
    frame["start_weekday_name"] = start_time.dt.day_name()

    if frame.empty:
        holiday_dates = pd.DatetimeIndex([])
    else:
        holiday_dates = USFederalHolidayCalendar().holidays(
            start=frame["start_date"].min(),
            end=frame["start_date"].max(),
        )

    frame["is_weekend"] = frame["start_weekday"].ge(5)
    frame["is_federal_holiday"] = frame["start_date"].isin(holiday_dates)
    frame["is_workday"] = ~frame["is_weekend"] & ~frame["is_federal_holiday"]

    hour = frame["start_hour"]
    conditions = [
        hour.lt(config.night_end) | hour.ge(config.night_start),
        hour.ge(config.morning_peak_start) & hour.lt(config.morning_peak_end),
        hour.ge(config.evening_peak_start) & hour.lt(config.evening_peak_end),
    ]
    choices = ["night", "morning_peak", "evening_peak"]
    frame["traffic_period"] = pd.Series(
        np.select(conditions, choices, default="off_peak"),
        index=frame.index,
        dtype="string",
    )


def preprocess_dataframe(
    raw: pd.DataFrame,
    config: CleaningConfig | None = None,
    seen_ride_ids: MutableSet[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean one DataFrame and return the enriched rows plus audit counts.

    Missing station fields are deliberately retained. Rows are removed only for
    missing/duplicate trip identifiers, invalid key fields, or invalid duration.
    Coordinate problems are neutralized and represented by eligibility flags so
    temporal analysis does not silently lose otherwise valid trips.
    """

    config = config or CleaningConfig()
    frame = canonicalize_columns(raw)
    _clean_string_columns(frame)

    metrics: dict[str, int] = {"input_rows": int(len(frame))}
    missing_ride_id = frame["ride_id"].isna()
    within_chunk_duplicate = (
        frame["ride_id"].notna() & frame["ride_id"].duplicated(keep="first")
    )

    if seen_ride_ids is None:
        seen_ride_ids = set()
    # ``Series.isin(large_set)`` rebuilds an array from the complete set for
    # every chunk. Across a full year that becomes progressively slower. Direct
    # hash-set membership keeps the global de-duplication cost linear in the
    # number of rows in the current chunk.
    already_seen = pd.Series(
        np.fromiter(
            (
                False if pd.isna(ride_id) else ride_id in seen_ride_ids
                for ride_id in frame["ride_id"].array
            ),
            dtype=bool,
            count=len(frame),
        ),
        index=frame.index,
    )

    metrics["missing_ride_id_rows"] = int(missing_ride_id.sum())
    metrics["duplicate_ride_id_within_chunk_rows"] = int(within_chunk_duplicate.sum())
    metrics["duplicate_ride_id_previous_chunk_rows"] = int(already_seen.sum())

    identifier_keep = ~(missing_ride_id | within_chunk_duplicate | already_seen)
    new_ids = frame.loc[identifier_keep, "ride_id"].dropna().tolist()
    seen_ride_ids.update(new_ids)
    frame = frame.loc[identifier_keep].copy()

    frame["start_time"] = pd.to_datetime(
        frame["start_time"], errors="coerce", format="mixed"
    )
    frame["end_time"] = pd.to_datetime(
        frame["end_time"], errors="coerce", format="mixed"
    )

    invalid_start_time = frame["start_time"].isna()
    invalid_end_time = frame["end_time"].isna()
    missing_user_type = frame["user_type"].isna()
    metrics["invalid_start_time_rows"] = int(invalid_start_time.sum())
    metrics["invalid_end_time_rows"] = int(invalid_end_time.sum())
    metrics["missing_user_type_rows"] = int(missing_user_type.sum())

    key_keep = ~(invalid_start_time | invalid_end_time | missing_user_type)
    frame = frame.loc[key_keep].copy()

    frame["trip_duration_s"] = (
        frame["end_time"] - frame["start_time"]
    ).dt.total_seconds()
    invalid_duration = ~np.isfinite(frame["trip_duration_s"])
    negative_duration = frame["trip_duration_s"].lt(0)
    too_short_duration = frame["trip_duration_s"].ge(0) & frame[
        "trip_duration_s"
    ].lt(config.min_duration_s)
    too_long_duration = frame["trip_duration_s"].gt(config.max_duration_s)
    duration_keep = ~(
        invalid_duration
        | negative_duration
        | too_short_duration
        | too_long_duration
    )

    metrics["invalid_duration_rows"] = int(invalid_duration.sum())
    metrics["negative_duration_rows"] = int(negative_duration.sum())
    metrics["too_short_duration_rows"] = int(too_short_duration.sum())
    metrics["too_long_duration_rows"] = int(too_long_duration.sum())
    metrics["duration_excluded_rows"] = int((~duration_keep).sum())

    frame = frame.loc[duration_keep].copy()
    frame["trip_duration"] = frame["trip_duration_s"]
    frame["trip_duration_min"] = frame["trip_duration_s"] / 60.0

    coordinate_flags = _coordinate_quality(frame, config)
    for column, values in coordinate_flags.items():
        frame[column] = values.astype(bool)
        metrics[f"{column}_rows"] = int(values.sum())

    valid_coordinate_pair = (
        frame["has_valid_start_coords"] & frame["has_valid_end_coords"]
    )
    distance_raw = pd.Series(np.nan, index=frame.index, dtype="float64")
    if valid_coordinate_pair.any():
        valid_rows = frame.loc[valid_coordinate_pair]
        distance_raw.loc[valid_coordinate_pair] = haversine_km(
            valid_rows["start_lon"],
            valid_rows["start_lat"],
            valid_rows["end_lon"],
            valid_rows["end_lat"],
        )

    frame["straight_line_distance_km_raw"] = distance_raw
    frame["speed_proxy_kmh"] = distance_raw / (frame["trip_duration_s"] / 3600.0)
    frame["distance_outlier"] = distance_raw.gt(
        config.max_straight_line_distance_km
    ).fillna(False)
    frame["speed_outlier"] = frame["speed_proxy_kmh"].gt(
        config.max_speed_proxy_kmh
    ).fillna(False)
    frame["eligible_distance_analysis"] = (
        valid_coordinate_pair
        & ~frame["distance_outlier"]
        & ~frame["speed_outlier"]
    )
    frame["distance_km"] = distance_raw.where(frame["eligible_distance_analysis"])

    frame["zero_displacement"] = (
        valid_coordinate_pair & distance_raw.le(1e-9).fillna(False)
    )
    frame["loop_trip"] = (
        frame["start_station_id"].notna()
        & frame["end_station_id"].notna()
        & frame["start_station_id"].eq(frame["end_station_id"])
    )

    frame["eligible_temporal_analysis"] = True
    frame["eligible_start_geo_analysis"] = frame["has_valid_start_coords"]
    frame["eligible_arrival_geo_analysis"] = frame["has_valid_end_coords"]
    frame["eligible_start_station_analysis"] = (
        frame["start_station_id"].notna() & frame["start_station_name"].notna()
    )
    frame["eligible_end_station_analysis"] = (
        frame["end_station_id"].notna() & frame["end_station_name"].notna()
    )
    frame["eligible_station_od_analysis"] = (
        frame["eligible_start_station_analysis"]
        & frame["eligible_end_station_analysis"]
    )

    metrics["distance_outlier_rows"] = int(frame["distance_outlier"].sum())
    metrics["speed_outlier_rows"] = int(frame["speed_outlier"].sum())
    metrics["missing_start_station_rows"] = int(
        (~frame["eligible_start_station_analysis"]).sum()
    )
    metrics["missing_end_station_rows"] = int(
        (~frame["eligible_end_station_analysis"]).sum()
    )
    metrics["eligible_start_geo_rows"] = int(
        frame["eligible_start_geo_analysis"].sum()
    )
    metrics["eligible_distance_rows"] = int(
        frame["eligible_distance_analysis"].sum()
    )
    metrics["eligible_station_od_rows"] = int(
        frame["eligible_station_od_analysis"].sum()
    )

    _add_calendar_features(frame, config)
    metrics["cleaned_rows_before_period_filter"] = int(len(frame))

    preferred_order = [
        "ride_id",
        "rideable_type",
        "start_time",
        "end_time",
        "start_station_id",
        "start_station_name",
        "end_station_id",
        "end_station_name",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
        "user_type",
        "trip_duration",
        "trip_duration_s",
        "trip_duration_min",
        "start_date",
        "start_year",
        "start_month",
        "start_hour",
        "start_weekday",
        "start_weekday_name",
        "is_weekend",
        "is_federal_holiday",
        "is_workday",
        "traffic_period",
        "straight_line_distance_km_raw",
        "distance_km",
        "speed_proxy_kmh",
        "distance_outlier",
        "speed_outlier",
        "zero_displacement",
        "loop_trip",
        "has_valid_start_coords",
        "has_valid_end_coords",
        "eligible_temporal_analysis",
        "eligible_start_geo_analysis",
        "eligible_arrival_geo_analysis",
        "eligible_distance_analysis",
        "eligible_start_station_analysis",
        "eligible_end_station_analysis",
        "eligible_station_od_analysis",
    ]
    remaining_columns = [c for c in frame.columns if c not in preferred_order]
    frame = frame[preferred_order + remaining_columns]
    return frame.reset_index(drop=True), metrics


def _read_csv_chunks(
    path: Path,
    chunksize: int,
    nrows: int | None = None,
) -> Iterable[pd.DataFrame]:
    header = pd.read_csv(path, nrows=0).columns
    string_candidates = {
        "ride_id",
        "rideable_type",
        "start_station_id",
        "start_station_name",
        "end_station_id",
        "end_station_name",
        "member_casual",
        "user_type",
    }
    dtypes = {column: "string" for column in header if column in string_candidates}
    return pd.read_csv(
        path,
        dtype=dtypes,
        chunksize=chunksize,
        nrows=nrows,
        low_memory=False,
    )


class _ChunkWriter:
    """Atomic incremental CSV or Parquet writer."""

    def __init__(self, destination: Path, output_format: str) -> None:
        self.destination = destination
        self.output_format = output_format
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        file_handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-",
            suffix=f".{output_format}.tmp",
            dir=destination.parent,
        )
        os.close(file_handle)
        self.temporary_path = Path(temporary_name)
        self.wrote_rows = False
        self.empty_template: pd.DataFrame | None = None
        self._parquet_writer: Any | None = None
        self._arrow_schema: Any | None = None

        if output_format == "parquet":
            try:
                import pyarrow as pa  # noqa: F401
                import pyarrow.parquet as pq  # noqa: F401
            except ImportError as exc:
                self.temporary_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Parquet 输出需要 pyarrow；请执行 pip install -r requirements.txt，"
                    "或使用 --output-format csv。"
                ) from exc

    def write(self, frame: pd.DataFrame) -> None:
        if self.empty_template is None:
            self.empty_template = frame.head(0).copy()
        if frame.empty:
            return

        if self.output_format == "csv":
            frame.to_csv(
                self.temporary_path,
                mode="a",
                header=not self.wrote_rows,
                index=False,
                encoding="utf-8",
            )
        else:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if self._parquet_writer is None:
                self._arrow_schema = table.schema
                self._parquet_writer = pq.ParquetWriter(
                    self.temporary_path,
                    self._arrow_schema,
                    compression="zstd",
                )
            elif not table.schema.equals(self._arrow_schema, check_metadata=False):
                table = table.cast(self._arrow_schema)
            self._parquet_writer.write_table(table)

        self.wrote_rows = True

    def finish(self) -> bool:
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None

        if not self.wrote_rows:
            if self.output_format == "csv" and self.empty_template is not None:
                self.empty_template.to_csv(
                    self.temporary_path, index=False, encoding="utf-8"
                )
            else:
                self.temporary_path.unlink(missing_ok=True)
                return False

        os.replace(self.temporary_path, self.destination)
        return True

    def abort(self) -> None:
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
        self.temporary_path.unlink(missing_ok=True)


def _parse_boundary(value: str | None, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{name} 不是有效日期/时间: {value!r}")
    return pd.Timestamp(parsed)


def _filter_period(
    frame: pd.DataFrame,
    start_time: pd.Timestamp | None,
    end_time: pd.Timestamp | None,
) -> tuple[pd.DataFrame, int]:
    keep = pd.Series(True, index=frame.index)
    if start_time is not None:
        keep &= frame["start_time"].ge(start_time)
    if end_time is not None:
        keep &= frame["start_time"].lt(end_time)
    return frame.loc[keep].copy(), int((~keep).sum())


def _write_quality_reports(
    reports: list[dict[str, Any]],
    output_dir: Path,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    chunk_report = pd.DataFrame(reports)
    metric_columns = [
        column
        for column in chunk_report.columns
        if column not in {"source_month", "source_file", "chunk_index"}
    ]
    by_file = (
        chunk_report.groupby(["source_month", "source_file"], as_index=False)[
            metric_columns
        ]
        .sum()
        .sort_values("source_month")
    )
    total: dict[str, Any] = {"source_month": "TOTAL", "source_file": "ALL"}
    total.update({column: int(by_file[column].sum()) for column in metric_columns})
    summary = pd.concat([by_file, pd.DataFrame([total])], ignore_index=True)

    chunk_report.to_csv(output_dir / "quality_report_by_chunk.csv", index=False)
    summary.to_csv(output_dir / "quality_report_summary.csv", index=False)
    (output_dir / "preprocess_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def process_trip_files(
    files: list[Path],
    output_dir: Path = PROCESSED_DIR,
    *,
    config: CleaningConfig | None = None,
    chunksize: int = 100_000,
    output_format: str = "parquet",
    start_time: str | None = None,
    end_time: str | None = None,
    max_rows: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Process discovered monthly files and return a quality-report summary."""

    if not files:
        raise ValueError("没有找到符合月份目录/文件命名规则的输入 CSV")
    if chunksize <= 0:
        raise ValueError("chunksize 必须大于 0")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows 必须大于 0")
    if output_format not in {"parquet", "csv"}:
        raise ValueError("output_format 必须是 parquet 或 csv")

    config = config or CleaningConfig()
    period_start = _parse_boundary(start_time, "start_time")
    period_end = _parse_boundary(end_time, "end_time")
    if period_start is not None and period_end is not None and period_start >= period_end:
        raise ValueError("start_time 必须早于 end_time（end_time 为开区间）")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_targets = [
            output_dir / "quality_report_by_chunk.csv",
            output_dir / "quality_report_summary.csv",
            output_dir / "preprocess_metadata.json",
        ]
        if not overwrite and any(path.exists() for path in report_targets):
            raise FileExistsError(
                "输出目录已有质量报告；如确认替换，请增加 --overwrite"
            )

    seen_ride_ids: set[str] = set()
    reports: list[dict[str, Any]] = []
    rows_read = 0
    output_files: list[str] = []

    for source_file in files:
        if max_rows is not None and rows_read >= max_rows:
            break

        source_month = month_from_trip_file(source_file)
        suffix = ".parquet" if output_format == "parquet" else ".csv"
        destination = output_dir / f"{source_month}-trips-cleaned{suffix}"
        if not dry_run and destination.exists() and not overwrite:
            raise FileExistsError(
                f"输出文件已存在: {destination}；如确认替换，请增加 --overwrite"
            )

        writer = None if dry_run else _ChunkWriter(destination, output_format)
        file_succeeded = False
        try:
            remaining = None if max_rows is None else max_rows - rows_read
            for chunk_index, raw_chunk in enumerate(
                _read_csv_chunks(source_file, chunksize, nrows=remaining), start=1
            ):
                rows_read += len(raw_chunk)
                cleaned, metrics = preprocess_dataframe(
                    raw_chunk,
                    config=config,
                    seen_ride_ids=seen_ride_ids,
                )
                cleaned, outside_period = _filter_period(
                    cleaned, period_start, period_end
                )
                cleaned["source_month"] = source_month
                cleaned["source_file"] = source_file.name

                metrics["outside_requested_period_rows"] = outside_period
                metrics["output_rows"] = int(len(cleaned))
                reports.append(
                    {
                        "source_month": source_month,
                        "source_file": source_file.name,
                        "chunk_index": chunk_index,
                        **metrics,
                    }
                )

                if writer is not None:
                    writer.write(cleaned)

                print(
                    f"[{source_month}] chunk {chunk_index}: "
                    f"read={len(raw_chunk):,}, output={len(cleaned):,}"
                )

            if writer is not None and writer.finish():
                output_files.append(str(destination))
            file_succeeded = True
        finally:
            if writer is not None and not file_succeeded:
                writer.abort()

    if not reports:
        raise ValueError("输入文件中没有读到任何记录")

    chunk_report = pd.DataFrame(reports)
    metric_columns = [
        column
        for column in chunk_report.columns
        if column not in {"source_month", "source_file", "chunk_index"}
    ]
    by_file = chunk_report.groupby(
        ["source_month", "source_file"], as_index=False
    )[metric_columns].sum()
    total: dict[str, Any] = {"source_month": "TOTAL", "source_file": "ALL"}
    total.update({column: int(by_file[column].sum()) for column in metric_columns})
    summary = pd.concat([by_file, pd.DataFrame([total])], ignore_index=True)

    if not dry_run:
        metadata = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_files": [str(path) for path in files],
            "output_files": output_files,
            "output_format": output_format,
            "chunksize": chunksize,
            "start_time_inclusive": start_time,
            "end_time_exclusive": end_time,
            "max_rows": max_rows,
            "source_times_interpreted_as": config.local_timezone,
            "cleaning_config": asdict(config),
        }
        summary = _write_quality_reports(reports, output_dir, metadata)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capital Bikeshare 数据读取、清洗与特征工程"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--start-month", help="输入文件起始月份 YYYYMM（含）")
    parser.add_argument("--end-month", help="输入文件结束月份 YYYYMM（含）")
    parser.add_argument("--start-time", help="按 start_time 筛选的起点（含）")
    parser.add_argument("--end-time", help="按 start_time 筛选的终点（不含）")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--output-format", choices=("parquet", "csv"), default="parquet"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="仅用于开发/冒烟测试：最多读取的总行数",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="执行全部读取与清洗统计，但不写输出文件",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = discover_trip_files(
        args.data_dir,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    summary = process_trip_files(
        files,
        output_dir=args.output_dir,
        chunksize=args.chunksize,
        output_format=args.output_format,
        start_time=args.start_time,
        end_time=args.end_time,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    display_columns = [
        "source_month",
        "input_rows",
        "duration_excluded_rows",
        "cleaned_rows_before_period_filter",
        "outside_requested_period_rows",
        "output_rows",
    ]
    print("\n清洗汇总：")
    print(summary[display_columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
