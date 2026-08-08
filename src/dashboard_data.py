"""Prepare compact, filterable data marts for the Streamlit dashboard.

The dashboard must respond quickly to date and user-type filters without
loading 6.5 million cleaned trip rows on every interaction.  This module scans
the 2025 Parquet files in batches and writes additive daily/hourly station,
grid, and Top-10-node OD aggregates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from pyproj import Transformer

from .paths import PROCESSED_DIR
from .spatial_od import DEFAULT_GRID_SIZE_M, UTM18N_EPSG, WGS84_EPSG
from .temporal import discover_cleaned_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dashboard" / "data"
DEFAULT_STATION_STATS = (
    PROJECT_ROOT / "outputs" / "spatial_od" / "tables" / "station_stats.csv"
)
DEFAULT_TOP_NODES = (
    PROJECT_ROOT
    / "outputs"
    / "spatial_od"
    / "tables"
    / "top10_activity_nodes.csv"
)

REQUIRED_COLUMNS = (
    "start_time",
    "user_type",
    "trip_duration_s",
    "distance_km",
    "start_station_id",
    "end_station_id",
    "start_lat",
    "start_lon",
    "end_lat",
    "end_lon",
    "has_valid_start_coords",
    "has_valid_end_coords",
)


@dataclass
class DashboardDataResult:
    daily_user_summary: pd.DataFrame
    hourly_user_volume: pd.DataFrame
    station_daily_user: pd.DataFrame
    grid_daily_user: pd.DataFrame
    od_daily_user: pd.DataFrame
    station_reference: pd.DataFrame
    top_activity_nodes: pd.DataFrame
    input_rows: int
    analysis_rows: int
    grid_size_m: int


def _read_dashboard_chunks(
    files: Sequence[Path], chunksize: int
) -> Iterator[pd.DataFrame]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "仪表板数据准备需要 pyarrow；请执行 pip install -r requirements.txt"
        ) from exc

    for path in files:
        if path.suffix == ".csv":
            header = set(pd.read_csv(path, nrows=0).columns)
            missing = sorted(set(REQUIRED_COLUMNS) - header)
            if missing:
                raise ValueError(f"{path.name} 缺少字段: {', '.join(missing)}")
            yield from pd.read_csv(
                path,
                usecols=list(REQUIRED_COLUMNS),
                chunksize=chunksize,
                low_memory=False,
            )
            continue

        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        missing = sorted(set(REQUIRED_COLUMNS) - available)
        if missing:
            raise ValueError(f"{path.name} 缺少字段: {', '.join(missing)}")
        for batch in parquet_file.iter_batches(
            batch_size=chunksize, columns=list(REQUIRED_COLUMNS)
        ):
            yield batch.to_pandas()


def _coerce_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError("清洗数据缺少仪表板字段: " + ", ".join(missing))
    frame = raw.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
    frame["user_type"] = frame["user_type"].astype("string").str.strip()
    frame = frame.loc[
        frame["start_time"].notna()
        & frame["user_type"].notna()
        & frame["user_type"].ne("")
    ].copy()
    frame["date"] = frame["start_time"].dt.normalize()
    frame["hour"] = frame["start_time"].dt.hour.astype("int8")
    for column in (
        "trip_duration_s",
        "distance_km",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("start_station_id", "end_station_id"):
        values = frame[column].astype("string").str.strip()
        frame[column] = values.mask(values.eq(""), pd.NA)
    for column in ("has_valid_start_coords", "has_valid_end_coords"):
        frame[column] = frame[column].fillna(False).astype(bool)
    return frame


def _collapse(
    parts: list[pd.DataFrame], keys: list[str], values: list[str]
) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=keys + values)
    combined = pd.concat(parts, ignore_index=True)
    return (
        combined.groupby(keys, as_index=False, observed=True)[values]
        .sum()
        .sort_values(keys)
        .reset_index(drop=True)
    )


def _grid_chunk_counts(
    frame: pd.DataFrame,
    endpoint: str,
    transformer: Transformer,
    grid_size_m: int,
) -> pd.DataFrame:
    valid_column = f"has_valid_{endpoint}_coords"
    lat_column = f"{endpoint}_lat"
    lon_column = f"{endpoint}_lon"
    mask = (
        frame[valid_column]
        & frame[lat_column].notna()
        & frame[lon_column].notna()
    )
    selected = frame.loc[mask, ["date", "user_type", lat_column, lon_column]]
    if selected.empty:
        return pd.DataFrame(
            columns=["date", "user_type", "grid_x", "grid_y", endpoint]
        )
    eastings, northings = transformer.transform(
        selected[lon_column].to_numpy(dtype="float64"),
        selected[lat_column].to_numpy(dtype="float64"),
    )
    grid = pd.DataFrame(
        {
            "date": selected["date"].to_numpy(),
            "user_type": selected["user_type"].to_numpy(),
            "grid_x": np.floor(np.asarray(eastings) / grid_size_m).astype("int32"),
            "grid_y": np.floor(np.asarray(northings) / grid_size_m).astype("int32"),
        }
    )
    value_column = "departures" if endpoint == "start" else "arrivals"
    return (
        grid.groupby(
            ["date", "user_type", "grid_x", "grid_y"],
            as_index=False,
            observed=True,
        )
        .size()
        .rename(columns={"size": value_column})
    )


def aggregate_dashboard_chunks(
    chunks: Iterable[pd.DataFrame],
    station_reference: pd.DataFrame,
    top_activity_nodes: pd.DataFrame,
    *,
    grid_size_m: int = DEFAULT_GRID_SIZE_M,
) -> DashboardDataResult:
    if grid_size_m <= 0:
        raise ValueError("grid_size_m 必须大于 0")

    station_reference = station_reference.copy()
    top_activity_nodes = top_activity_nodes.copy()
    for frame in (station_reference, top_activity_nodes):
        frame["station_id"] = frame["station_id"].astype("string")
    top_node_ids = set(top_activity_nodes["station_id"].dropna().tolist())
    if not top_node_ids:
        raise ValueError("Top 10 活动节点为空，无法准备 OD 仪表板数据")

    forward = Transformer.from_crs(WGS84_EPSG, UTM18N_EPSG, always_xy=True)
    inverse = Transformer.from_crs(UTM18N_EPSG, WGS84_EPSG, always_xy=True)

    daily_parts: list[pd.DataFrame] = []
    hourly_parts: list[pd.DataFrame] = []
    start_station_parts: list[pd.DataFrame] = []
    end_station_parts: list[pd.DataFrame] = []
    departure_grid_parts: list[pd.DataFrame] = []
    arrival_grid_parts: list[pd.DataFrame] = []
    od_parts: list[pd.DataFrame] = []
    input_rows = 0
    analysis_rows = 0

    for raw in chunks:
        input_rows += len(raw)
        frame = _coerce_chunk(raw)
        analysis_rows += len(frame)
        if frame.empty:
            continue

        frame["duration_min"] = frame["trip_duration_s"] / 60.0
        daily = (
            frame.groupby(["date", "user_type"], as_index=False, observed=True)
            .agg(
                trip_count=("start_time", "size"),
                total_duration_min=("duration_min", "sum"),
                total_distance_km=("distance_km", "sum"),
                distance_valid_trips=("distance_km", "count"),
            )
        )
        daily_parts.append(daily)
        hourly_parts.append(
            frame.groupby(
                ["date", "hour", "user_type"], as_index=False, observed=True
            )
            .size()
            .rename(columns={"size": "trip_count"})
        )

        valid_start_station = frame["start_station_id"].notna()
        if valid_start_station.any():
            start_station_parts.append(
                frame.loc[valid_start_station]
                .groupby(
                    ["date", "user_type", "start_station_id"],
                    as_index=False,
                    observed=True,
                )
                .size()
                .rename(
                    columns={
                        "start_station_id": "station_id",
                        "size": "departures",
                    }
                )
            )
        valid_end_station = frame["end_station_id"].notna()
        if valid_end_station.any():
            end_station_parts.append(
                frame.loc[valid_end_station]
                .groupby(
                    ["date", "user_type", "end_station_id"],
                    as_index=False,
                    observed=True,
                )
                .size()
                .rename(
                    columns={"end_station_id": "station_id", "size": "arrivals"}
                )
            )

        departure_grid_parts.append(
            _grid_chunk_counts(frame, "start", forward, grid_size_m)
        )
        arrival_grid_parts.append(
            _grid_chunk_counts(frame, "end", forward, grid_size_m)
        )

        valid_od = (
            frame["start_station_id"].isin(top_node_ids)
            & frame["end_station_id"].isin(top_node_ids)
            & frame["start_station_id"].ne(frame["end_station_id"])
        )
        if valid_od.any():
            od_parts.append(
                frame.loc[valid_od]
                .groupby(
                    [
                        "date",
                        "user_type",
                        "start_station_id",
                        "end_station_id",
                    ],
                    as_index=False,
                    observed=True,
                )
                .size()
                .rename(
                    columns={
                        "start_station_id": "source_station_id",
                        "end_station_id": "target_station_id",
                        "size": "trip_count",
                    }
                )
            )

    if analysis_rows == 0:
        raise ValueError("没有可用于仪表板聚合的数据")

    daily_user = _collapse(
        daily_parts,
        ["date", "user_type"],
        [
            "trip_count",
            "total_duration_min",
            "total_distance_km",
            "distance_valid_trips",
        ],
    )
    hourly_user = _collapse(
        hourly_parts, ["date", "hour", "user_type"], ["trip_count"]
    )
    starts = _collapse(
        start_station_parts, ["date", "user_type", "station_id"], ["departures"]
    )
    ends = _collapse(
        end_station_parts, ["date", "user_type", "station_id"], ["arrivals"]
    )
    station_daily = starts.merge(
        ends, on=["date", "user_type", "station_id"], how="outer"
    )
    station_daily[["departures", "arrivals"]] = station_daily[
        ["departures", "arrivals"]
    ].fillna(0).astype("int64")
    station_daily["total_activity"] = (
        station_daily["departures"] + station_daily["arrivals"]
    )
    station_daily["net_inflow"] = (
        station_daily["arrivals"] - station_daily["departures"]
    )
    station_daily = station_daily.sort_values(
        ["date", "user_type", "station_id"]
    ).reset_index(drop=True)

    departures = _collapse(
        departure_grid_parts,
        ["date", "user_type", "grid_x", "grid_y"],
        ["departures"],
    )
    arrivals = _collapse(
        arrival_grid_parts,
        ["date", "user_type", "grid_x", "grid_y"],
        ["arrivals"],
    )
    grid_daily = departures.merge(
        arrivals,
        on=["date", "user_type", "grid_x", "grid_y"],
        how="outer",
    )
    grid_daily[["departures", "arrivals"]] = grid_daily[
        ["departures", "arrivals"]
    ].fillna(0).astype("int64")
    grid_daily["total_activity"] = (
        grid_daily["departures"] + grid_daily["arrivals"]
    )
    grid_daily["net_inflow"] = grid_daily["arrivals"] - grid_daily["departures"]
    grid_daily["grid_id"] = (
        "E"
        + grid_daily["grid_x"].astype(str)
        + "_N"
        + grid_daily["grid_y"].astype(str)
    )
    centers = grid_daily[["grid_x", "grid_y", "grid_id"]].drop_duplicates()
    center_easting = (centers["grid_x"] + 0.5) * grid_size_m
    center_northing = (centers["grid_y"] + 0.5) * grid_size_m
    center_lon, center_lat = inverse.transform(
        center_easting.to_numpy(), center_northing.to_numpy()
    )
    centers["center_lat"] = center_lat
    centers["center_lon"] = center_lon
    grid_daily = grid_daily.merge(
        centers, on=["grid_x", "grid_y", "grid_id"], validate="many_to_one"
    ).sort_values(["date", "user_type", "grid_id"])

    od_daily = _collapse(
        od_parts,
        ["date", "user_type", "source_station_id", "target_station_id"],
        ["trip_count"],
    )
    name_map = station_reference.set_index("station_id")["station_name"]
    od_daily["source_station_name"] = od_daily["source_station_id"].map(name_map)
    od_daily["target_station_name"] = od_daily["target_station_id"].map(name_map)

    integer_columns = {
        "trip_count",
        "distance_valid_trips",
        "departures",
        "arrivals",
        "total_activity",
        "net_inflow",
    }
    for output in (daily_user, hourly_user, station_daily, grid_daily, od_daily):
        for column in integer_columns.intersection(output.columns):
            output[column] = output[column].astype("int64")

    return DashboardDataResult(
        daily_user_summary=daily_user,
        hourly_user_volume=hourly_user,
        station_daily_user=station_daily,
        grid_daily_user=grid_daily.reset_index(drop=True),
        od_daily_user=od_daily,
        station_reference=station_reference,
        top_activity_nodes=top_activity_nodes,
        input_rows=input_rows,
        analysis_rows=analysis_rows,
        grid_size_m=grid_size_m,
    )


def prepare_dashboard_data(
    files: Sequence[Path],
    *,
    station_stats_path: Path = DEFAULT_STATION_STATS,
    top_nodes_path: Path = DEFAULT_TOP_NODES,
    chunksize: int = 200_000,
    grid_size_m: int = DEFAULT_GRID_SIZE_M,
) -> DashboardDataResult:
    if not files:
        raise ValueError("没有找到仪表板所需的2025清洗文件")
    if not station_stats_path.exists() or not top_nodes_path.exists():
        raise FileNotFoundError("缺少空间分析站点表；请先运行 python -m src.spatial_od")
    station_reference = pd.read_csv(station_stats_path, dtype={"station_id": "string"})
    top_nodes = pd.read_csv(top_nodes_path, dtype={"station_id": "string"})
    return aggregate_dashboard_chunks(
        _read_dashboard_chunks(files, chunksize),
        station_reference,
        top_nodes,
        grid_size_m=grid_size_m,
    )


def save_dashboard_data(
    result: DashboardDataResult,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    source_files: Sequence[Path] = (),
    overwrite: bool = False,
) -> list[Path]:
    outputs = {
        "daily_user_summary.parquet": result.daily_user_summary,
        "hourly_user_volume.parquet": result.hourly_user_volume,
        "station_daily_user.parquet": result.station_daily_user,
        "grid_daily_user.parquet": result.grid_daily_user,
        "od_daily_user.parquet": result.od_daily_user,
        "station_reference.parquet": result.station_reference,
        "top_activity_nodes.parquet": result.top_activity_nodes,
    }
    targets = [output_dir / name for name in outputs]
    metadata_path = output_dir / "dashboard_data_metadata.json"
    existing = [path for path in [*targets, metadata_path] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "仪表板聚合数据已存在；如需重建请使用 --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for target, frame in zip(targets, outputs.values()):
        frame.to_parquet(target, index=False)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [str(path.resolve()) for path in source_files],
        "input_rows": result.input_rows,
        "analysis_rows": result.analysis_rows,
        "grid_crs": f"EPSG:{UTM18N_EPSG}",
        "grid_size_m": result.grid_size_m,
        "date_range": {
            "start": result.daily_user_summary["date"].min().isoformat(),
            "end": result.daily_user_summary["date"].max().isoformat(),
        },
        "definitions": {
            "station_activity": "observed departures + observed arrivals",
            "net_inflow": "observed arrivals - observed departures",
            "od_scope": "directed flows among fixed 2025 Top 10 activity stations; self-loops excluded",
            "distance": "valid Haversine straight-line displacement only",
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return [*targets, metadata_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="为 Capital Bikeshare Streamlit 仪表板准备2025聚合数据"
    )
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-month", default="202501")
    parser.add_argument("--end-month", default="202512")
    parser.add_argument("--station-stats", type=Path, default=DEFAULT_STATION_STATS)
    parser.add_argument("--top-nodes", type=Path, default=DEFAULT_TOP_NODES)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--grid-size-m", type=int, default=DEFAULT_GRID_SIZE_M)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = discover_cleaned_files(args.input_dir, args.start_month, args.end_month)
    result = prepare_dashboard_data(
        files,
        station_stats_path=args.station_stats,
        top_nodes_path=args.top_nodes,
        chunksize=args.chunksize,
        grid_size_m=args.grid_size_m,
    )
    outputs = save_dashboard_data(
        result,
        args.output_dir,
        source_files=files,
        overwrite=args.overwrite,
    )
    print("\nDashboard data preparation complete:")
    print(f"  rows: {result.analysis_rows:,}")
    print(f"  dates: {result.daily_user_summary['date'].nunique():,}")
    print(f"  station-day-user rows: {len(result.station_daily_user):,}")
    print(f"  grid-day-user rows: {len(result.grid_daily_user):,}")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
