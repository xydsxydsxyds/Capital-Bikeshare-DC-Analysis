"""Station, OD, and 500 m grid analysis for Capital Bikeshare trips."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

from .od_sankey import build_bipartite_od_sankey
from .paths import PROCESSED_DIR
from .temporal import discover_cleaned_files


WGS84_EPSG = 4326
UTM18N_EPSG = 32618
DEFAULT_GRID_SIZE_M = 500

REQUIRED_SPATIAL_COLUMNS = (
    "start_station_id",
    "start_station_name",
    "end_station_id",
    "end_station_name",
    "start_lat",
    "start_lon",
    "end_lat",
    "end_lon",
    "traffic_period",
    "has_valid_start_coords",
    "has_valid_end_coords",
)


@dataclass
class SpatialODResult:
    station_stats: pd.DataFrame
    top10_start_stations: pd.DataFrame
    top10_end_stations: pd.DataFrame
    top10_activity_nodes: pd.DataFrame
    top20_od_pairs: pd.DataFrame
    grid_stats: pd.DataFrame
    inference_summary: pd.DataFrame
    input_rows: int
    grid_size_m: int


def _add_series(left: pd.Series | None, right: pd.Series) -> pd.Series:
    if left is None:
        return right.astype("int64")
    return left.add(right, fill_value=0).astype("int64")


def _coerce_spatial_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_SPATIAL_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError("清洗数据缺少空间分析字段: " + ", ".join(missing))
    frame = raw.loc[:, list(REQUIRED_SPATIAL_COLUMNS)].copy()
    for column in (
        "start_station_id",
        "start_station_name",
        "end_station_id",
        "end_station_name",
        "traffic_period",
    ):
        values = frame[column].astype("string").str.strip()
        frame[column] = values.mask(values.eq(""), pd.NA)
    for column in ("start_lat", "start_lon", "end_lat", "end_lon"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("has_valid_start_coords", "has_valid_end_coords"):
        frame[column] = frame[column].fillna(False).astype(bool)
    return frame


def _read_spatial_parquet_chunks(
    files: list[Path], chunksize: int
) -> Iterator[pd.DataFrame]:
    import pyarrow.parquet as pq

    for path in files:
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        missing = sorted(set(REQUIRED_SPATIAL_COLUMNS) - available)
        if missing:
            raise ValueError(f"{path.name} 缺少字段: {', '.join(missing)}")
        for batch in parquet_file.iter_batches(
            batch_size=chunksize, columns=list(REQUIRED_SPATIAL_COLUMNS)
        ):
            yield batch.to_pandas()


def _station_counts(
    frame: pd.DataFrame, endpoint: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    station_id = f"{endpoint}_station_id"
    station_name = f"{endpoint}_station_name"
    latitude = f"{endpoint}_lat"
    longitude = f"{endpoint}_lon"
    valid_coordinate = f"has_valid_{endpoint}_coords"

    valid_station = frame[station_id].notna() & frame[station_name].notna()
    station_rows = frame.loc[valid_station]
    counts = station_rows.groupby(station_id, observed=True).size()
    counts.index.name = "station_id"

    names = station_rows.groupby(
        [station_id, station_name], observed=True
    ).size()
    names.index = names.index.set_names(["station_id", "station_name"])

    valid_reference = valid_station & frame[valid_coordinate]
    coordinate_rows = frame.loc[
        valid_reference, [station_id, latitude, longitude]
    ].copy()
    # One-millionth of a degree is roughly 0.1 m and sharply reduces repeated
    # coordinate variants without affecting station-level precision.
    coordinate_rows[latitude] = coordinate_rows[latitude].round(6)
    coordinate_rows[longitude] = coordinate_rows[longitude].round(6)
    coordinates = coordinate_rows.groupby(
        [station_id, latitude, longitude], observed=True
    ).size()
    coordinates.index = coordinates.index.set_names(
        ["station_id", "latitude", "longitude"]
    )
    return counts, names, coordinates


def _grid_counts(
    frame: pd.DataFrame,
    endpoint: str,
    transformer: Transformer,
    grid_size_m: int,
    extra_mask: pd.Series | None = None,
) -> pd.Series:
    valid_column = f"has_valid_{endpoint}_coords"
    mask = frame[valid_column].copy()
    if extra_mask is not None:
        mask &= extra_mask
    selected = frame.loc[mask]
    if selected.empty:
        index = pd.MultiIndex.from_arrays([[], []], names=["grid_x", "grid_y"])
        return pd.Series([], index=index, dtype="int64")

    eastings, northings = transformer.transform(
        selected[f"{endpoint}_lon"].to_numpy(dtype="float64").tolist(),
        selected[f"{endpoint}_lat"].to_numpy(dtype="float64").tolist(),
    )
    grid_x = np.floor(np.asarray(eastings) / grid_size_m).astype("int32")
    grid_y = np.floor(np.asarray(northings) / grid_size_m).astype("int32")
    grid_frame = pd.DataFrame({"grid_x": grid_x, "grid_y": grid_y})
    counts = grid_frame.groupby(["grid_x", "grid_y"], observed=True).size()
    counts.index = counts.index.set_names(["grid_x", "grid_y"])
    return counts


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    threshold = sorted_weights.sum() / 2.0
    position = int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left"))
    return float(sorted_values[min(position, len(sorted_values) - 1)])


def _build_station_table(
    start_counts: pd.Series,
    end_counts: pd.Series,
    name_counts: pd.Series,
    coordinate_counts: pd.Series,
) -> pd.DataFrame:
    station_index = start_counts.index.union(end_counts.index)
    station = pd.DataFrame(index=station_index)
    station.index.name = "station_id"
    station["observed_departures"] = start_counts.reindex(
        station_index, fill_value=0
    ).astype("int64")
    station["observed_arrivals"] = end_counts.reindex(
        station_index, fill_value=0
    ).astype("int64")

    names = name_counts.rename("count").reset_index()
    names = names.sort_values(
        ["station_id", "count", "station_name"],
        ascending=[True, False, True],
    )
    preferred_names = names.drop_duplicates("station_id").set_index("station_id")[
        "station_name"
    ]
    station["station_name"] = preferred_names.reindex(station_index)

    coordinates = coordinate_counts.rename("count").reset_index()
    reference_rows: list[dict[str, object]] = []
    for station_id, group in coordinates.groupby("station_id", observed=True):
        weights = group["count"].to_numpy(dtype="float64")
        reference_rows.append(
            {
                "station_id": station_id,
                "station_lat": _weighted_median(
                    group["latitude"].to_numpy(dtype="float64"), weights
                ),
                "station_lon": _weighted_median(
                    group["longitude"].to_numpy(dtype="float64"), weights
                ),
            }
        )
    reference = pd.DataFrame(reference_rows).set_index("station_id")
    station = station.join(reference, how="left")
    station["total_activity"] = (
        station["observed_departures"] + station["observed_arrivals"]
    )
    station["net_inflow"] = (
        station["observed_arrivals"] - station["observed_departures"]
    )
    return station.reset_index()


def _build_grid_table(
    count_series: dict[str, pd.Series],
    inverse_transformer: Transformer,
    grid_size_m: int,
) -> pd.DataFrame:
    grid_index: pd.MultiIndex | None = None
    for series in count_series.values():
        grid_index = series.index if grid_index is None else grid_index.union(series.index)
    if grid_index is None or len(grid_index) == 0:
        raise ValueError("没有有效坐标可构建网格")

    grid = pd.DataFrame(index=grid_index.sort_values())
    for name, series in count_series.items():
        grid[name] = series.reindex(grid.index, fill_value=0).astype("int64")

    for column in (
        "departures",
        "arrivals",
        "morning_departures",
        "morning_arrivals",
        "evening_departures",
        "evening_arrivals",
    ):
        if column not in grid:
            grid[column] = 0

    grid["total_activity"] = grid["departures"] + grid["arrivals"]
    grid["net_inflow"] = grid["arrivals"] - grid["departures"]
    grid["morning_net_inflow"] = (
        grid["morning_arrivals"] - grid["morning_departures"]
    )
    grid["evening_net_inflow"] = (
        grid["evening_arrivals"] - grid["evening_departures"]
    )
    grid = grid.reset_index()
    grid["grid_id"] = (
        "E"
        + grid["grid_x"].astype(str)
        + "_N"
        + grid["grid_y"].astype(str)
    )
    grid["x_min_m"] = grid["grid_x"] * grid_size_m
    grid["y_min_m"] = grid["grid_y"] * grid_size_m
    grid["center_easting_m"] = grid["x_min_m"] + grid_size_m / 2.0
    grid["center_northing_m"] = grid["y_min_m"] + grid_size_m / 2.0
    center_lon, center_lat = inverse_transformer.transform(
        grid["center_easting_m"].to_numpy(),
        grid["center_northing_m"].to_numpy(),
    )
    grid["center_lat"] = center_lat
    grid["center_lon"] = center_lon
    grid["activity_rank"] = grid["total_activity"].rank(
        method="min", ascending=False
    ).astype("int64")
    preferred = [
        "grid_id",
        "grid_x",
        "grid_y",
        "x_min_m",
        "y_min_m",
        "center_easting_m",
        "center_northing_m",
        "center_lat",
        "center_lon",
        "departures",
        "arrivals",
        "total_activity",
        "net_inflow",
        "morning_departures",
        "morning_arrivals",
        "morning_net_inflow",
        "evening_departures",
        "evening_arrivals",
        "evening_net_inflow",
        "activity_rank",
    ]
    return grid[preferred].sort_values("total_activity", ascending=False).reset_index(
        drop=True
    )


def analyze_spatial_chunks(
    chunk_factory: Callable[[], Iterable[pd.DataFrame]],
    *,
    grid_size_m: int = DEFAULT_GRID_SIZE_M,
    inference_distance_m: float = 100.0,
    sensitivity_distance_m: float = 150.0,
    minimum_margin_m: float = 25.0,
) -> SpatialODResult:
    """Run two-pass station, OD, grid, and nearest-station analysis."""

    if grid_size_m <= 0:
        raise ValueError("grid_size_m 必须大于 0")
    if not 0 < inference_distance_m <= sensitivity_distance_m:
        raise ValueError("推断距离阈值必须为正，且主阈值不能超过敏感性阈值")

    forward = Transformer.from_crs(WGS84_EPSG, UTM18N_EPSG, always_xy=True)
    inverse = Transformer.from_crs(UTM18N_EPSG, WGS84_EPSG, always_xy=True)

    start_counts: pd.Series | None = None
    end_counts: pd.Series | None = None
    name_counts: pd.Series | None = None
    coordinate_counts: pd.Series | None = None
    grid_counts: dict[str, pd.Series | None] = {
        "departures": None,
        "arrivals": None,
        "morning_departures": None,
        "morning_arrivals": None,
        "evening_departures": None,
        "evening_arrivals": None,
    }
    input_rows = 0

    for raw_chunk in chunk_factory():
        frame = _coerce_spatial_chunk(raw_chunk)
        input_rows += len(frame)

        chunk_start, start_names, start_coordinates = _station_counts(frame, "start")
        chunk_end, end_names, end_coordinates = _station_counts(frame, "end")
        start_counts = _add_series(start_counts, chunk_start)
        end_counts = _add_series(end_counts, chunk_end)
        name_counts = _add_series(name_counts, start_names)
        name_counts = _add_series(name_counts, end_names)
        coordinate_counts = _add_series(coordinate_counts, start_coordinates)
        coordinate_counts = _add_series(coordinate_counts, end_coordinates)

        period_masks = {
            "": None,
            "morning_": frame["traffic_period"].eq("morning_peak"),
            "evening_": frame["traffic_period"].eq("evening_peak"),
        }
        for period_prefix, period_mask in period_masks.items():
            for endpoint, output_name in (
                ("start", f"{period_prefix}departures"),
                ("end", f"{period_prefix}arrivals"),
            ):
                chunk_grid = _grid_counts(
                    frame,
                    endpoint,
                    forward,
                    grid_size_m,
                    extra_mask=period_mask,
                )
                grid_counts[output_name] = _add_series(
                    grid_counts[output_name], chunk_grid
                )

    if input_rows == 0 or start_counts is None or end_counts is None:
        raise ValueError("没有可用于空间分析的数据")
    assert name_counts is not None
    assert coordinate_counts is not None

    station = _build_station_table(
        start_counts, end_counts, name_counts, coordinate_counts
    )
    top_nodes = station.nlargest(10, "total_activity").copy()
    top_node_ids = set(top_nodes["station_id"].astype(str))

    reference = station.dropna(subset=["station_lat", "station_lon"]).copy()
    reference_easting, reference_northing = forward.transform(
        reference["station_lon"].to_numpy().tolist(),
        reference["station_lat"].to_numpy().tolist(),
    )
    reference_points = np.column_stack([reference_easting, reference_northing])
    if len(reference_points) < 2:
        raise ValueError("有效站点参考坐标不足，无法执行最近站点推断")
    tree = cKDTree(reference_points)
    reference_ids = reference["station_id"].astype(str).to_numpy()

    inferred_100: pd.Series | None = None
    inferred_150: pd.Series | None = None
    od_counts: pd.Series | None = None
    nearest_distance_parts: list[np.ndarray] = []
    missing_station_candidates = 0
    within_100_count = 0
    within_150_count = 0
    high_confidence_100_count = 0
    high_confidence_150_count = 0

    for raw_chunk in chunk_factory():
        frame = _coerce_spatial_chunk(raw_chunk)
        observed_start = (
            frame["start_station_id"].notna()
            & frame["start_station_name"].notna()
        )
        inference_mask = ~observed_start & frame["has_valid_start_coords"]
        candidates = frame.loc[inference_mask]
        missing_station_candidates += len(candidates)
        if not candidates.empty:
            eastings, northings = forward.transform(
                candidates["start_lon"].to_numpy().tolist(),
                candidates["start_lat"].to_numpy().tolist(),
            )
            distances, positions = tree.query(
                np.column_stack([eastings, northings]), k=2, workers=-1
            )
            nearest = distances[:, 0]
            margin = distances[:, 1] - distances[:, 0]
            nearest_ids = reference_ids[positions[:, 0]]
            nearest_distance_parts.append(nearest.astype("float64"))
            within_100_count += int((nearest <= inference_distance_m).sum())
            within_150_count += int((nearest <= sensitivity_distance_m).sum())

            confidence_100 = (
                (nearest <= inference_distance_m) & (margin >= minimum_margin_m)
            )
            confidence_150 = (
                (nearest <= sensitivity_distance_m) & (margin >= minimum_margin_m)
            )
            high_confidence_100_count += int(confidence_100.sum())
            high_confidence_150_count += int(confidence_150.sum())
            if confidence_100.any():
                counts = pd.Series(nearest_ids[confidence_100]).value_counts()
                counts.index.name = "station_id"
                inferred_100 = _add_series(inferred_100, counts)
            if confidence_150.any():
                counts = pd.Series(nearest_ids[confidence_150]).value_counts()
                counts.index.name = "station_id"
                inferred_150 = _add_series(inferred_150, counts)

        valid_od = (
            frame["start_station_id"].isin(top_node_ids)
            & frame["end_station_id"].isin(top_node_ids)
            & frame["start_station_name"].notna()
            & frame["end_station_name"].notna()
            & frame["start_station_id"].ne(frame["end_station_id"])
        )
        od_rows = frame.loc[valid_od, ["start_station_id", "end_station_id"]]
        chunk_od = od_rows.groupby(
            ["start_station_id", "end_station_id"], observed=True
        ).size()
        chunk_od.index = chunk_od.index.set_names(
            ["source_station_id", "target_station_id"]
        )
        od_counts = _add_series(od_counts, chunk_od)

    if inferred_100 is None:
        inferred_100 = pd.Series(dtype="int64")
    if inferred_150 is None:
        inferred_150 = pd.Series(dtype="int64")
    station = station.set_index("station_id")
    station["inferred_departures_100m"] = inferred_100.reindex(
        station.index, fill_value=0
    ).astype("int64")
    station["inferred_departures_150m_sensitivity"] = inferred_150.reindex(
        station.index, fill_value=0
    ).astype("int64")
    station["observed_plus_inferred_100m"] = (
        station["observed_departures"] + station["inferred_departures_100m"]
    )
    station["observed_plus_inferred_150m_sensitivity"] = (
        station["observed_departures"]
        + station["inferred_departures_150m_sensitivity"]
    )
    station["observed_departure_rank"] = station["observed_departures"].rank(
        method="min", ascending=False
    ).astype("int64")
    station["observed_plus_100m_rank"] = station[
        "observed_plus_inferred_100m"
    ].rank(method="min", ascending=False).astype("int64")
    station["sensitivity_150m_rank"] = station[
        "observed_plus_inferred_150m_sensitivity"
    ].rank(method="min", ascending=False).astype("int64")
    station = station.reset_index()

    name_lookup = station.set_index("station_id")["station_name"].to_dict()
    if od_counts is None:
        od_table = pd.DataFrame(
            columns=["source_station_id", "target_station_id", "trip_count"]
        )
    else:
        od_table = (
            od_counts.rename("trip_count")
            .reset_index()
            .sort_values("trip_count", ascending=False)
            .head(20)
        )
    od_table["source_station_name"] = od_table["source_station_id"].map(name_lookup)
    od_table["target_station_name"] = od_table["target_station_id"].map(name_lookup)
    od_table = od_table[
        [
            "source_station_id",
            "source_station_name",
            "target_station_id",
            "target_station_name",
            "trip_count",
        ]
    ].reset_index(drop=True)

    nearest_distances = (
        np.concatenate(nearest_distance_parts)
        if nearest_distance_parts
        else np.array([], dtype="float64")
    )
    inference_summary = pd.DataFrame(
        [
            {
                "missing_start_station_candidates": missing_station_candidates,
                "nearest_distance_median_m": (
                    float(np.median(nearest_distances))
                    if len(nearest_distances)
                    else math.nan
                ),
                "nearest_distance_p75_m": (
                    float(np.percentile(nearest_distances, 75))
                    if len(nearest_distances)
                    else math.nan
                ),
                "nearest_distance_p90_m": (
                    float(np.percentile(nearest_distances, 90))
                    if len(nearest_distances)
                    else math.nan
                ),
                "within_100m_rows": within_100_count,
                "high_confidence_100m_rows": high_confidence_100_count,
                "within_150m_rows": within_150_count,
                "high_confidence_150m_rows": high_confidence_150_count,
                "minimum_second_nearest_margin_m": minimum_margin_m,
            }
        ]
    )

    complete_grid_counts = {
        name: series if series is not None else pd.Series(dtype="int64")
        for name, series in grid_counts.items()
    }
    grid = _build_grid_table(complete_grid_counts, inverse, grid_size_m)
    return SpatialODResult(
        station_stats=station.sort_values(
            "total_activity", ascending=False
        ).reset_index(drop=True),
        top10_start_stations=station.nlargest(10, "observed_departures").reset_index(
            drop=True
        ),
        top10_end_stations=station.nlargest(10, "observed_arrivals").reset_index(
            drop=True
        ),
        top10_activity_nodes=station.nlargest(10, "total_activity").reset_index(
            drop=True
        ),
        top20_od_pairs=od_table,
        grid_stats=grid,
        inference_summary=inference_summary,
        input_rows=input_rows,
        grid_size_m=grid_size_m,
    )


def analyze_spatial_files(
    files: list[Path],
    *,
    chunksize: int = 100_000,
    grid_size_m: int = DEFAULT_GRID_SIZE_M,
) -> SpatialODResult:
    if not files:
        raise ValueError("没有找到清洗后的月度 Parquet 文件")
    non_parquet = [path.name for path in files if path.suffix != ".parquet"]
    if non_parquet:
        raise ValueError(
            "空间分析要求 Parquet 输入；发现非 Parquet 文件: "
            + ", ".join(non_parquet)
        )
    if chunksize <= 0:
        raise ValueError("chunksize 必须大于 0")

    pass_number = 0

    def factory() -> Iterator[pd.DataFrame]:
        nonlocal pass_number
        pass_number += 1
        print(f"Spatial analysis pass {pass_number}/2")
        yield from _read_spatial_parquet_chunks(files, chunksize)

    return analyze_spatial_chunks(factory, grid_size_m=grid_size_m)


def _shorten_label(value: str, limit: int = 30) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _prepare_matplotlib(output_dir: Path) -> None:
    cache = Path(tempfile.gettempdir()) / "capital_bikeshare_mpl_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def create_spatial_outputs(
    result: SpatialODResult,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Export static figures, interactive maps, Sankey, tables, and metadata."""

    _prepare_matplotlib(output_dir)
    import matplotlib

    matplotlib.use("Agg")
    import branca.colormap as cm
    import folium
    import matplotlib.pyplot as plt
    from folium.plugins import HeatMap
    from matplotlib import colors
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    figure_dir = output_dir / "figures"
    map_dir = output_dir / "maps"
    table_dir = output_dir / "tables"
    for directory in (figure_dir, map_dir, table_dir):
        directory.mkdir(parents=True, exist_ok=True)

    targets = [
        figure_dir / "01_station_top10.png",
        figure_dir / "02_station_activity_map.png",
        figure_dir / "03_top10_od_matrix.png",
        figure_dir / "04_grid_density_500m.png",
        figure_dir / "05_peak_net_flow_500m.png",
        map_dir / "station_heatmap.html",
        map_dir / "grid_density_map.html",
        map_dir / "top20_od_sankey.html",
        table_dir / "station_stats.csv",
        table_dir / "top10_start_stations.csv",
        table_dir / "top10_end_stations.csv",
        table_dir / "top10_activity_nodes.csv",
        table_dir / "top20_od_pairs.csv",
        table_dir / "grid_stats_500m.csv",
        table_dir / "nearest_station_inference_summary.csv",
        output_dir / "spatial_analysis_metadata.json",
        figure_dir / "06_station_inference_comparison.png",
        table_dir / "top10_start_observed_plus_100m.csv",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "输出已存在（使用 --overwrite 替换）: "
            + ", ".join(path.name for path in existing)
        )

    plt.style.use("seaborn-v0_8-whitegrid")
    blue = "#2F6BFF"
    orange = "#FF8A34"

    # 1. Top observed start and end stations.
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    panels = [
        (result.top10_start_stations, "observed_departures", "Top 10 Start Stations", blue),
        (result.top10_end_stations, "observed_arrivals", "Top 10 End Stations", orange),
    ]
    for ax, (data, metric, title, color) in zip(axes, panels):
        plot_data = data.sort_values(metric)
        labels = [
            _shorten_label(str(name), 38) for name in plot_data["station_name"]
        ]
        bars = ax.barh(labels, plot_data[metric], color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Trips")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.1f}K"))
        for bar, station_id in zip(bars, plot_data["station_id"]):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"  ID {station_id}",
                va="center",
                fontsize=8,
            )
    fig.suptitle("Observed Station Demand Rankings", fontsize=16)
    fig.tight_layout()
    fig.savefig(targets[0], dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2. Station activity and net-flow locations.
    station_map_data = result.station_stats.dropna(
        subset=["station_lat", "station_lon"]
    )
    fig, ax = plt.subplots(figsize=(10, 9))
    sizes = 8 + 150 * np.sqrt(
        station_map_data["total_activity"] / station_map_data["total_activity"].max()
    )
    limit = float(np.percentile(np.abs(station_map_data["net_inflow"]), 98))
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    scatter = ax.scatter(
        station_map_data["station_lon"],
        station_map_data["station_lat"],
        s=sizes,
        c=station_map_data["net_inflow"].clip(-limit, limit),
        cmap="RdBu_r",
        norm=norm,
        alpha=0.72,
        linewidths=0.2,
        edgecolors="white",
    )
    for rank, row in result.top10_activity_nodes.iterrows():
        ax.annotate(
            str(rank + 1),
            (row["station_lon"], row["station_lat"]),
            fontsize=7,
            ha="center",
            va="center",
            color="black",
        )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.75)
    colorbar.set_label("Net inflow (arrivals - departures)")
    ax.set_title("Station Activity and Net Flow\n(marker size = total activity; labels = top 10)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(targets[1], dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3. OD matrix among the selected top 10 activity nodes.
    nodes = result.top10_activity_nodes["station_id"].astype(str).tolist()
    node_names = result.top10_activity_nodes.set_index("station_id")[
        "station_name"
    ].to_dict()
    matrix = pd.DataFrame(0, index=nodes, columns=nodes, dtype="int64")
    for row in result.top20_od_pairs.itertuples(index=False):
        matrix.loc[str(row.source_station_id), str(row.target_station_id)] = int(
            row.trip_count
        )
    fig, ax = plt.subplots(figsize=(10, 8.5))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues")
    labels = [_shorten_label(str(node_names[node]), 22) for node in nodes]
    ax.set_xticks(range(len(nodes)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(nodes)), labels=labels)
    ax.set_xlabel("Destination")
    ax.set_ylabel("Origin")
    ax.set_title("Top 20 Directed OD Flows Among Top 10 Activity Stations")
    for row_index in range(len(nodes)):
        for column_index in range(len(nodes)):
            value = int(matrix.iloc[row_index, column_index])
            if value:
                ax.text(
                    column_index,
                    row_index,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > matrix.to_numpy().max() * 0.55 else "black",
                )
    fig.colorbar(image, ax=ax, shrink=0.78, label="Trips")
    fig.tight_layout()
    fig.savefig(targets[2], dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 4. Exact 500 m UTM grid density.
    grid = result.grid_stats
    patches = [
        Rectangle((row.x_min_m, row.y_min_m), result.grid_size_m, result.grid_size_m)
        for row in grid.itertuples(index=False)
    ]
    positive = grid.loc[grid["total_activity"].gt(0), "total_activity"]
    collection = PatchCollection(
        patches,
        cmap="YlOrRd",
        norm=colors.LogNorm(vmin=max(1, int(positive.min())), vmax=int(positive.max())),
        linewidth=0,
    )
    collection.set_array(grid["total_activity"].to_numpy())
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1000:.0f}"))
    ax.set_xlabel("UTM easting (km)")
    ax.set_ylabel("UTM northing (km)")
    ax.set_title("500 m Grid Activity Density (Departures + Arrivals)")
    fig.colorbar(collection, ax=ax, shrink=0.76, label="Total activity (log scale)")
    for rank, row in grid.head(10).iterrows():
        ax.text(
            row["center_easting_m"],
            row["center_northing_m"],
            str(rank + 1),
            ha="center",
            va="center",
            fontsize=7,
            color="black",
        )
    fig.tight_layout()
    fig.savefig(targets[3], dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 5. Morning/evening net inflow comparison.
    peak_columns = [
        ("morning_net_inflow", "Morning Peak (07:00-09:59)"),
        ("evening_net_inflow", "Evening Peak (16:00-18:59)"),
    ]
    peak_limit = float(
        np.percentile(
            np.abs(grid[[column for column, _ in peak_columns]].to_numpy()).ravel(),
            99,
        )
    )
    peak_limit = max(1.0, peak_limit)
    peak_norm = colors.TwoSlopeNorm(
        vmin=-peak_limit, vcenter=0.0, vmax=peak_limit
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    last_collection = None
    for ax, (column, title) in zip(axes, peak_columns):
        patches = [
            Rectangle(
                (row.x_min_m, row.y_min_m), result.grid_size_m, result.grid_size_m
            )
            for row in grid.itertuples(index=False)
        ]
        last_collection = PatchCollection(
            patches,
            cmap="RdBu_r",
            norm=peak_norm,
            linewidth=0,
        )
        last_collection.set_array(grid[column].clip(-peak_limit, peak_limit).to_numpy())
        ax.add_collection(last_collection)
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1000:.0f}"))
        ax.set_xlabel("UTM easting (km)")
    axes[0].set_ylabel("UTM northing (km)")
    fig.suptitle("500 m Grid Net Inflow (Arrivals - Departures)", fontsize=15)
    assert last_collection is not None
    fig.colorbar(
        last_collection,
        ax=axes,
        shrink=0.75,
        label="Net inflow (clipped at 99th percentile)",
    )
    fig.savefig(targets[4], dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Interactive station heat map.
    map_center = [
        float(station_map_data["station_lat"].median()),
        float(station_map_data["station_lon"].median()),
    ]
    station_map = folium.Map(
        location=map_center, zoom_start=12, tiles="CartoDB positron"
    )
    HeatMap(
        station_map_data[
            ["station_lat", "station_lon", "observed_departures"]
        ].values.tolist(),
        name="Observed departures",
        radius=18,
        blur=14,
        min_opacity=0.25,
    ).add_to(station_map)
    HeatMap(
        station_map_data[
            ["station_lat", "station_lon", "observed_arrivals"]
        ].values.tolist(),
        name="Observed arrivals",
        radius=18,
        blur=14,
        min_opacity=0.25,
        show=False,
        gradient={0.2: "#fee8c8", 0.5: "#fdbb84", 0.8: "#e34a33", 1: "#b30000"},
    ).add_to(station_map)
    top_group = folium.FeatureGroup(name="Top 10 activity stations", show=True)
    for rank, row in result.top10_activity_nodes.iterrows():
        folium.CircleMarker(
            location=[row["station_lat"], row["station_lon"]],
            radius=7,
            color="#25324A",
            fill=True,
            fill_opacity=0.9,
            tooltip=(
                f"#{rank + 1} {row['station_name']}<br>"
                f"Departures: {row['observed_departures']:,}<br>"
                f"Arrivals: {row['observed_arrivals']:,}"
            ),
        ).add_to(top_group)
    top_group.add_to(station_map)
    folium.LayerControl(collapsed=False).add_to(station_map)
    station_map.save(targets[5])

    # Interactive 500 m density polygons.
    grid_map = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB positron")
    cap = float(np.percentile(grid["total_activity"], 99))
    colormap = cm.linear.YlOrRd_09.scale(0, cap)
    forward_corners = Transformer.from_crs(
        UTM18N_EPSG, WGS84_EPSG, always_xy=True
    )
    for row in grid.itertuples(index=False):
        x0, y0 = row.x_min_m, row.y_min_m
        x1, y1 = x0 + result.grid_size_m, y0 + result.grid_size_m
        corner_lon, corner_lat = forward_corners.transform(
            [x0, x1, x1, x0], [y0, y0, y1, y1]
        )
        locations = list(zip(corner_lat, corner_lon))
        folium.Polygon(
            locations=locations,
            color=None,
            weight=0,
            fill=True,
            fill_color=colormap(min(float(row.total_activity), cap)),
            fill_opacity=0.68,
            tooltip=(
                f"{row.grid_id}<br>Departures: {row.departures:,}<br>"
                f"Arrivals: {row.arrivals:,}<br>Net inflow: {row.net_inflow:,}"
            ),
        ).add_to(grid_map)
    colormap.caption = "Grid activity (departures + arrivals; color capped at p99)"
    colormap.add_to(grid_map)
    grid_map.save(targets[6])

    # Interactive Top 20 directed OD Sankey.
    sankey = build_bipartite_od_sankey(
        result.top20_od_pairs,
        result.top10_activity_nodes,
        title="Top 20 Directed OD Flows Among Top 10 Activity Stations",
    )
    sankey.update_layout(
        width=1200,
    )
    sankey.write_html(targets[7], include_plotlyjs=True, full_html=True)

    result.station_stats.to_csv(targets[8], index=False)
    result.top10_start_stations.to_csv(targets[9], index=False)
    result.top10_end_stations.to_csv(targets[10], index=False)
    result.top10_activity_nodes.to_csv(targets[11], index=False)
    result.top20_od_pairs.to_csv(targets[12], index=False)
    result.grid_stats.to_csv(targets[13], index=False)
    result.inference_summary.to_csv(targets[14], index=False)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": result.input_rows,
        "station_count": int(len(result.station_stats)),
        "grid_count": int(len(result.grid_stats)),
        "grid_crs": f"EPSG:{UTM18N_EPSG}",
        "grid_size_m": result.grid_size_m,
        "net_inflow_definition": "arrivals - departures",
        "top_node_definition": "observed departures + observed arrivals",
        "od_definition": "observed start/end station fields only; self-loops excluded",
        "station_inference_definition": (
            "nearest station within 100 m and at least 25 m closer than second nearest; "
            "150 m reported only as sensitivity"
        ),
    }
    targets[15].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Explicitly separate observed departures from the 100 m service-area
    # extension required by the project plan. The extension is stacked rather
    # than presented as if it were observed station demand.
    observed_top = result.station_stats.nlargest(10, "observed_departures").sort_values(
        "observed_departures"
    )
    extended_top = result.station_stats.nlargest(
        10, "observed_plus_inferred_100m"
    ).sort_values("observed_plus_inferred_100m")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].barh(
        [_shorten_label(str(value), 38) for value in observed_top["station_name"]],
        observed_top["observed_departures"],
        color=blue,
        alpha=0.86,
    )
    axes[0].set_title("Observed Station Departures")
    axes[0].set_xlabel("Trips")
    axes[1].barh(
        [_shorten_label(str(value), 38) for value in extended_top["station_name"]],
        extended_top["observed_departures"],
        color=blue,
        alpha=0.86,
        label="Observed",
    )
    axes[1].barh(
        [_shorten_label(str(value), 38) for value in extended_top["station_name"]],
        extended_top["inferred_departures_100m"],
        left=extended_top["observed_departures"],
        color=orange,
        alpha=0.88,
        label="Inferred within 100 m",
    )
    axes[1].set_title("Observed + High-confidence 100 m Inference")
    axes[1].set_xlabel("Trips")
    axes[1].legend(loc="lower right")
    for ax in axes:
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x / 1000:.1f}K")
        )
    fig.suptitle(
        "Start-station Ranking Sensitivity (inferred demand kept separate)",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(targets[16], dpi=180, bbox_inches="tight")
    plt.close(fig)
    result.station_stats.nlargest(10, "observed_plus_inferred_100m").to_csv(
        targets[17], index=False
    )
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capital Bikeshare 站点、OD 与 500 米网格空间分析"
    )
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spatial_od"))
    parser.add_argument("--start-month", help="清洗文件起始月份 YYYYMM（含）")
    parser.add_argument("--end-month", help="清洗文件结束月份 YYYYMM（含）")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--grid-size-m", type=int, default=DEFAULT_GRID_SIZE_M)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = discover_cleaned_files(
        args.input_dir,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    result = analyze_spatial_files(
        files,
        chunksize=args.chunksize,
        grid_size_m=args.grid_size_m,
    )
    outputs = create_spatial_outputs(
        result, args.output_dir, overwrite=args.overwrite
    )
    print("\nSpatial and OD analysis complete:")
    print(f"  rows: {result.input_rows:,}")
    print(f"  stations: {len(result.station_stats):,}")
    print(f"  grids: {len(result.grid_stats):,}")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
