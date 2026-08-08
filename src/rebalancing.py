"""Grid-level vehicle rebalancing simulation from one live GBFS snapshot."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from .grid import DEFAULT_GRID_SIZE_M, lonlat_to_grid
from .paths import PROCESSED_DIR
from .realtime_gbfs import DEFAULT_SNAPSHOT_DIR, load_latest_snapshot
from .temporal import discover_cleaned_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rebalancing"
DEFAULT_PROFILE_PATH = DEFAULT_OUTPUT_DIR / "tables" / "historical_grid_flow_profiles.csv"
DEFAULT_TOP_GRIDS = (
    PROJECT_ROOT / "outputs" / "forecasting" / "tables" / "top10_forecast_grids.csv"
)
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "forecasting"
    / "tables"
    / "forecast_predictions_2026.parquet"
)

FLOW_COLUMNS = (
    "start_time",
    "end_time",
    "start_lat",
    "start_lon",
    "end_lat",
    "end_lon",
    "has_valid_start_coords",
    "has_valid_end_coords",
)


@dataclass(frozen=True)
class SimulationConfig:
    horizon_hours: int = 6
    truck_count: int = 3
    truck_capacity: int = 10
    max_relocated_bikes_per_hour: int = 30
    low_inventory_ratio: float = 0.25
    high_inventory_ratio: float = 0.75
    demand_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.horizon_hours <= 0:
            raise ValueError("horizon_hours 必须大于 0")
        if self.truck_count < 0 or self.truck_capacity <= 0:
            raise ValueError("truck_count不能为负且truck_capacity必须大于0")
        if self.max_relocated_bikes_per_hour < 0:
            raise ValueError("max_relocated_bikes_per_hour不能为负")
        if not 0 <= self.low_inventory_ratio < self.high_inventory_ratio <= 1:
            raise ValueError("库存率阈值必须满足 0 <= low < high <= 1")
        if self.demand_multiplier <= 0:
            raise ValueError("demand_multiplier 必须大于 0")


@dataclass
class RebalancingResult:
    baseline_trajectory: pd.DataFrame
    strategy_trajectory: pd.DataFrame
    dispatch_plan: pd.DataFrame
    metrics: pd.DataFrame
    initial_inventory: pd.DataFrame
    simulation_start: pd.Timestamp
    simulation_end: pd.Timestamp
    demand_mode: str
    snapshot_path: str | None = None


def _read_flow_chunks(files: Sequence[Path], chunksize: int) -> Iterator[pd.DataFrame]:
    import pyarrow.parquet as pq

    for path in files:
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        missing = sorted(set(FLOW_COLUMNS) - available)
        if missing:
            raise ValueError(f"{path.name} 缺少调度画像字段: {', '.join(missing)}")
        for batch in parquet_file.iter_batches(
            batch_size=chunksize, columns=list(FLOW_COLUMNS)
        ):
            yield batch.to_pandas()


def _add_counts(left: pd.Series | None, right: pd.Series) -> pd.Series:
    if left is None:
        return right.astype("int64")
    return left.add(right, fill_value=0).astype("int64")


def build_historical_grid_flow_profiles(
    files: Sequence[Path],
    top_grids: pd.DataFrame,
    *,
    profile_start: str | pd.Timestamp = "2025-01-01",
    profile_end: str | pd.Timestamp = "2026-01-01",
    grid_size_m: int = DEFAULT_GRID_SIZE_M,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Create complete grid x weekday x hour mean departure/arrival profiles."""

    if not files:
        raise ValueError("没有清洗文件可构建调度需求画像")
    required = {"grid_id"}
    if not required.issubset(top_grids.columns):
        raise ValueError("Top网格表缺少grid_id")
    start = pd.Timestamp(profile_start)
    end = pd.Timestamp(profile_end)
    if start >= end:
        raise ValueError("profile_start必须早于profile_end")
    target_ids = set(top_grids["grid_id"].astype(str))
    totals: dict[str, pd.Series | None] = {"departures": None, "arrivals": None}

    for raw in _read_flow_chunks(files, chunksize):
        for endpoint, metric in (("start", "departures"), ("end", "arrivals")):
            timestamps = pd.to_datetime(raw[f"{endpoint}_time"], errors="coerce")
            latitude = pd.to_numeric(raw[f"{endpoint}_lat"], errors="coerce")
            longitude = pd.to_numeric(raw[f"{endpoint}_lon"], errors="coerce")
            valid = (
                raw[f"has_valid_{endpoint}_coords"].fillna(False).astype(bool)
                & timestamps.ge(start)
                & timestamps.lt(end)
                & latitude.notna()
                & longitude.notna()
            )
            if not valid.any():
                continue
            selected_time = timestamps.loc[valid].reset_index(drop=True)
            mapped = lonlat_to_grid(
                longitude.loc[valid].reset_index(drop=True),
                latitude.loc[valid].reset_index(drop=True),
                grid_size_m=grid_size_m,
            )
            events = pd.DataFrame(
                {
                    "grid_id": mapped["grid_id"],
                    "weekday": selected_time.dt.weekday.astype("int8"),
                    "hour": selected_time.dt.hour.astype("int8"),
                }
            )
            events = events.loc[events["grid_id"].isin(target_ids)]
            if events.empty:
                continue
            counts = events.groupby(
                ["grid_id", "weekday", "hour"], observed=True
            ).size()
            totals[metric] = _add_counts(totals[metric], counts)

    calendar = pd.date_range(start=start, end=end, freq="D", inclusive="left")
    weekday_occurrences = pd.Series(calendar.weekday).value_counts().to_dict()
    grid_order = top_grids.sort_values(
        "demand_rank" if "demand_rank" in top_grids else "grid_id"
    )["grid_id"].astype(str)
    complete_index = pd.MultiIndex.from_product(
        [grid_order, range(7), range(24)],
        names=["grid_id", "weekday", "hour"],
    )
    profile = pd.DataFrame(index=complete_index)
    for metric in ("departures", "arrivals"):
        counts = totals[metric]
        if counts is None:
            values = pd.Series(0, index=complete_index, dtype="int64")
        else:
            values = counts.reindex(complete_index, fill_value=0)
        denominators = complete_index.get_level_values("weekday").map(
            weekday_occurrences
        )
        profile[f"mean_{metric}"] = values.to_numpy(dtype="float64") / np.asarray(
            denominators, dtype="float64"
        )
    profile = profile.reset_index()
    profile["profile_start"] = start
    profile["profile_end"] = end
    return profile


def save_flow_profiles(
    profiles: pd.DataFrame,
    output_path: Path = DEFAULT_PROFILE_PATH,
    *,
    overwrite: bool = False,
) -> Path:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"画像输出已存在（使用 --overwrite 替换）: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output_path, index=False)
    return output_path


def snapshot_to_grid_inventory(
    snapshot: pd.DataFrame,
    top_grids: pd.DataFrame,
    *,
    grid_size_m: int = DEFAULT_GRID_SIZE_M,
) -> pd.DataFrame:
    required_snapshot = {
        "lat",
        "lon",
        "vehicles_available",
        "docks_available",
        "is_installed",
        "is_renting",
        "is_returning",
    }
    missing = sorted(required_snapshot - set(snapshot.columns))
    if missing:
        raise ValueError("实时快照缺少字段: " + ", ".join(missing))
    required_grids = {"grid_id", "center_lat", "center_lon"}
    missing_grids = sorted(required_grids - set(top_grids.columns))
    if missing_grids:
        raise ValueError("Top网格表缺少字段: " + ", ".join(missing_grids))

    frame = snapshot.copy().reset_index(drop=True)
    mapped = lonlat_to_grid(frame["lon"], frame["lat"], grid_size_m=grid_size_m)
    frame = pd.concat([frame, mapped], axis=1)
    frame = frame.loc[
        frame["is_installed"].fillna(False).astype(bool)
        & frame["grid_id"].isin(top_grids["grid_id"].astype(str))
    ].copy()
    frame["sim_bikes"] = np.where(
        frame["is_renting"].fillna(False), frame["vehicles_available"], 0
    )
    frame["sim_docks"] = np.where(
        frame["is_returning"].fillna(False), frame["docks_available"], 0
    )
    frame["sim_capacity"] = frame["sim_bikes"] + frame["sim_docks"]
    grouped = frame.groupby("grid_id", as_index=False).agg(
        initial_bikes=("sim_bikes", "sum"),
        capacity=("sim_capacity", "sum"),
        station_count=("station_id", "nunique") if "station_id" in frame else ("grid_id", "size"),
    )
    inventory = top_grids[
        ["grid_id", "center_lat", "center_lon"]
    ].copy()
    inventory["grid_id"] = inventory["grid_id"].astype(str)
    inventory = inventory.merge(grouped, on="grid_id", how="left")
    for column in ("initial_bikes", "capacity", "station_count"):
        inventory[column] = inventory[column].fillna(0)
    inventory["initial_bikes"] = inventory["initial_bikes"].astype("float64")
    inventory["capacity"] = inventory["capacity"].astype("float64")
    inventory["station_count"] = inventory["station_count"].astype("int32")
    inventory["initial_inventory_ratio"] = np.where(
        inventory["capacity"].gt(0),
        inventory["initial_bikes"] / inventory["capacity"],
        np.nan,
    )
    if not inventory["capacity"].gt(0).any():
        raise ValueError("实时快照在Top预测网格内没有可用站点容量")
    return inventory


def _local_naive_hour(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("America/New_York").tz_localize(None)
    return timestamp.floor("h")


def build_profile_demand(
    profiles: pd.DataFrame,
    grid_ids: Sequence[str],
    *,
    start_time: object,
    horizon_hours: int,
    demand_multiplier: float = 1.0,
) -> pd.DataFrame:
    start = _local_naive_hour(start_time)
    hours = pd.date_range(start=start, periods=horizon_hours, freq="h")
    index = pd.MultiIndex.from_product(
        [hours, list(grid_ids)], names=["timestamp", "grid_id"]
    )
    demand = index.to_frame(index=False)
    demand["weekday"] = demand["timestamp"].dt.weekday
    demand["hour"] = demand["timestamp"].dt.hour
    profile = profiles.copy()
    profile["grid_id"] = profile["grid_id"].astype(str)
    demand = demand.merge(
        profile[["grid_id", "weekday", "hour", "mean_departures", "mean_arrivals"]],
        on=["grid_id", "weekday", "hour"],
        how="left",
        validate="many_to_one",
    )
    if demand[["mean_departures", "mean_arrivals"]].isna().any().any():
        raise ValueError("历史流量画像未覆盖全部模拟网格×星期×小时")
    demand["forecast_departures"] = demand["mean_departures"] * demand_multiplier
    demand["realized_departures"] = demand["forecast_departures"]
    demand["forecast_arrivals"] = demand["mean_arrivals"] * demand_multiplier
    demand["realized_arrivals"] = demand["forecast_arrivals"]
    return demand.drop(columns=["mean_departures", "mean_arrivals"])


def build_backtest_demand(
    predictions: pd.DataFrame,
    profiles: pd.DataFrame,
    grid_ids: Sequence[str],
    *,
    start_time: object,
    horizon_hours: int,
    prediction_column: str = "hist_gradient_boosting_prediction",
    demand_multiplier: float = 1.0,
) -> pd.DataFrame:
    required = {"timestamp", "grid_id", "actual_demand", prediction_column}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError("预测表缺少字段: " + ", ".join(missing))
    start = _local_naive_hour(start_time)
    end = start + pd.Timedelta(hours=horizon_hours)
    selected = predictions.copy()
    selected["timestamp"] = pd.to_datetime(selected["timestamp"])
    selected["grid_id"] = selected["grid_id"].astype(str)
    selected = selected.loc[
        selected["timestamp"].ge(start)
        & selected["timestamp"].lt(end)
        & selected["grid_id"].isin(grid_ids),
        ["timestamp", "grid_id", "actual_demand", prediction_column],
    ]
    expected_rows = horizon_hours * len(grid_ids)
    if len(selected) != expected_rows:
        raise ValueError(
            f"回测预测期应有{expected_rows}行，实际{len(selected)}行；请检查时间范围"
        )
    selected["weekday"] = selected["timestamp"].dt.weekday
    selected["hour"] = selected["timestamp"].dt.hour
    arrivals = profiles[
        ["grid_id", "weekday", "hour", "mean_arrivals"]
    ].copy()
    arrivals["grid_id"] = arrivals["grid_id"].astype(str)
    selected = selected.merge(
        arrivals,
        on=["grid_id", "weekday", "hour"],
        how="left",
        validate="many_to_one",
    )
    selected["forecast_departures"] = (
        pd.to_numeric(selected[prediction_column], errors="raise").clip(lower=0)
        * demand_multiplier
    )
    selected["realized_departures"] = (
        pd.to_numeric(selected["actual_demand"], errors="raise").clip(lower=0)
        * demand_multiplier
    )
    selected["forecast_arrivals"] = selected["mean_arrivals"] * demand_multiplier
    selected["realized_arrivals"] = selected["forecast_arrivals"]
    return selected[
        [
            "timestamp",
            "grid_id",
            "weekday",
            "hour",
            "forecast_departures",
            "realized_departures",
            "forecast_arrivals",
            "realized_arrivals",
        ]
    ].sort_values(["timestamp", "grid_id"])


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def build_greedy_dispatch_plan(
    state: pd.DataFrame,
    forecast: pd.DataFrame,
    config: SimulationConfig,
) -> pd.DataFrame:
    columns = [
        "source_grid_id",
        "target_grid_id",
        "bikes_moved",
        "distance_km",
    ]
    if config.truck_count == 0 or config.max_relocated_bikes_per_hour == 0:
        return pd.DataFrame(columns=columns)
    working = state.set_index("grid_id").copy()
    demand = forecast.set_index("grid_id")
    working = working.join(
        demand[["forecast_departures", "forecast_arrivals"]], how="left"
    ).fillna({"forecast_departures": 0.0, "forecast_arrivals": 0.0})
    projected = (
        working["bikes"]
        + working["forecast_arrivals"]
        - working["forecast_departures"]
    )
    minimum_end = config.low_inventory_ratio * working["capacity"]
    maximum_end = config.high_inventory_ratio * working["capacity"]
    working["deficit"] = np.ceil(
        np.maximum.reduce(
            [
                (minimum_end - projected).to_numpy(),
                (working["forecast_departures"] - working["bikes"]).to_numpy(),
                np.zeros(len(working)),
            ]
        )
    )
    projected_surplus = np.floor((projected - maximum_end).clip(lower=0))
    physically_movable = np.floor(
        (working["bikes"] - working["forecast_departures"]).clip(lower=0)
    )
    working["surplus"] = np.minimum(projected_surplus, physically_movable)

    remaining_budget = min(
        config.max_relocated_bikes_per_hour,
        config.truck_count * config.truck_capacity,
    )
    routes: list[dict[str, object]] = []
    deficits = working.loc[
        working["deficit"].gt(0) & working["capacity"].gt(0)
    ].sort_values("deficit", ascending=False)
    for target_id, target in deficits.iterrows():
        needed = int(target["deficit"])
        if needed <= 0 or remaining_budget <= 0 or len(routes) >= config.truck_count:
            continue
        donors = working.loc[working["surplus"].gt(0)].copy()
        if donors.empty:
            break
        donors["distance_km"] = [
            _haversine_km(
                float(row["center_lat"]),
                float(row["center_lon"]),
                float(target["center_lat"]),
                float(target["center_lon"]),
            )
            for _, row in donors.iterrows()
        ]
        for source_id, source in donors.sort_values("distance_km").iterrows():
            if len(routes) >= config.truck_count or needed <= 0 or remaining_budget <= 0:
                break
            target_space = int(
                math.floor(max(working.loc[target_id, "capacity"] - working.loc[target_id, "bikes"], 0))
            )
            moved = min(
                needed,
                int(working.loc[source_id, "surplus"]),
                config.truck_capacity,
                remaining_budget,
                target_space,
            )
            if moved <= 0:
                continue
            routes.append(
                {
                    "source_grid_id": source_id,
                    "target_grid_id": target_id,
                    "bikes_moved": moved,
                    "distance_km": float(source["distance_km"]),
                }
            )
            working.loc[source_id, "bikes"] -= moved
            working.loc[source_id, "surplus"] -= moved
            working.loc[target_id, "bikes"] += moved
            needed -= moved
            remaining_budget -= moved
    return pd.DataFrame(routes, columns=columns)


def _simulate(
    inventory: pd.DataFrame,
    demand: pd.DataFrame,
    config: SimulationConfig,
    *,
    rebalancing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = inventory.loc[inventory["capacity"].gt(0)].copy()
    current = active.set_index("grid_id")["initial_bikes"].astype(float)
    state_reference = active.set_index("grid_id")
    trajectory: list[dict[str, object]] = []
    dispatches: list[pd.DataFrame] = []

    for timestamp, hourly in demand.groupby("timestamp", sort=True):
        hourly = hourly.loc[hourly["grid_id"].isin(active["grid_id"])].copy()
        state = state_reference.reset_index()[
            ["grid_id", "capacity", "center_lat", "center_lon"]
        ]
        state["bikes"] = state["grid_id"].map(current).astype(float)
        plan = (
            build_greedy_dispatch_plan(state, hourly, config)
            if rebalancing
            else pd.DataFrame(
                columns=["source_grid_id", "target_grid_id", "bikes_moved", "distance_km"]
            )
        )
        if not plan.empty:
            plan = plan.copy()
            plan.insert(0, "timestamp", timestamp)
            dispatches.append(plan)
        moved_out = plan.groupby("source_grid_id")["bikes_moved"].sum() if not plan.empty else pd.Series(dtype=float)
        moved_in = plan.groupby("target_grid_id")["bikes_moved"].sum() if not plan.empty else pd.Series(dtype=float)

        hourly = hourly.set_index("grid_id")
        for grid_id, reference in state_reference.iterrows():
            start_bikes = float(current.loc[grid_id])
            outflow = float(moved_out.get(grid_id, 0.0))
            inflow = float(moved_in.get(grid_id, 0.0))
            after_dispatch = start_bikes - outflow + inflow
            row = hourly.loc[grid_id]
            departures = float(row["realized_departures"])
            arrivals = float(row["realized_arrivals"])
            served_departures = min(after_dispatch, departures)
            unmet_departures = departures - served_departures
            after_departures = after_dispatch - served_departures
            available_docks = max(float(reference["capacity"]) - after_departures, 0.0)
            served_arrivals = min(arrivals, available_docks)
            failed_returns = arrivals - served_arrivals
            end_bikes = after_departures + served_arrivals
            end_bikes = min(max(end_bikes, 0.0), float(reference["capacity"]))
            current.loc[grid_id] = end_bikes
            trajectory.append(
                {
                    "timestamp": timestamp,
                    "grid_id": grid_id,
                    "capacity": float(reference["capacity"]),
                    "start_bikes": start_bikes,
                    "bikes_moved_out": outflow,
                    "bikes_moved_in": inflow,
                    "bikes_after_dispatch": after_dispatch,
                    "forecast_departures": float(row["forecast_departures"]),
                    "realized_departures": departures,
                    "served_departures": served_departures,
                    "unmet_departures": unmet_departures,
                    "forecast_arrivals": float(row["forecast_arrivals"]),
                    "realized_arrivals": arrivals,
                    "served_arrivals": served_arrivals,
                    "failed_returns": failed_returns,
                    "end_bikes": end_bikes,
                    "end_inventory_ratio": end_bikes / float(reference["capacity"]),
                }
            )
    dispatch = (
        pd.concat(dispatches, ignore_index=True)
        if dispatches
        else pd.DataFrame(
            columns=["timestamp", "source_grid_id", "target_grid_id", "bikes_moved", "distance_km"]
        )
    )
    return pd.DataFrame(trajectory), dispatch


def simulation_metrics(
    trajectory: pd.DataFrame,
    dispatch_plan: pd.DataFrame,
    *,
    strategy: str,
) -> dict[str, object]:
    departures = float(trajectory["realized_departures"].sum())
    arrivals = float(trajectory["realized_arrivals"].sum())
    unmet = float(trajectory["unmet_departures"].sum())
    failed = float(trajectory["failed_returns"].sum())
    relocated = float(dispatch_plan["bikes_moved"].sum()) if not dispatch_plan.empty else 0.0
    relocation_km = (
        float((dispatch_plan["bikes_moved"] * dispatch_plan["distance_km"]).sum())
        if not dispatch_plan.empty
        else 0.0
    )
    imbalance = float(
        (trajectory["end_inventory_ratio"].sub(0.5).abs() * trajectory["capacity"]).sum()
    )
    return {
        "strategy": strategy,
        "realized_departures": departures,
        "served_departures": departures - unmet,
        "unmet_departures": unmet,
        "demand_satisfaction_pct": 100.0 * (departures - unmet) / departures if departures else 100.0,
        "realized_arrivals": arrivals,
        "failed_returns": failed,
        "return_success_pct": 100.0 * (arrivals - failed) / arrivals if arrivals else 100.0,
        "empty_grid_hours": int(trajectory["end_bikes"].le(1e-9).sum()),
        "full_grid_hours": int(
            trajectory["end_bikes"].ge(trajectory["capacity"] - 1e-9).sum()
        ),
        "imbalance_bike_hours": imbalance,
        "dispatch_count": int(len(dispatch_plan)),
        "relocated_bikes": relocated,
        "relocation_bike_km": relocation_km,
    }


def run_rebalancing_simulation(
    snapshot: pd.DataFrame,
    top_grids: pd.DataFrame,
    demand: pd.DataFrame,
    config: SimulationConfig,
    *,
    demand_mode: str,
    snapshot_path: str | None = None,
    grid_size_m: int = DEFAULT_GRID_SIZE_M,
) -> RebalancingResult:
    inventory = snapshot_to_grid_inventory(
        snapshot, top_grids, grid_size_m=grid_size_m
    )
    active_ids = inventory.loc[inventory["capacity"].gt(0), "grid_id"].tolist()
    demand = demand.loc[demand["grid_id"].isin(active_ids)].copy()
    expected = demand["timestamp"].nunique() * len(active_ids)
    if len(demand) != expected:
        raise ValueError("模拟需求未覆盖全部有容量网格×小时")
    baseline, _ = _simulate(inventory, demand, config, rebalancing=False)
    strategy, dispatch = _simulate(inventory, demand, config, rebalancing=True)
    metrics = pd.DataFrame(
        [
            simulation_metrics(
                baseline,
                pd.DataFrame(columns=["bikes_moved", "distance_km"]),
                strategy="no_rebalancing",
            ),
            simulation_metrics(strategy, dispatch, strategy="greedy_rebalancing"),
        ]
    )
    start = pd.Timestamp(demand["timestamp"].min())
    end = pd.Timestamp(demand["timestamp"].max()) + pd.Timedelta(hours=1)
    return RebalancingResult(
        baseline_trajectory=baseline,
        strategy_trajectory=strategy,
        dispatch_plan=dispatch,
        metrics=metrics,
        initial_inventory=inventory,
        simulation_start=start,
        simulation_end=end,
        demand_mode=demand_mode,
        snapshot_path=snapshot_path,
    )


def save_rebalancing_outputs(
    result: RebalancingResult,
    config: SimulationConfig,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
) -> list[Path]:
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    map_dir = output_dir / "maps"
    targets = [
        table_dir / "baseline_trajectory.parquet",
        table_dir / "strategy_trajectory.parquet",
        table_dir / "dispatch_plan.csv",
        table_dir / "strategy_metrics.csv",
        figure_dir / "01_strategy_comparison.png",
        map_dir / "01_inventory_before_after.html",
        output_dir / "rebalancing_metadata.json",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "调度输出已存在（使用 --overwrite 替换）: "
            + ", ".join(path.name for path in existing)
        )
    for directory in (table_dir, figure_dir, map_dir):
        directory.mkdir(parents=True, exist_ok=True)
    result.baseline_trajectory.to_parquet(targets[0], index=False)
    result.strategy_trajectory.to_parquet(targets[1], index=False)
    result.dispatch_plan.to_csv(targets[2], index=False)
    result.metrics.to_csv(targets[3], index=False)

    matplotlib_cache = Path(tempfile.gettempdir()) / "capital_bikeshare_mpl_cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_names = [
        "unmet_departures",
        "failed_returns",
        "empty_grid_hours",
        "imbalance_bike_hours",
    ]
    labels = ["Unmet departures", "Failed returns", "Empty grid-hours", "Imbalance bike-hours"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    colors = ["#7A8CA5", "#2F6BFF"]
    for ax, metric, label in zip(axes.flat, metric_names, labels):
        values = result.metrics.set_index("strategy")[metric]
        ax.bar(["No rebalance", "Greedy"], values, color=colors)
        ax.set_title(label)
        upper = max(float(values.max()), 1.0)
        ax.set_ylim(0, upper * 1.15)
        for index, value in enumerate(values):
            ax.text(
                index,
                float(value) + upper * 0.015,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("Rebalancing Strategy Simulation")
    fig.tight_layout()
    fig.savefig(targets[4], dpi=180, bbox_inches="tight")
    plt.close(fig)

    import plotly.express as px

    initial = result.initial_inventory[
        ["grid_id", "center_lat", "center_lon", "initial_bikes", "capacity"]
    ].copy()
    initial["state"] = "Before"
    initial["bikes"] = initial["initial_bikes"]
    final = (
        result.strategy_trajectory.sort_values("timestamp")
        .groupby("grid_id", as_index=False)
        .tail(1)[["grid_id", "end_bikes"]]
        .merge(
            result.initial_inventory[["grid_id", "center_lat", "center_lon", "capacity"]],
            on="grid_id",
            how="left",
        )
    )
    final["state"] = "After"
    final["bikes"] = final["end_bikes"]
    map_frame = pd.concat(
        [
            initial[["grid_id", "center_lat", "center_lon", "capacity", "state", "bikes"]],
            final[["grid_id", "center_lat", "center_lon", "capacity", "state", "bikes"]],
        ],
        ignore_index=True,
    )
    map_frame["inventory_ratio"] = np.where(
        map_frame["capacity"].gt(0), map_frame["bikes"] / map_frame["capacity"], 0
    )
    figure = px.scatter_map(
        map_frame,
        lat="center_lat",
        lon="center_lon",
        size="capacity",
        color="inventory_ratio",
        hover_name="grid_id",
        hover_data={"bikes": ":.1f", "capacity": ":.0f"},
        animation_frame="state",
        color_continuous_scale="RdYlGn",
        range_color=(0, 1),
        zoom=11,
        center={"lat": 38.9072, "lon": -77.0369},
        map_style="carto-positron",
        title="Grid inventory before and after simulation",
    )
    figure.write_html(targets[5], include_plotlyjs=True, full_html=True)

    baseline = result.metrics.set_index("strategy").loc["no_rebalancing"]
    strategy = result.metrics.set_index("strategy").loc["greedy_rebalancing"]
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "simulation_start": result.simulation_start.isoformat(),
        "simulation_end": result.simulation_end.isoformat(),
        "demand_mode": result.demand_mode,
        "snapshot_path": result.snapshot_path,
        "config": asdict(config),
        "interpretation": "Counterfactual simulation initialized by one GBFS snapshot; not an observed operator backtest.",
        "unmet_departure_reduction": float(
            baseline["unmet_departures"] - strategy["unmet_departures"]
        ),
    }
    targets[6].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return targets


def _top_grids(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"grid_id": "string"})


def _prepare_profiles(args: argparse.Namespace) -> int:
    files = discover_cleaned_files(args.input_dir, args.start_month, args.end_month)
    profiles = build_historical_grid_flow_profiles(
        files,
        _top_grids(args.top_grids),
        profile_start=args.profile_start,
        profile_end=args.profile_end,
        grid_size_m=args.grid_size_m,
        chunksize=args.chunksize,
    )
    path = save_flow_profiles(profiles, args.output, overwrite=args.overwrite)
    print("\nHistorical flow profile complete:")
    print(f"  rows: {len(profiles):,}")
    print(f"  {path}")
    return 0


def _simulate_command(args: argparse.Namespace) -> int:
    if args.snapshot is None:
        snapshot, snapshot_path = load_latest_snapshot(args.snapshot_dir)
    else:
        snapshot_path = args.snapshot
        snapshot = pd.read_parquet(snapshot_path)
    top_grids = _top_grids(args.top_grids)
    profiles = pd.read_csv(args.profiles, dtype={"grid_id": "string"})
    config = SimulationConfig(
        horizon_hours=args.horizon_hours,
        truck_count=args.truck_count,
        truck_capacity=args.truck_capacity,
        max_relocated_bikes_per_hour=args.max_relocated_bikes_per_hour,
        low_inventory_ratio=args.low_inventory_ratio,
        high_inventory_ratio=args.high_inventory_ratio,
        demand_multiplier=args.demand_multiplier,
    )
    grid_order = top_grids.sort_values("demand_rank")["grid_id"].astype(str).tolist()
    snapshot_start = snapshot["snapshot_time"].iloc[0]
    simulation_start = args.start_time or snapshot_start
    if args.mode == "profile":
        demand = build_profile_demand(
            profiles,
            grid_order,
            start_time=simulation_start,
            horizon_hours=config.horizon_hours,
            demand_multiplier=config.demand_multiplier,
        )
    else:
        predictions = pd.read_parquet(args.predictions)
        demand = build_backtest_demand(
            predictions,
            profiles,
            grid_order,
            start_time=simulation_start,
            horizon_hours=config.horizon_hours,
            prediction_column=args.prediction_column,
            demand_multiplier=config.demand_multiplier,
        )
    result = run_rebalancing_simulation(
        snapshot,
        top_grids,
        demand,
        config,
        demand_mode=args.mode,
        snapshot_path=str(snapshot_path),
        grid_size_m=args.grid_size_m,
    )
    paths = save_rebalancing_outputs(
        result, config, output_dir=args.output_dir, overwrite=args.overwrite
    )
    print("\nRebalancing simulation complete:")
    print(result.metrics.to_string(index=False))
    for path in paths:
        print(f"  {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于一次GBFS快照的500米网格动态调度策略模拟"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("prepare-profiles", help="构建2025年网格小时出发/到达画像")
    profile.add_argument("--input-dir", type=Path, default=PROCESSED_DIR)
    profile.add_argument("--start-month", default="202501")
    profile.add_argument("--end-month", default="202512")
    profile.add_argument("--profile-start", default="2025-01-01")
    profile.add_argument("--profile-end", default="2026-01-01")
    profile.add_argument("--top-grids", type=Path, default=DEFAULT_TOP_GRIDS)
    profile.add_argument("--output", type=Path, default=DEFAULT_PROFILE_PATH)
    profile.add_argument("--grid-size-m", type=int, default=DEFAULT_GRID_SIZE_M)
    profile.add_argument("--chunksize", type=int, default=200_000)
    profile.add_argument("--overwrite", action="store_true")
    profile.set_defaults(handler=_prepare_profiles)

    simulate = subparsers.add_parser("simulate", help="运行无调度与贪心调度对比")
    simulate.add_argument("--snapshot", type=Path)
    simulate.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    simulate.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    simulate.add_argument("--top-grids", type=Path, default=DEFAULT_TOP_GRIDS)
    simulate.add_argument("--mode", choices=("profile", "backtest"), default="profile")
    simulate.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    simulate.add_argument(
        "--prediction-column",
        choices=(
            "weekly_naive_prediction",
            "historical_mean_prediction",
            "hist_gradient_boosting_prediction",
        ),
        default="hist_gradient_boosting_prediction",
    )
    simulate.add_argument("--start-time")
    simulate.add_argument("--horizon-hours", type=int, default=6)
    simulate.add_argument("--demand-multiplier", type=float, default=1.0)
    simulate.add_argument("--truck-count", type=int, default=3)
    simulate.add_argument("--truck-capacity", type=int, default=10)
    simulate.add_argument("--max-relocated-bikes-per-hour", type=int, default=30)
    simulate.add_argument("--low-inventory-ratio", type=float, default=0.25)
    simulate.add_argument("--high-inventory-ratio", type=float, default=0.75)
    simulate.add_argument("--grid-size-m", type=int, default=DEFAULT_GRID_SIZE_M)
    simulate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    simulate.add_argument("--overwrite", action="store_true")
    simulate.set_defaults(handler=_simulate_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
