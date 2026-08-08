"""Interactive Streamlit dashboard for the Capital Bikeshare DC project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.od_sankey import build_bipartite_od_sankey
from src.realtime_gbfs import (
    DEFAULT_SNAPSHOT_DIR,
    fetch_capital_bikeshare_snapshot,
    find_latest_snapshot,
    save_snapshot,
)
from src.rebalancing import (
    DEFAULT_PROFILE_PATH,
    DEFAULT_TOP_GRIDS,
    SimulationConfig,
    build_profile_demand,
    run_rebalancing_simulation,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD_DATA_DIR = PROJECT_ROOT / "outputs" / "dashboard" / "data"
FORECAST_TABLE_DIR = PROJECT_ROOT / "outputs" / "forecasting" / "tables"

MODEL_COLUMNS = {
    "Weekly naive (t-168)": "weekly_naive_prediction",
    "Historical grid × weekday × hour mean": "historical_mean_prediction",
    "HistGradientBoosting": "hist_gradient_boosting_prediction",
}
MODEL_KEYS = {
    "Weekly naive (t-168)": "weekly_naive",
    "Historical grid × weekday × hour mean": "historical_mean",
    "HistGradientBoosting": "hist_gradient_boosting",
}


st.set_page_config(
    page_title="Capital Bikeshare DC Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    names = (
        "daily_user_summary",
        "hourly_user_volume",
        "station_daily_user",
        "grid_daily_user",
        "od_daily_user",
        "station_reference",
        "top_activity_nodes",
    )
    data: dict[str, pd.DataFrame] = {}
    for name in names:
        path = DASHBOARD_DATA_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        for column in (
            "station_id",
            "source_station_id",
            "target_station_id",
            "grid_id",
            "user_type",
        ):
            if column in frame.columns:
                frame[column] = frame[column].astype("string")
        data[name] = frame
    return data


@st.cache_data(show_spinner=False)
def load_forecast_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_path = FORECAST_TABLE_DIR / "forecast_predictions_2026.parquet"
    metric_path = FORECAST_TABLE_DIR / "test_metrics_by_model_and_month.csv"
    grid_metric_path = FORECAST_TABLE_DIR / "test_metrics_by_grid.csv"
    for path in (prediction_path, metric_path, grid_metric_path):
        if not path.exists():
            raise FileNotFoundError(path)
    predictions = pd.read_parquet(prediction_path)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    metrics = pd.read_csv(metric_path, dtype={"period": "string"})
    grid_metrics = pd.read_csv(grid_metric_path, dtype={"grid_id": "string"})
    return predictions, metrics, grid_metrics


def comma(value: float) -> str:
    return f"{value:,.0f}"


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    y_true = actual.to_numpy(dtype="float64")
    y_pred = np.clip(predicted.to_numpy(dtype="float64"), 0.0, None)
    absolute_error = np.abs(y_true - y_pred)
    positive = y_true > 0
    return {
        "MAE": float(absolute_error.mean()),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "MAPE (%)": (
            float(np.mean(absolute_error[positive] / y_true[positive]) * 100)
            if positive.any()
            else np.nan
        ),
        "WAPE (%)": (
            float(absolute_error.sum() / y_true.sum() * 100)
            if y_true.sum() > 0
            else np.nan
        ),
    }


def plot_chart(figure: go.Figure) -> None:
    figure.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch", theme="streamlit")


try:
    data = load_dashboard_data()
    forecast_predictions, forecast_metrics, forecast_grid_metrics = load_forecast_data()
except FileNotFoundError as exc:
    st.error(f"缺少仪表板数据文件：{exc}")
    st.code("python -m src.dashboard_data --overwrite\npython -m src.forecast --overwrite")
    st.stop()

daily_user = data["daily_user_summary"]
available_users = sorted(daily_user["user_type"].dropna().unique().tolist())
minimum_date = daily_user["date"].min().date()
maximum_date = daily_user["date"].max().date()

st.sidebar.header("Historical filters")
selected_dates = st.sidebar.date_input(
    "Date range",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date,
)
selected_users = st.sidebar.multiselect(
    "User type",
    available_users,
    default=available_users,
)
if len(selected_dates) != 2:
    st.sidebar.info("Select both a start and end date.")
    st.stop()
if not selected_users:
    st.sidebar.warning("Select at least one user type.")
    st.stop()

start_date = pd.Timestamp(selected_dates[0])
end_date = pd.Timestamp(selected_dates[1])
if start_date > end_date:
    st.sidebar.error("Start date must not be after end date.")
    st.stop()


def historical_filter(frame: pd.DataFrame) -> pd.DataFrame:
    mask = frame["date"].between(start_date, end_date)
    if "user_type" in frame.columns:
        mask &= frame["user_type"].isin(selected_users)
    return frame.loc[mask].copy()


st.title("Capital Bikeshare DC")
st.caption(
    "2025 spatiotemporal demand and user behavior · 2026 Jan–Jul out-of-time forecasting"
)

overview_tab, spatial_tab, od_tab, forecast_tab, rebalancing_tab = st.tabs(
    [
        "Overview",
        "Spatial demand",
        "OD flows",
        "Forecasting",
        "Live rebalancing",
    ]
)

with overview_tab:
    filtered_daily = historical_filter(daily_user)
    if filtered_daily.empty:
        st.warning("No trips match the selected historical filters.")
        st.stop()
    daily_total = (
        filtered_daily.groupby("date", as_index=False)
        .agg(
            trip_count=("trip_count", "sum"),
            total_duration_min=("total_duration_min", "sum"),
            total_distance_km=("total_distance_km", "sum"),
            distance_valid_trips=("distance_valid_trips", "sum"),
        )
        .sort_values("date")
    )
    selected_day_count = int((end_date - start_date).days + 1)
    total_trips = int(filtered_daily["trip_count"].sum())
    total_duration = float(filtered_daily["total_duration_min"].sum())
    total_distance = float(filtered_daily["total_distance_km"].sum())
    valid_distance_trips = int(filtered_daily["distance_valid_trips"].sum())

    metric_columns = st.columns(4)
    metric_columns[0].metric("Trips", comma(total_trips))
    metric_columns[1].metric(
        "Average trips/day", comma(total_trips / selected_day_count)
    )
    metric_columns[2].metric(
        "Average duration", f"{total_duration / total_trips:.1f} min"
    )
    metric_columns[3].metric(
        "Straight-line distance",
        f"{total_distance:,.0f} km",
        help=f"Based on {valid_distance_trips:,} trips with valid Haversine distance.",
    )

    trend_metric = st.radio(
        "Trend measure",
        ["Trips", "Total distance (km)"],
        horizontal=True,
        key="trend_measure",
    )
    trend_column = "trip_count" if trend_metric == "Trips" else "total_distance_km"
    trend = px.line(
        daily_total,
        x="date",
        y=trend_column,
        labels={"date": "Date", trend_column: trend_metric},
        title=f"Daily {trend_metric.lower()}",
    )
    trend.update_traces(line=dict(width=2))
    plot_chart(trend)

    left, right = st.columns(2)
    filtered_hourly = historical_filter(data["hourly_user_volume"])
    hourly = (
        filtered_hourly.groupby("hour", as_index=False)["trip_count"].sum()
        .set_index("hour")
        .reindex(range(24), fill_value=0)
        .reset_index()
    )
    hourly["average_trips_per_day"] = hourly["trip_count"] / selected_day_count
    hourly_figure = px.bar(
        hourly,
        x="hour",
        y="average_trips_per_day",
        labels={"hour": "Start hour", "average_trips_per_day": "Average trips/day"},
        title="Average hourly activity",
    )
    hourly_figure.update_xaxes(dtick=1)
    with left:
        plot_chart(hourly_figure)

    user_share = (
        filtered_daily.groupby("user_type", as_index=False)["trip_count"].sum()
    )
    user_figure = px.bar(
        user_share,
        x="user_type",
        y="trip_count",
        color="user_type",
        labels={"user_type": "User type", "trip_count": "Trips"},
        title="Trips by user type",
    )
    user_figure.update_layout(showlegend=False)
    with right:
        plot_chart(user_figure)

    holidays = USFederalHolidayCalendar().holidays(start=start_date, end=end_date)
    daily_total["day_type"] = np.where(
        daily_total["date"].dt.weekday.ge(5) | daily_total["date"].isin(holidays),
        "Weekend / federal holiday",
        "Workday",
    )
    day_type_figure = px.box(
        daily_total,
        x="day_type",
        y="trip_count",
        color="day_type",
        points="outliers",
        labels={"day_type": "Day type", "trip_count": "Daily trips"},
        title="Workday and weekend/holiday daily volume",
    )
    day_type_figure.update_layout(showlegend=False)
    plot_chart(day_type_figure)

with spatial_tab:
    layer = st.radio(
        "Map layer", ["Stations", "500 m grids"], horizontal=True, key="map_layer"
    )
    volume_measure = st.selectbox(
        "Volume measure",
        ["total_activity", "departures", "arrivals"],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    if layer == "Stations":
        station_counts = historical_filter(data["station_daily_user"])
        station_counts = station_counts.groupby("station_id", as_index=False)[
            ["departures", "arrivals", "total_activity", "net_inflow"]
        ].sum()
        station_map = station_counts.merge(
            data["station_reference"][
                ["station_id", "station_name", "station_lat", "station_lon"]
            ],
            on="station_id",
            how="left",
            validate="one_to_one",
        )
        station_map = station_map.loc[
            station_map[volume_measure].gt(0)
            & station_map["station_lat"].notna()
            & station_map["station_lon"].notna()
        ]
        map_figure = px.scatter_map(
            station_map,
            lat="station_lat",
            lon="station_lon",
            size=volume_measure,
            color=volume_measure,
            hover_name="station_name",
            hover_data={
                "station_id": True,
                "departures": ":,",
                "arrivals": ":,",
                "total_activity": ":,",
                "net_inflow": ":,",
                "station_lat": False,
                "station_lon": False,
            },
            zoom=11,
            center={"lat": 38.9072, "lon": -77.0369},
            map_style="carto-positron",
            size_max=34,
            title="Observed station activity",
        )
        map_figure.update_layout(height=620)
        plot_chart(map_figure)
        ranking = station_map.nlargest(10, volume_measure).sort_values(volume_measure)
        ranking_figure = px.bar(
            ranking,
            x=volume_measure,
            y="station_name",
            orientation="h",
            labels={volume_measure: volume_measure.replace("_", " ").title(), "station_name": "Station"},
            title="Top 10 stations for selected filters",
        )
        plot_chart(ranking_figure)
        st.dataframe(
            station_map.nlargest(10, volume_measure)[
                ["station_id", "station_name", "departures", "arrivals", "total_activity", "net_inflow"]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        grid_counts = historical_filter(data["grid_daily_user"])
        grid_map = grid_counts.groupby(
            ["grid_id", "center_lat", "center_lon"], as_index=False
        )[["departures", "arrivals", "total_activity", "net_inflow"]].sum()
        grid_map = grid_map.loc[grid_map[volume_measure].gt(0)]
        map_figure = px.scatter_map(
            grid_map,
            lat="center_lat",
            lon="center_lon",
            size=volume_measure,
            color=volume_measure,
            hover_name="grid_id",
            hover_data={
                "departures": ":,",
                "arrivals": ":,",
                "total_activity": ":,",
                "net_inflow": ":,",
                "center_lat": False,
                "center_lon": False,
            },
            zoom=10.5,
            center={"lat": 38.9072, "lon": -77.0369},
            map_style="carto-positron",
            size_max=30,
            title="500 m grid activity (centroid symbols)",
        )
        map_figure.update_layout(height=620)
        plot_chart(map_figure)
        st.caption(
            "Each symbol represents the center of an exact 500 m × 500 m EPSG:32618 grid."
        )

with od_tab:
    od = historical_filter(data["od_daily_user"])
    od = (
        od.groupby(
            [
                "source_station_id",
                "target_station_id",
                "source_station_name",
                "target_station_name",
            ],
            as_index=False,
            dropna=False,
        )["trip_count"]
        .sum()
        .nlargest(20, "trip_count")
    )
    if od.empty:
        st.info("No Top-10-node OD flows match the selected filters.")
    else:
        sankey = build_bipartite_od_sankey(
            od,
            data["station_reference"],
            title="Top 20 directed OD flows among fixed Top 10 activity stations",
        )
        st.plotly_chart(sankey, width="stretch", theme="streamlit")
        st.caption(
            "Origins and destinations are separated into fixed columns. Within "
            "each column, stations are ordered approximately north to south by latitude; "
            "horizontal placement does not represent longitude."
        )
        st.dataframe(
            od[
                ["source_station_name", "target_station_name", "trip_count"]
            ].rename(
                columns={
                    "source_station_name": "Origin",
                    "target_station_name": "Destination",
                    "trip_count": "Trips",
                }
            ),
            width="stretch",
            hide_index=True,
        )

with forecast_tab:
    forecast_min = forecast_predictions["timestamp"].min().date()
    forecast_max = forecast_predictions["timestamp"].max().date()
    forecast_controls = st.columns([2, 2, 3])
    with forecast_controls[0]:
        selected_model_label = st.selectbox("Forecast model", list(MODEL_COLUMNS))
    grid_options = ["All Top 10 grids", *sorted(forecast_predictions["grid_id"].unique())]
    with forecast_controls[1]:
        selected_grid = st.selectbox("Forecast grid", grid_options)
    with forecast_controls[2]:
        forecast_dates = st.date_input(
            "Forecast evaluation range",
            value=(forecast_min, forecast_max),
            min_value=forecast_min,
            max_value=forecast_max,
            key="forecast_dates",
        )
    if len(forecast_dates) != 2:
        st.info("Select both forecast evaluation dates.")
        st.stop()
    forecast_start = pd.Timestamp(forecast_dates[0])
    forecast_end = pd.Timestamp(forecast_dates[1]) + pd.Timedelta(days=1)
    forecast_subset = forecast_predictions.loc[
        forecast_predictions["timestamp"].ge(forecast_start)
        & forecast_predictions["timestamp"].lt(forecast_end)
    ].copy()
    if selected_grid != "All Top 10 grids":
        forecast_subset = forecast_subset.loc[
            forecast_subset["grid_id"].eq(selected_grid)
        ]

    selected_prediction_column = MODEL_COLUMNS[selected_model_label]
    selected_metrics = regression_metrics(
        forecast_subset["actual_demand"], forecast_subset[selected_prediction_column]
    )
    metric_columns = st.columns(4)
    for column, (name, value) in zip(metric_columns, selected_metrics.items()):
        column.metric(name, f"{value:.2f}")

    daily_forecast = (
        forecast_subset.set_index("timestamp")
        .groupby(pd.Grouper(freq="D"))[
            ["actual_demand", selected_prediction_column]
        ]
        .sum()
        .reset_index()
        .rename(columns={selected_prediction_column: "predicted_demand"})
    )
    comparison = go.Figure()
    comparison.add_scatter(
        x=daily_forecast["timestamp"],
        y=daily_forecast["actual_demand"],
        mode="lines",
        name="Actual",
    )
    comparison.add_scatter(
        x=daily_forecast["timestamp"],
        y=daily_forecast["predicted_demand"],
        mode="lines",
        name=selected_model_label,
    )
    comparison.update_layout(
        title="Daily actual vs predicted departures",
        xaxis_title="Date",
        yaxis_title="Trips",
    )
    plot_chart(comparison)

    left, right = st.columns(2)
    dynamic_model_metrics: list[dict[str, object]] = []
    for label, prediction_column in MODEL_COLUMNS.items():
        dynamic_model_metrics.append(
            {
                "model": label,
                **regression_metrics(
                    forecast_subset["actual_demand"],
                    forecast_subset[prediction_column],
                ),
            }
        )
    dynamic_metrics = pd.DataFrame(dynamic_model_metrics)
    comparison_metric = st.selectbox(
        "Model comparison metric",
        ["RMSE", "MAE", "WAPE (%)", "MAPE (%)"],
        key="forecast_comparison_metric",
    )
    metric_figure = px.bar(
        dynamic_metrics,
        x="model",
        y=comparison_metric,
        color="model",
        text_auto=".2f",
        labels={"model": "Model"},
        title=f"Model comparison: {comparison_metric}",
    )
    metric_figure.update_layout(showlegend=False)
    with left:
        plot_chart(metric_figure)

    monthly_metrics = forecast_metrics.loc[
        forecast_metrics["period"].ne("overall")
    ].copy()
    monthly_metrics["model_label"] = monthly_metrics["model"].map(
        {value: key for key, value in MODEL_KEYS.items()}
    )
    monthly_figure = px.line(
        monthly_metrics,
        x="period",
        y="rmse",
        color="model_label",
        markers=True,
        labels={"period": "Month", "rmse": "RMSE", "model_label": "Model"},
        title="2026 monthly RMSE",
    )
    with right:
        plot_chart(monthly_figure)

    with st.expander("Forecast definitions"):
        st.markdown(
            "- **Weekly naive:** demand in the same grid 168 hours earlier.\n"
            "- **Historical mean:** 2025 grid × weekday × hour average.\n"
            "- **HistGradientBoosting:** Poisson tree model using calendar, lagged, rolling and grid features.\n"
            "- MAPE is calculated only where actual hourly demand is greater than zero."
        )

with rebalancing_tab:
    st.subheader("Live snapshot rebalancing simulation")
    st.caption(
        "One GBFS station-status snapshot initializes a grid-level counterfactual "
        "simulation. Demand is represented by the 2025 weekday × hour profile; "
        "results are not an observed Capital Bikeshare operator backtest."
    )

    if st.button("Fetch latest GBFS snapshot", type="primary"):
        try:
            with st.spinner("Fetching Capital Bikeshare station status..."):
                live_snapshot, live_metadata = fetch_capital_bikeshare_snapshot()
                parquet_path, _ = save_snapshot(
                    live_snapshot,
                    live_metadata,
                    output_dir=DEFAULT_SNAPSHOT_DIR,
                    overwrite=True,
                )
            st.success(f"Saved {len(live_snapshot):,} stations to {parquet_path.name}.")
            st.rerun()
        except Exception as exc:  # Network/feed errors should not break other tabs.
            st.error(f"GBFS snapshot fetch failed: {exc}")

    latest_snapshot_path = find_latest_snapshot(DEFAULT_SNAPSHOT_DIR)
    required_rebalancing_files = [DEFAULT_PROFILE_PATH, DEFAULT_TOP_GRIDS]
    missing_rebalancing_files = [
        path for path in required_rebalancing_files if not path.exists()
    ]
    if latest_snapshot_path is None:
        st.info(
            "No local snapshot is available. Use the button above or run "
            "`python -m src.realtime_gbfs`."
        )
    elif missing_rebalancing_files:
        st.info(
            "The historical flow profile or Top-10 grid table is missing. Run "
            "`python -m src.forecast --overwrite` and "
            "`python -m src.rebalancing prepare-profiles --overwrite`."
        )
        st.code("\n".join(str(path) for path in missing_rebalancing_files))
    else:
        try:
            snapshot = pd.read_parquet(latest_snapshot_path)
            profiles = pd.read_csv(
                DEFAULT_PROFILE_PATH, dtype={"grid_id": "string"}
            )
            top_grids = pd.read_csv(
                DEFAULT_TOP_GRIDS, dtype={"grid_id": "string"}
            )
            snapshot_time = pd.Timestamp(snapshot["snapshot_time"].iloc[0])
            if snapshot_time.tzinfo is None:
                snapshot_time = snapshot_time.tz_localize("UTC")
            else:
                snapshot_time = snapshot_time.tz_convert("UTC")
            snapshot_age_minutes = max(
                0.0,
                (pd.Timestamp.now(tz="UTC") - snapshot_time).total_seconds() / 60,
            )

            snapshot_metrics = st.columns(4)
            snapshot_metrics[0].metric("Stations", comma(len(snapshot)))
            snapshot_metrics[1].metric(
                "Available bikes", comma(snapshot["vehicles_available"].sum())
            )
            snapshot_metrics[2].metric(
                "Available docks", comma(snapshot["docks_available"].sum())
            )
            snapshot_metrics[3].metric(
                "Snapshot age", f"{snapshot_age_minutes:,.1f} min"
            )
            st.caption(
                f"Snapshot timestamp (UTC): {snapshot_time:%Y-%m-%d %H:%M:%S} · "
                f"Local file: {latest_snapshot_path.name}"
            )
            if snapshot_age_minutes > 5:
                st.warning(
                    "This snapshot is older than five minutes. It remains useful for "
                    "demonstrating the method, but should not be interpreted as current supply."
                )

            station_map_data = snapshot.loc[
                snapshot["lat"].notna()
                & snapshot["lon"].notna()
                & snapshot["functional_capacity"].gt(0)
                & snapshot["is_installed"].fillna(False),
                [
                    "station_name",
                    "lat",
                    "lon",
                    "vehicles_available",
                    "docks_available",
                    "functional_capacity",
                ],
            ].copy()
            station_map_data["inventory_ratio"] = (
                station_map_data["vehicles_available"]
                / station_map_data["functional_capacity"]
            )
            station_map = px.scatter_map(
                station_map_data,
                lat="lat",
                lon="lon",
                size="functional_capacity",
                color="inventory_ratio",
                hover_name="station_name",
                hover_data={
                    "vehicles_available": ":.0f",
                    "docks_available": ":.0f",
                    "functional_capacity": ":.0f",
                    "lat": False,
                    "lon": False,
                },
                color_continuous_scale="RdYlGn",
                range_color=(0, 1),
                zoom=10,
                center={"lat": 38.9072, "lon": -77.0369},
                map_style="carto-positron",
                title="Station inventory in the selected live snapshot",
            )
            station_map.update_layout(height=520)
            st.plotly_chart(station_map, width="stretch", theme="streamlit")

            st.markdown("#### Simulation controls")
            control_columns = st.columns(4)
            with control_columns[0]:
                horizon_hours = st.slider(
                    "Horizon (hours)", 1, 12, 6, key="rebalance_horizon"
                )
            with control_columns[1]:
                demand_multiplier = st.slider(
                    "Demand multiplier",
                    0.5,
                    3.0,
                    1.0,
                    0.1,
                    key="rebalance_demand_multiplier",
                )
            with control_columns[2]:
                truck_count = st.slider(
                    "Trucks per hour", 0, 10, 3, key="rebalance_trucks"
                )
            with control_columns[3]:
                truck_capacity = st.slider(
                    "Bikes per truck", 1, 30, 10, key="rebalance_capacity"
                )

            config = SimulationConfig(
                horizon_hours=horizon_hours,
                truck_count=truck_count,
                truck_capacity=truck_capacity,
                max_relocated_bikes_per_hour=truck_count * truck_capacity,
                low_inventory_ratio=0.25,
                high_inventory_ratio=0.75,
                demand_multiplier=demand_multiplier,
            )
            grid_order = (
                top_grids.sort_values("demand_rank")["grid_id"]
                .astype(str)
                .tolist()
            )
            demand = build_profile_demand(
                profiles,
                grid_order,
                start_time=snapshot_time,
                horizon_hours=config.horizon_hours,
                demand_multiplier=config.demand_multiplier,
            )
            result = run_rebalancing_simulation(
                snapshot,
                top_grids,
                demand,
                config,
                demand_mode="profile",
                snapshot_path=str(latest_snapshot_path),
            )
            strategy_metrics = result.metrics.set_index("strategy")
            baseline = strategy_metrics.loc["no_rebalancing"]
            strategy = strategy_metrics.loc["greedy_rebalancing"]

            outcome_columns = st.columns(4)
            outcome_columns[0].metric(
                "Unmet departures reduced",
                f"{baseline['unmet_departures'] - strategy['unmet_departures']:.1f}",
            )
            outcome_columns[1].metric(
                "Failed returns reduced",
                f"{baseline['failed_returns'] - strategy['failed_returns']:.1f}",
            )
            outcome_columns[2].metric(
                "Imbalance reduced",
                f"{baseline['imbalance_bike_hours'] - strategy['imbalance_bike_hours']:.1f} bike-hours",
            )
            outcome_columns[3].metric(
                "Relocation workload",
                f"{strategy['relocated_bikes']:.0f} bikes",
                f"{strategy['relocation_bike_km']:.1f} bike-km",
            )

            metric_labels = {
                "unmet_departures": "Unmet departures",
                "failed_returns": "Failed returns",
                "empty_grid_hours": "Empty grid-hours",
                "imbalance_bike_hours": "Imbalance bike-hours",
            }
            metric_frame = result.metrics.melt(
                id_vars="strategy",
                value_vars=list(metric_labels),
                var_name="metric",
                value_name="value",
            )
            metric_frame["metric"] = metric_frame["metric"].map(metric_labels)
            metric_frame["strategy"] = metric_frame["strategy"].map(
                {
                    "no_rebalancing": "No rebalancing",
                    "greedy_rebalancing": "Greedy rebalancing",
                }
            )
            strategy_chart = px.bar(
                metric_frame,
                x="metric",
                y="value",
                color="strategy",
                barmode="group",
                facet_col="metric",
                facet_col_wrap=2,
                title="Counterfactual strategy comparison",
                labels={"strategy": "Strategy", "value": "Value"},
            )
            strategy_chart.for_each_annotation(
                lambda annotation: annotation.update(
                    text=annotation.text.split("=")[-1]
                )
            )
            strategy_chart.update_xaxes(showticklabels=False, title_text="")
            strategy_chart.update_yaxes(matches=None, title_text="")
            strategy_chart.update_layout(height=620)
            st.plotly_chart(strategy_chart, width="stretch", theme="streamlit")

            final_inventory = (
                result.strategy_trajectory.sort_values("timestamp")
                .groupby("grid_id", as_index=False)
                .tail(1)[["grid_id", "end_bikes"]]
                .merge(
                    result.initial_inventory,
                    on="grid_id",
                    how="left",
                    validate="one_to_one",
                )
            )
            final_inventory["final_inventory_ratio"] = np.where(
                final_inventory["capacity"].gt(0),
                final_inventory["end_bikes"] / final_inventory["capacity"],
                np.nan,
            )
            final_inventory = final_inventory.loc[final_inventory["capacity"].gt(0)]
            final_map = px.scatter_map(
                final_inventory,
                lat="center_lat",
                lon="center_lon",
                size="capacity",
                color="final_inventory_ratio",
                hover_name="grid_id",
                hover_data={
                    "initial_bikes": ":.1f",
                    "end_bikes": ":.1f",
                    "capacity": ":.0f",
                    "station_count": True,
                    "center_lat": False,
                    "center_lon": False,
                },
                color_continuous_scale="RdYlGn",
                range_color=(0, 1),
                zoom=11,
                center={"lat": 38.9072, "lon": -77.0369},
                map_style="carto-positron",
                title="Top-10 grid inventory after greedy rebalancing simulation",
            )
            final_map.update_layout(height=520)
            st.plotly_chart(final_map, width="stretch", theme="streamlit")

            st.markdown("#### Dispatch plan")
            if result.dispatch_plan.empty:
                st.info(
                    "No relocation was triggered under the current inventory, demand, "
                    "threshold and fleet settings."
                )
            else:
                dispatch_table = result.dispatch_plan.copy()
                dispatch_table["timestamp"] = pd.to_datetime(
                    dispatch_table["timestamp"]
                ).dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(
                    dispatch_table.rename(
                        columns={
                            "timestamp": "Hour",
                            "source_grid_id": "From grid",
                            "target_grid_id": "To grid",
                            "bikes_moved": "Bikes",
                            "distance_km": "Distance (km)",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
        except Exception as exc:
            st.error(f"Unable to run the rebalancing simulation: {exc}")
