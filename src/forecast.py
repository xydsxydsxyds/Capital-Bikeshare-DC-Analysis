"""Leakage-safe hourly demand forecasting for Capital Bikeshare grids.

The forecasting target is the number of departures in each of the ten busiest
500 m grids during the next local clock hour.  Grid selection is performed on
2025 training data only.  The module compares two transparent baselines with a
Poisson histogram gradient-boosting model and uses expanding-window validation
before the untouched 2026 out-of-time evaluation.

Weather is intentionally excluded from this implementation.  Every model
feature is either known from the calendar or derived from demand observed
strictly before the target hour.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

# Configure loky before importing joblib/sklearn. Restricted Windows images may
# not provide WMIC, so physical-core detection otherwise emits a harmless
# warning even though the models can use the logical processor count normally.
_logical_core_count = int(
    os.environ.get("NUMBER_OF_PROCESSORS", str(os.cpu_count() or 1))
)
_loky_core_limit = min(8, _logical_core_count - 1) if _logical_core_count > 1 else 1
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_loky_core_limit))

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pyproj import Transformer
from sklearn.ensemble import HistGradientBoostingRegressor

from .paths import PROCESSED_DIR
from .spatial_od import DEFAULT_GRID_SIZE_M, UTM18N_EPSG, WGS84_EPSG
from .temporal import discover_cleaned_files


REQUIRED_FORECAST_COLUMNS = (
    "start_time",
    "start_lat",
    "start_lon",
    "has_valid_start_coords",
)

LAG_HOURS = (1, 2, 3, 24, 48, 168)
ROLLING_WINDOWS = (3, 6, 24, 168)


@dataclass(frozen=True)
class ValidationFold:
    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str


VALIDATION_FOLDS = (
    ValidationFold(
        "train_jan_jun_validate_jul_aug",
        "2025-01-01",
        "2025-07-01",
        "2025-07-01",
        "2025-09-01",
    ),
    ValidationFold(
        "train_jan_aug_validate_sep_oct",
        "2025-01-01",
        "2025-09-01",
        "2025-09-01",
        "2025-11-01",
    ),
    ValidationFold(
        "train_jan_oct_validate_nov_dec",
        "2025-01-01",
        "2025-11-01",
        "2025-11-01",
        "2026-01-01",
    ),
)


MODEL_CANDIDATES: tuple[tuple[str, dict[str, object]], ...] = (
    (
        "balanced",
        {
            "learning_rate": 0.06,
            "max_iter": 220,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
        },
    ),
    (
        "compact",
        {
            "learning_rate": 0.08,
            "max_iter": 180,
            "max_leaf_nodes": 23,
            "min_samples_leaf": 40,
            "l2_regularization": 2.0,
        },
    ),
    (
        "flexible",
        {
            "learning_rate": 0.04,
            "max_iter": 280,
            "max_leaf_nodes": 47,
            "min_samples_leaf": 25,
            "l2_regularization": 2.0,
        },
    ),
)


@dataclass
class ForecastTrainingResult:
    panel: pd.DataFrame
    feature_columns: list[str]
    top_grids: pd.DataFrame
    validation_metrics: pd.DataFrame
    validation_summary: pd.DataFrame
    best_candidate_name: str
    best_candidate_parameters: dict[str, object]
    selected_model: str
    main_model: HistGradientBoostingRegressor
    historical_group_means: pd.DataFrame
    test_predictions: pd.DataFrame | None
    test_metrics: pd.DataFrame | None
    test_metrics_by_grid: pd.DataFrame | None
    test_start_time: pd.Timestamp | None
    test_end_time: pd.Timestamp | None


def _month_sequence(start_month: str, end_month: str) -> list[str]:
    try:
        start = pd.Period(start_month, freq="M")
        end = pd.Period(end_month, freq="M")
    except (TypeError, ValueError) as exc:
        raise ValueError("月份必须使用 YYYYMM 格式") from exc
    if start > end:
        raise ValueError("起始月份不能晚于结束月份")
    return [period.strftime("%Y%m") for period in pd.period_range(start, end)]


def _month_time_bounds(
    start_month: str, end_month: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return an inclusive month start and exclusive month-after end."""

    _month_sequence(start_month, end_month)
    start = pd.Period(start_month, freq="M").start_time
    end = (pd.Period(end_month, freq="M") + 1).start_time
    return start, end


def require_complete_months(
    files: Sequence[Path], start_month: str, end_month: str, label: str
) -> None:
    expected = set(_month_sequence(start_month, end_month))
    present = {path.name[:6] for path in files}
    missing = sorted(expected - present)
    if missing:
        raise FileNotFoundError(
            f"{label}缺少清洗文件: {', '.join(missing)}。请先运行 src.preprocess。"
        )


def _read_forecast_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    if path.suffix == ".csv":
        header = set(pd.read_csv(path, nrows=0).columns)
        missing = sorted(set(REQUIRED_FORECAST_COLUMNS) - header)
        if missing:
            raise ValueError(f"{path.name} 缺少字段: {', '.join(missing)}")
        yield from pd.read_csv(
            path,
            usecols=list(REQUIRED_FORECAST_COLUMNS),
            chunksize=chunksize,
            low_memory=False,
        )
        return

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "读取 Parquet 需要 pyarrow；请执行 pip install -r requirements.txt"
        ) from exc

    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    missing = sorted(set(REQUIRED_FORECAST_COLUMNS) - available)
    if missing:
        raise ValueError(f"{path.name} 缺少字段: {', '.join(missing)}")
    for batch in parquet_file.iter_batches(
        batch_size=chunksize, columns=list(REQUIRED_FORECAST_COLUMNS)
    ):
        yield batch.to_pandas()


def aggregate_hourly_departures(
    files: Sequence[Path],
    *,
    grid_size_m: int = DEFAULT_GRID_SIZE_M,
    chunksize: int = 200_000,
) -> pd.Series:
    """Aggregate valid trip origins to local hour and exact UTM grid."""

    if grid_size_m <= 0:
        raise ValueError("grid_size_m 必须大于 0")
    if not files:
        raise ValueError("没有可用于需求预测的清洗数据文件")

    transformer = Transformer.from_crs(
        WGS84_EPSG, UTM18N_EPSG, always_xy=True
    )
    total: pd.Series | None = None
    for path in files:
        for raw in _read_forecast_chunks(path, chunksize):
            start_time = pd.to_datetime(raw["start_time"], errors="coerce")
            latitude = pd.to_numeric(raw["start_lat"], errors="coerce")
            longitude = pd.to_numeric(raw["start_lon"], errors="coerce")
            valid = (
                raw["has_valid_start_coords"].fillna(False).astype(bool)
                & start_time.notna()
                & latitude.notna()
                & longitude.notna()
            )
            if not valid.any():
                continue

            eastings, northings = transformer.transform(
                longitude.loc[valid].to_numpy(dtype="float64"),
                latitude.loc[valid].to_numpy(dtype="float64"),
            )
            chunk = pd.DataFrame(
                {
                    "timestamp": start_time.loc[valid].dt.floor("h").to_numpy(),
                    "grid_x": np.floor(np.asarray(eastings) / grid_size_m).astype(
                        "int32"
                    ),
                    "grid_y": np.floor(np.asarray(northings) / grid_size_m).astype(
                        "int32"
                    ),
                }
            )
            counts = chunk.groupby(
                ["timestamp", "grid_x", "grid_y"], observed=True
            ).size()
            total = (
                counts.astype("int64")
                if total is None
                else total.add(counts, fill_value=0).astype("int64")
            )

    if total is None or total.empty:
        raise ValueError("没有坐标有效的出发行程可用于需求预测")
    total.index = total.index.set_names(["timestamp", "grid_x", "grid_y"])
    return total.sort_index()


def _grid_id(grid_x: pd.Series, grid_y: pd.Series) -> pd.Series:
    return "E" + grid_x.astype(str) + "_N" + grid_y.astype(str)


def select_top_grids(
    training_counts: pd.Series,
    *,
    top_n: int = 10,
    grid_size_m: int = DEFAULT_GRID_SIZE_M,
) -> pd.DataFrame:
    """Select target grids using training departures only."""

    if top_n <= 0:
        raise ValueError("top_n 必须大于 0")
    by_grid = (
        training_counts.groupby(level=["grid_x", "grid_y"])
        .sum()
        .rename("training_departures")
        .reset_index()
    )
    by_grid = by_grid.sort_values(
        ["training_departures", "grid_x", "grid_y"],
        ascending=[False, True, True],
    ).head(top_n)
    if len(by_grid) < top_n:
        raise ValueError(f"有效网格只有 {len(by_grid)} 个，无法选择 Top {top_n}")

    by_grid["grid_id"] = _grid_id(by_grid["grid_x"], by_grid["grid_y"])
    by_grid["demand_rank"] = np.arange(1, len(by_grid) + 1, dtype="int16")
    by_grid["x_min_m"] = by_grid["grid_x"] * grid_size_m
    by_grid["y_min_m"] = by_grid["grid_y"] * grid_size_m
    by_grid["center_easting_m"] = by_grid["x_min_m"] + grid_size_m / 2.0
    by_grid["center_northing_m"] = by_grid["y_min_m"] + grid_size_m / 2.0
    inverse = Transformer.from_crs(
        UTM18N_EPSG, WGS84_EPSG, always_xy=True
    )
    center_lon, center_lat = inverse.transform(
        by_grid["center_easting_m"].to_numpy(),
        by_grid["center_northing_m"].to_numpy(),
    )
    by_grid["center_lat"] = center_lat
    by_grid["center_lon"] = center_lon
    return by_grid[
        [
            "demand_rank",
            "grid_id",
            "grid_x",
            "grid_y",
            "center_lat",
            "center_lon",
            "training_departures",
        ]
    ].reset_index(drop=True)


def build_hourly_panel(
    counts: pd.Series,
    top_grids: pd.DataFrame,
    *,
    start_time: str | pd.Timestamp,
    end_time: str | pd.Timestamp,
) -> pd.DataFrame:
    """Build a complete grid-hour Cartesian panel and fill no-trip hours with 0."""

    start = pd.Timestamp(start_time)
    end = pd.Timestamp(end_time)
    if start >= end:
        raise ValueError("面板开始时间必须早于结束时间")

    grid_lookup = top_grids.set_index(["grid_x", "grid_y"])["grid_id"]
    observed = counts.rename("demand").reset_index()
    observed = observed.merge(
        grid_lookup.rename("grid_id").reset_index(),
        on=["grid_x", "grid_y"],
        how="inner",
        validate="many_to_one",
    )
    observed = observed.loc[
        observed["timestamp"].ge(start) & observed["timestamp"].lt(end)
    ]
    observed_counts = observed.groupby(
        ["grid_id", "timestamp"], observed=True
    )["demand"].sum()

    hours = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    grid_ids = top_grids.sort_values("demand_rank")["grid_id"].tolist()
    complete_index = pd.MultiIndex.from_product(
        [grid_ids, hours], names=["grid_id", "timestamp"]
    )
    panel = observed_counts.reindex(complete_index, fill_value=0).rename("demand")
    panel = panel.reset_index()
    panel["demand"] = panel["demand"].astype("int32")
    coordinates = top_grids.set_index("grid_id")[["grid_x", "grid_y"]]
    panel = panel.join(coordinates, on="grid_id", validate="many_to_one")
    return panel.sort_values(["grid_id", "timestamp"]).reset_index(drop=True)


def engineer_features(
    panel: pd.DataFrame, grid_order: Sequence[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Create calendar and strictly lagged demand features."""

    required = {"grid_id", "timestamp", "demand", "grid_x", "grid_y"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError("小时面板缺少字段: " + ", ".join(missing))

    frame = panel.copy().sort_values(["grid_id", "timestamp"]).reset_index(
        drop=True
    )
    timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamp.isna().any():
        raise ValueError("小时面板包含无效时间")
    frame["timestamp"] = timestamp
    frame["hour"] = timestamp.dt.hour.astype("int8")
    frame["weekday"] = timestamp.dt.weekday.astype("int8")
    frame["month"] = timestamp.dt.month.astype("int8")
    frame["day_of_year"] = timestamp.dt.dayofyear.astype("int16")
    frame["is_weekend"] = frame["weekday"].ge(5).astype("int8")

    holidays = USFederalHolidayCalendar().holidays(
        start=timestamp.min().normalize(), end=timestamp.max().normalize()
    )
    frame["is_federal_holiday"] = (
        timestamp.dt.normalize().isin(holidays).astype("int8")
    )
    frame["is_workday"] = (
        frame["is_weekend"].eq(0) & frame["is_federal_holiday"].eq(0)
    ).astype("int8")
    frame["is_morning_peak"] = frame["hour"].between(7, 9).astype("int8")
    frame["is_evening_peak"] = frame["hour"].between(16, 18).astype("int8")
    frame["is_night"] = (
        frame["hour"].lt(6) | frame["hour"].ge(22)
    ).astype("int8")

    frame["hour_sin"] = np.sin(2.0 * math.pi * frame["hour"] / 24.0)
    frame["hour_cos"] = np.cos(2.0 * math.pi * frame["hour"] / 24.0)
    frame["weekday_sin"] = np.sin(2.0 * math.pi * frame["weekday"] / 7.0)
    frame["weekday_cos"] = np.cos(2.0 * math.pi * frame["weekday"] / 7.0)
    frame["month_sin"] = np.sin(2.0 * math.pi * (frame["month"] - 1) / 12.0)
    frame["month_cos"] = np.cos(2.0 * math.pi * (frame["month"] - 1) / 12.0)

    group = frame.groupby("grid_id", sort=False)["demand"]
    for lag in LAG_HOURS:
        frame[f"lag_{lag}"] = group.shift(lag)
    shifted = group.shift(1)
    shifted_by_grid = shifted.groupby(frame["grid_id"], sort=False)
    for window in ROLLING_WINDOWS:
        frame[f"rolling_mean_{window}"] = shifted_by_grid.transform(
            lambda values, size=window: values.rolling(
                size, min_periods=1
            ).mean()
        )
        frame[f"rolling_std_{window}"] = shifted_by_grid.transform(
            lambda values, size=window: values.rolling(
                size, min_periods=1
            ).std(ddof=0)
        )
    frame["grid_expanding_mean"] = shifted_by_grid.transform(
        lambda values: values.expanding(min_periods=1).mean()
    )

    ordered_grids = list(grid_order)
    unknown = sorted(set(frame["grid_id"]) - set(ordered_grids))
    if unknown:
        raise ValueError("grid_order 缺少网格: " + ", ".join(unknown))
    for index, grid_id in enumerate(ordered_grids):
        frame[f"grid_{index}"] = frame["grid_id"].eq(grid_id).astype("int8")

    feature_columns = [
        "hour",
        "weekday",
        "month",
        "day_of_year",
        "is_weekend",
        "is_federal_holiday",
        "is_workday",
        "is_morning_peak",
        "is_evening_peak",
        "is_night",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "grid_x",
        "grid_y",
        *[f"grid_{index}" for index in range(len(ordered_grids))],
        *[f"lag_{lag}" for lag in LAG_HOURS],
        *[
            name
            for window in ROLLING_WINDOWS
            for name in (f"rolling_mean_{window}", f"rolling_std_{window}")
        ],
        "grid_expanding_mean",
    ]
    return frame, feature_columns


def regression_metrics(
    actual: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> dict[str, float | int]:
    """Return count-regression metrics with an explicit zero-demand policy."""

    y_true = np.asarray(actual, dtype="float64")
    y_pred = np.asarray(predicted, dtype="float64")
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = np.clip(y_pred[valid], 0.0, None)
    if not len(y_true):
        raise ValueError("没有可计算指标的预测值")

    absolute_error = np.abs(y_true - y_pred)
    positive = y_true > 0
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape_terms = np.divide(
        2.0 * absolute_error,
        denominator,
        out=np.zeros_like(absolute_error),
        where=denominator > 0,
    )
    return {
        "n": int(len(y_true)),
        "positive_actual_n": int(positive.sum()),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mape_pct": (
            float(np.mean(absolute_error[positive] / y_true[positive]) * 100.0)
            if positive.any()
            else float("nan")
        ),
        "wape_pct": (
            float(absolute_error.sum() / y_true.sum() * 100.0)
            if y_true.sum() > 0
            else float("nan")
        ),
        "smape_pct": float(smape_terms.mean() * 100.0),
    }


def fit_historical_mean(train: pd.DataFrame) -> dict[str, object]:
    group = train.groupby(["grid_id", "weekday", "hour"], observed=True)[
        "demand"
    ].mean()
    grid = train.groupby("grid_id", observed=True)["demand"].mean()
    return {
        "group": group,
        "grid": grid,
        "global": float(train["demand"].mean()),
    }


def predict_historical_mean(
    rows: pd.DataFrame, fitted: dict[str, object]
) -> np.ndarray:
    group = fitted["group"]
    grid = fitted["grid"]
    assert isinstance(group, pd.Series)
    assert isinstance(grid, pd.Series)
    keys = pd.MultiIndex.from_frame(rows[["grid_id", "weekday", "hour"]])
    # ``Series.to_numpy`` may expose a read-only view under Pandas' copy-on-write
    # mode.  A writable copy is required for the hierarchical fallback below.
    predictions = group.reindex(keys).to_numpy(dtype="float64").copy()
    missing = ~np.isfinite(predictions)
    if missing.any():
        predictions[missing] = (
            rows.loc[missing, "grid_id"].map(grid).to_numpy(dtype="float64")
        )
    predictions = np.nan_to_num(
        predictions, nan=float(fitted["global"]), posinf=0.0, neginf=0.0
    )
    return np.clip(predictions, 0.0, None)


def _model_matrix(rows: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    return rows.loc[:, list(feature_columns)].to_numpy(dtype="float32")


def build_main_model(
    parameters: dict[str, object], *, random_state: int = 42
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson",
        early_stopping=False,
        random_state=random_state,
        **parameters,
    )


def _append_metric(
    rows: list[dict[str, object]],
    *,
    fold: str,
    model: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    fit_seconds: float,
) -> None:
    rows.append(
        {
            "fold": fold,
            "model": model,
            "fit_seconds": fit_seconds,
            **regression_metrics(actual, predicted),
        }
    )


def run_rolling_validation(
    featured: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    candidates: Sequence[tuple[str, dict[str, object]]] = MODEL_CANDIDATES,
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, object]]:
    if not candidates:
        raise ValueError("至少需要一个主模型参数候选")

    metric_rows: list[dict[str, object]] = []
    for fold in VALIDATION_FOLDS:
        train = featured.loc[
            featured["timestamp"].ge(fold.train_start)
            & featured["timestamp"].lt(fold.train_end)
        ].copy()
        validation = featured.loc[
            featured["timestamp"].ge(fold.validation_start)
            & featured["timestamp"].lt(fold.validation_end)
        ].copy()
        if train.empty or validation.empty:
            raise ValueError(f"滚动窗口 {fold.name} 没有完整数据")

        actual = validation["demand"].to_numpy(dtype="float64")
        weekly = validation["lag_168"].to_numpy(dtype="float64")
        _append_metric(
            metric_rows,
            fold=fold.name,
            model="weekly_naive",
            actual=actual,
            predicted=weekly,
            fit_seconds=0.0,
        )

        start = time.perf_counter()
        mean_model = fit_historical_mean(train)
        mean_prediction = predict_historical_mean(validation, mean_model)
        _append_metric(
            metric_rows,
            fold=fold.name,
            model="historical_mean",
            actual=actual,
            predicted=mean_prediction,
            fit_seconds=time.perf_counter() - start,
        )

        ready_train = train.dropna(subset=list(feature_columns))
        ready_validation = validation.dropna(subset=list(feature_columns))
        if len(ready_train) == 0 or len(ready_validation) != len(validation):
            raise ValueError(f"滚动窗口 {fold.name} 的模型特征不完整")
        train_x = _model_matrix(ready_train, feature_columns)
        train_y = ready_train["demand"].to_numpy(dtype="float64")
        validation_x = _model_matrix(ready_validation, feature_columns)

        for candidate_name, parameters in candidates:
            start = time.perf_counter()
            model = build_main_model(parameters, random_state=random_state)
            model.fit(train_x, train_y)
            prediction = np.clip(model.predict(validation_x), 0.0, None)
            _append_metric(
                metric_rows,
                fold=fold.name,
                model=f"hist_gradient_boosting[{candidate_name}]",
                actual=actual,
                predicted=prediction,
                fit_seconds=time.perf_counter() - start,
            )
            if verbose:
                print(
                    f"  {fold.name}: {candidate_name} complete "
                    f"({metric_rows[-1]['rmse']:.3f} RMSE)"
                )

    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mape_pct=("mape_pct", "mean"),
            mean_wape_pct=("wape_pct", "mean"),
            mean_smape_pct=("smape_pct", "mean"),
            total_fit_seconds=("fit_seconds", "sum"),
        )
        .sort_values(["mean_rmse", "mean_mae", "model"])
        .reset_index(drop=True)
    )
    candidate_labels = {
        f"hist_gradient_boosting[{name}]": parameters
        for name, parameters in candidates
    }
    candidate_summary = summary.loc[summary["model"].isin(candidate_labels)]
    best_label = str(candidate_summary.iloc[0]["model"])
    best_name = best_label.removeprefix("hist_gradient_boosting[").removesuffix("]")
    return metrics, summary, best_name, dict(candidate_labels[best_label])


def _metric_table_for_predictions(
    predictions: pd.DataFrame, prediction_columns: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    by_grid_rows: list[dict[str, object]] = []
    month = predictions["timestamp"].dt.to_period("M").astype(str)
    periods = [("overall", pd.Series(True, index=predictions.index))]
    periods.extend(
        (value, month.eq(value)) for value in sorted(month.unique().tolist())
    )
    for model_name, column in prediction_columns.items():
        for period, mask in periods:
            rows.append(
                {
                    "model": model_name,
                    "period": period,
                    **regression_metrics(
                        predictions.loc[mask, "actual_demand"],
                        predictions.loc[mask, column],
                    ),
                }
            )
        for grid_id, group in predictions.groupby("grid_id", observed=True):
            by_grid_rows.append(
                {
                    "model": model_name,
                    "grid_id": grid_id,
                    **regression_metrics(group["actual_demand"], group[column]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(by_grid_rows)


def train_forecasting_models(
    panel: pd.DataFrame,
    top_grids: pd.DataFrame,
    *,
    candidate_count: int = len(MODEL_CANDIDATES),
    random_state: int = 42,
    run_test: bool = True,
    test_start_time: str | pd.Timestamp = "2026-01-01",
    test_end_time: str | pd.Timestamp = "2026-08-01",
    verbose: bool = False,
) -> ForecastTrainingResult:
    grid_order = top_grids.sort_values("demand_rank")["grid_id"].tolist()
    featured, feature_columns = engineer_features(panel, grid_order)
    candidates = MODEL_CANDIDATES[:candidate_count]
    validation_metrics, validation_summary, best_name, best_parameters = (
        run_rolling_validation(
            featured,
            feature_columns,
            candidates=candidates,
            random_state=random_state,
            verbose=verbose,
        )
    )

    best_hgb_label = f"hist_gradient_boosting[{best_name}]"
    comparison = validation_summary.loc[
        validation_summary["model"].isin(
            ["weekly_naive", "historical_mean", best_hgb_label]
        )
    ].sort_values(["mean_rmse", "mean_mae", "model"])
    selected_validation_label = str(comparison.iloc[0]["model"])
    selected_model = (
        "hist_gradient_boosting"
        if selected_validation_label == best_hgb_label
        else selected_validation_label
    )

    final_train = featured.loc[
        featured["timestamp"].ge("2025-01-01")
        & featured["timestamp"].lt("2026-01-01")
    ].copy()
    ready_train = final_train.dropna(subset=feature_columns)
    main_model = build_main_model(best_parameters, random_state=random_state)
    main_model.fit(
        _model_matrix(ready_train, feature_columns),
        ready_train["demand"].to_numpy(dtype="float64"),
    )
    historical_fitted = fit_historical_mean(final_train)
    historical_group_means = (
        historical_fitted["group"].rename("mean_demand").reset_index()
    )

    predictions: pd.DataFrame | None = None
    test_metrics: pd.DataFrame | None = None
    test_metrics_by_grid: pd.DataFrame | None = None
    parsed_test_start: pd.Timestamp | None = None
    parsed_test_end: pd.Timestamp | None = None
    if run_test:
        parsed_test_start = pd.Timestamp(test_start_time)
        parsed_test_end = pd.Timestamp(test_end_time)
        if parsed_test_start >= parsed_test_end:
            raise ValueError("测试期开始时间必须早于结束时间")
        test = featured.loc[
            featured["timestamp"].ge(parsed_test_start)
            & featured["timestamp"].lt(parsed_test_end)
        ].copy()
        if test.empty:
            raise ValueError(
                f"没有 {parsed_test_start:%Y-%m-%d} 至 "
                f"{parsed_test_end:%Y-%m-%d} 的数据可进行时间外测试"
            )
        if test[feature_columns].isna().any().any():
            raise ValueError("2026时间外测试特征不完整")
        predictions = test[["timestamp", "grid_id", "demand"]].rename(
            columns={"demand": "actual_demand"}
        )
        predictions["weekly_naive_prediction"] = np.clip(
            test["lag_168"].to_numpy(dtype="float64"), 0.0, None
        )
        predictions["historical_mean_prediction"] = predict_historical_mean(
            test, historical_fitted
        )
        predictions["hist_gradient_boosting_prediction"] = np.clip(
            main_model.predict(_model_matrix(test, feature_columns)), 0.0, None
        )
        prediction_columns = {
            "weekly_naive": "weekly_naive_prediction",
            "historical_mean": "historical_mean_prediction",
            "hist_gradient_boosting": "hist_gradient_boosting_prediction",
        }
        predictions["selected_prediction"] = predictions[
            prediction_columns[selected_model]
        ]
        predictions["selected_model"] = selected_model
        predictions["residual"] = (
            predictions["selected_prediction"] - predictions["actual_demand"]
        )
        test_metrics, test_metrics_by_grid = _metric_table_for_predictions(
            predictions, prediction_columns
        )

    return ForecastTrainingResult(
        panel=featured,
        feature_columns=feature_columns,
        top_grids=top_grids,
        validation_metrics=validation_metrics,
        validation_summary=validation_summary,
        best_candidate_name=best_name,
        best_candidate_parameters=best_parameters,
        selected_model=selected_model,
        main_model=main_model,
        historical_group_means=historical_group_means,
        test_predictions=predictions,
        test_metrics=test_metrics,
        test_metrics_by_grid=test_metrics_by_grid,
        test_start_time=parsed_test_start,
        test_end_time=parsed_test_end,
    )


def _check_output_targets(output_dir: Path, overwrite: bool) -> None:
    expected = [
        output_dir / "tables" / "top10_forecast_grids.csv",
        output_dir / "tables" / "validation_metrics.csv",
        output_dir / "models" / "hist_gradient_boosting.joblib",
        output_dir / "forecast_training_metadata.json",
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "预测输出已存在；如需重跑请使用 --overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def _create_forecast_figures(
    result: ForecastTrainingResult, output_dir: Path
) -> list[Path]:
    if (
        result.test_predictions is None
        or result.test_metrics is None
        or result.test_metrics_by_grid is None
    ):
        return []
    matplotlib_cache = output_dir / ".matplotlib-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    predictions = result.test_predictions.copy()
    paths = [
        figure_dir / "01_actual_vs_predicted_daily.png",
        figure_dir / "02_validation_rmse_comparison.png",
        figure_dir / "03_selected_model_grid_rmse.png",
        figure_dir / "04_weekday_hour_residual_heatmap.png",
        figure_dir / "05_test_overall_metrics_comparison.png",
        figure_dir / "06_test_monthly_rmse_by_model.png",
    ]

    daily = predictions.set_index("timestamp").groupby(pd.Grouper(freq="D"))[
        ["actual_demand", "selected_prediction"]
    ].sum()
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(daily.index, daily["actual_demand"], label="Actual", linewidth=1.5)
    ax.plot(
        daily.index,
        daily["selected_prediction"],
        label=f"Predicted ({result.selected_model})",
        linewidth=1.3,
        alpha=0.85,
    )
    ax.set_title("Daily Departures Across Top 10 Forecast Grids: Actual vs Predicted")
    ax.set_ylabel("Trips")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(paths[0], dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = result.validation_summary.sort_values("mean_rmse")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(summary["model"], summary["mean_rmse"], color="#2F6BFF")
    ax.invert_yaxis()
    ax.set_title("Rolling Validation RMSE by Model")
    ax.set_xlabel("Mean RMSE across three folds")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(paths[1], dpi=180, bbox_inches="tight")
    plt.close(fig)

    by_grid = result.test_metrics_by_grid.loc[
        result.test_metrics_by_grid["model"].eq(result.selected_model)
    ].sort_values("rmse")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(by_grid["grid_id"], by_grid["rmse"], color="#FF8A34")
    ax.set_title(f"2026 RMSE by Grid: {result.selected_model}")
    ax.set_xlabel("RMSE")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(paths[2], dpi=180, bbox_inches="tight")
    plt.close(fig)

    predictions["weekday"] = predictions["timestamp"].dt.weekday
    predictions["hour"] = predictions["timestamp"].dt.hour
    residual = predictions.pivot_table(
        index="weekday", columns="hour", values="residual", aggfunc="mean"
    ).reindex(index=range(7), columns=range(24), fill_value=0.0)
    bound = max(0.1, float(np.nanmax(np.abs(residual.to_numpy()))))
    fig, ax = plt.subplots(figsize=(13, 5))
    image = ax.imshow(
        residual.to_numpy(),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
    )
    ax.set_xticks(range(24), labels=range(24))
    ax.set_yticks(
        range(7), labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Weekday")
    ax.set_title("Mean Residual (Prediction - Actual) by Weekday and Hour")
    fig.colorbar(image, ax=ax, label="Trips")
    fig.tight_layout()
    fig.savefig(paths[3], dpi=180, bbox_inches="tight")
    plt.close(fig)

    model_labels = {
        "weekly_naive": "Weekly naive",
        "historical_mean": "Historical mean",
        "hist_gradient_boosting": "HistGradientBoosting",
    }
    model_colors = {
        "weekly_naive": "#7A8CA5",
        "historical_mean": "#FF8A34",
        "hist_gradient_boosting": "#2F6BFF",
    }
    overall = result.test_metrics.loc[
        result.test_metrics["period"].eq("overall")
    ].copy()
    overall["display_model"] = overall["model"].map(model_labels)
    metric_specs = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("mape_pct", "MAPE (%)"),
        ("wape_pct", "WAPE (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (column, title) in zip(axes.ravel(), metric_specs):
        colors = [model_colors[str(model)] for model in overall["model"]]
        bars = ax.bar(overall["display_model"], overall[column], color=colors)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("2026 Jan-Jul Out-of-Time Test Metrics", fontsize=15)
    fig.tight_layout()
    fig.savefig(paths[4], dpi=180, bbox_inches="tight")
    plt.close(fig)

    monthly = result.test_metrics.loc[
        ~result.test_metrics["period"].eq("overall")
    ].copy()
    month_order = sorted(monthly["period"].astype(str).unique().tolist())
    fig, ax = plt.subplots(figsize=(11, 6))
    for model_name in model_labels:
        values = (
            monthly.loc[monthly["model"].eq(model_name)]
            .set_index("period")
            .reindex(month_order)
        )
        ax.plot(
            month_order,
            values["rmse"],
            marker="o",
            linewidth=2,
            label=model_labels[model_name],
            color=model_colors[model_name],
        )
    ax.set_title("2026 Monthly Out-of-Time RMSE by Model")
    ax.set_xlabel("Month")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[5], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def save_training_outputs(
    result: ForecastTrainingResult,
    output_dir: Path,
    *,
    train_files: Sequence[Path],
    test_files: Sequence[Path],
    grid_size_m: int,
    overwrite: bool = False,
) -> list[Path]:
    _check_output_targets(output_dir, overwrite)
    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        table_dir / "top10_forecast_grids.csv",
        table_dir / "validation_metrics.csv",
        table_dir / "validation_summary.csv",
        table_dir / "historical_group_means.csv",
        table_dir / "feature_schema.csv",
        model_dir / "hist_gradient_boosting.joblib",
    ]
    result.top_grids.to_csv(targets[0], index=False)
    result.validation_metrics.to_csv(targets[1], index=False)
    result.validation_summary.to_csv(targets[2], index=False)
    result.historical_group_means.to_csv(targets[3], index=False)
    pd.DataFrame(
        {
            "feature": result.feature_columns,
            "dtype": [str(result.panel[column].dtype) for column in result.feature_columns],
        }
    ).to_csv(targets[4], index=False)
    joblib.dump(
        {
            "model": result.main_model,
            "feature_columns": result.feature_columns,
            "top_grids": result.top_grids,
            "best_candidate_name": result.best_candidate_name,
            "best_candidate_parameters": result.best_candidate_parameters,
        },
        targets[5],
    )

    if result.test_predictions is not None:
        prediction_path = table_dir / "forecast_predictions_2026.parquet"
        metric_path = table_dir / "test_metrics_by_model_and_month.csv"
        grid_metric_path = table_dir / "test_metrics_by_grid.csv"
        result.test_predictions.to_parquet(prediction_path, index=False)
        assert result.test_metrics is not None
        assert result.test_metrics_by_grid is not None
        result.test_metrics.to_csv(metric_path, index=False)
        result.test_metrics_by_grid.to_csv(grid_metric_path, index=False)
        targets.extend([prediction_path, metric_path, grid_metric_path])

    figure_paths = _create_forecast_figures(result, output_dir)
    targets.extend(figure_paths)
    metadata_path = output_dir / "forecast_training_metadata.json"
    overall_test = None
    if result.test_metrics is not None:
        overall_test = result.test_metrics.loc[
            result.test_metrics["period"].eq("overall")
        ].to_dict(orient="records")
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "weather_features_included": False,
        "target": "hourly departures per fixed high-demand 500 m grid",
        "prediction_protocol": (
            "rolling one-hour-ahead evaluation; target-hour features use only "
            "calendar information and demand observed through the prior hour"
        ),
        "grid_crs": f"EPSG:{UTM18N_EPSG}",
        "grid_size_m": grid_size_m,
        "top_grid_selection_period": "2025-01-01/2026-01-01",
        "top_grid_count": int(len(result.top_grids)),
        "train_files": [str(path.resolve()) for path in train_files],
        "test_files": [str(path.resolve()) for path in test_files],
        "validation_folds": [asdict(fold) for fold in VALIDATION_FOLDS],
        "best_main_model_candidate": result.best_candidate_name,
        "best_main_model_parameters": result.best_candidate_parameters,
        "selected_model_by_validation_rmse": result.selected_model,
        "test_period_start_inclusive": (
            result.test_start_time.isoformat() if result.test_start_time is not None else None
        ),
        "test_period_end_exclusive": (
            result.test_end_time.isoformat() if result.test_end_time is not None else None
        ),
        "feature_columns": result.feature_columns,
        "metrics": {
            "mape": "computed only where actual demand > 0",
            "wape": "sum absolute error / sum actual demand",
            "smape": "zero/zero contributes 0",
        },
        "overall_2026_test_metrics": overall_test,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    targets.append(metadata_path)
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capital Bikeshare Top 10 网格级下一小时需求预测（无天气）"
    )
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/forecasting"))
    parser.add_argument("--train-start-month", default="202501")
    parser.add_argument("--train-end-month", default="202512")
    parser.add_argument("--test-start-month", default="202601")
    parser.add_argument("--test-end-month", default="202607")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--grid-size-m", type=int, default=DEFAULT_GRID_SIZE_M)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--model-candidates",
        type=int,
        choices=range(1, len(MODEL_CANDIDATES) + 1),
        default=len(MODEL_CANDIDATES),
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_files = discover_cleaned_files(
        args.input_dir, args.train_start_month, args.train_end_month
    )
    require_complete_months(
        train_files, args.train_start_month, args.train_end_month, "训练期"
    )
    test_files: list[Path] = []
    if not args.skip_test:
        test_files = discover_cleaned_files(
            args.input_dir, args.test_start_month, args.test_end_month
        )
        require_complete_months(
            test_files, args.test_start_month, args.test_end_month, "测试期"
        )
    test_start_time, test_end_time = _month_time_bounds(
        args.test_start_month, args.test_end_month
    )

    print("Aggregating 2025 training departures...")
    training_counts = aggregate_hourly_departures(
        train_files, grid_size_m=args.grid_size_m, chunksize=args.chunksize
    )
    top_grids = select_top_grids(
        training_counts, top_n=args.top_n, grid_size_m=args.grid_size_m
    )
    all_counts = training_counts
    if test_files:
        print("Aggregating 2026 out-of-time departures...")
        test_counts = aggregate_hourly_departures(
            test_files, grid_size_m=args.grid_size_m, chunksize=args.chunksize
        )
        all_counts = all_counts.add(test_counts, fill_value=0).astype("int64")

    panel_end = test_end_time if test_files else pd.Timestamp("2026-01-01")
    panel = build_hourly_panel(
        all_counts,
        top_grids,
        start_time="2025-01-01",
        end_time=panel_end,
    )
    print(
        f"Training {args.model_candidates} main-model candidates on "
        f"{len(panel):,} grid-hour rows..."
    )
    result = train_forecasting_models(
        panel,
        top_grids,
        candidate_count=args.model_candidates,
        random_state=args.random_state,
        run_test=bool(test_files),
        test_start_time=test_start_time,
        test_end_time=test_end_time,
        verbose=True,
    )
    outputs = save_training_outputs(
        result,
        args.output_dir,
        train_files=train_files,
        test_files=test_files,
        grid_size_m=args.grid_size_m,
        overwrite=args.overwrite,
    )
    print("\nForecast training complete:")
    print(f"  selected model: {result.selected_model}")
    print(f"  best HGB candidate: {result.best_candidate_name}")
    if result.test_metrics is not None:
        overall = result.test_metrics.loc[
            result.test_metrics["period"].eq("overall")
        ]
        print(overall[["model", "mae", "rmse", "mape_pct", "wape_pct"]].to_string(index=False))
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
