"""Temporal and user-type analysis for cleaned Capital Bikeshare trips."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .paths import PROCESSED_DIR, _validate_month


REQUIRED_ANALYSIS_COLUMNS = (
    "start_time",
    "user_type",
    "trip_duration_s",
    "distance_km",
)
PROCESSED_FILE_PATTERN = re.compile(
    r"^(?P<month>\d{6})-trips-cleaned\.(?P<format>parquet|csv)$"
)
FREQUENCY_LABELS = {
    "hour": "Hourly",
    "day": "Daily",
    "week": "Weekly",
    "month": "Monthly",
}


@dataclass
class TemporalAnalysisResult:
    distance_over_time: pd.DataFrame
    hourly_average_volume: pd.DataFrame
    daily_volume: pd.DataFrame
    user_type_summary: pd.DataFrame
    duration_samples: dict[str, np.ndarray]
    input_rows: int
    analysis_rows: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp


def discover_cleaned_files(
    input_dir: Path = PROCESSED_DIR,
    start_month: str | None = None,
    end_month: str | None = None,
) -> list[Path]:
    """Discover one cleaned file per month, preferring Parquet over CSV."""

    start_month = _validate_month(start_month)
    end_month = _validate_month(end_month)
    if start_month and end_month and start_month > end_month:
        raise ValueError("start_month 不能晚于 end_month")
    if not input_dir.exists():
        return []

    selected: dict[str, Path] = {}
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        match = PROCESSED_FILE_PATTERN.fullmatch(path.name)
        if not match:
            continue
        month = match.group("month")
        if start_month and month < start_month:
            continue
        if end_month and month > end_month:
            continue
        existing = selected.get(month)
        if existing is None or path.suffix == ".parquet":
            selected[month] = path

    return [selected[month] for month in sorted(selected)]


def _read_cleaned_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    if path.suffix == ".csv":
        header = pd.read_csv(path, nrows=0).columns
        missing = sorted(set(REQUIRED_ANALYSIS_COLUMNS) - set(header))
        if missing:
            raise ValueError(f"{path.name} 缺少字段: {', '.join(missing)}")
        yield from pd.read_csv(
            path,
            usecols=list(REQUIRED_ANALYSIS_COLUMNS),
            dtype={"user_type": "string"},
            chunksize=chunksize,
            low_memory=False,
        )
        return

    try:
        frame = pd.read_parquet(path, columns=list(REQUIRED_ANALYSIS_COLUMNS))
    except ImportError as exc:
        raise RuntimeError(
            "读取 Parquet 需要 pyarrow；请执行 pip install -r requirements.txt"
        ) from exc
    missing = sorted(set(REQUIRED_ANALYSIS_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} 缺少字段: {', '.join(missing)}")
    for start in range(0, len(frame), chunksize):
        yield frame.iloc[start : start + chunksize].copy()


def _time_bucket(times: pd.Series, frequency: str) -> pd.Series:
    if frequency == "hour":
        return times.dt.floor("h")
    if frequency == "day":
        return times.dt.normalize()
    if frequency == "week":
        return times.dt.to_period("W-SUN").dt.start_time
    if frequency == "month":
        return times.dt.to_period("M").dt.start_time
    raise ValueError(f"不支持的聚合频率: {frequency}")


def _complete_time_index(
    start: pd.Timestamp, end: pd.Timestamp, frequency: str
) -> pd.DatetimeIndex:
    first = _time_bucket(pd.Series([start]), frequency).iloc[0]
    last = _time_bucket(pd.Series([end]), frequency).iloc[0]
    pandas_frequency = {
        "hour": "h",
        "day": "D",
        "week": "W-MON",
        "month": "MS",
    }[frequency]
    return pd.date_range(first, last, freq=pandas_frequency)


def _coerce_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_ANALYSIS_COLUMNS) - set(chunk.columns))
    if missing:
        raise ValueError("清洗数据缺少字段: " + ", ".join(missing))
    frame = chunk.loc[:, list(REQUIRED_ANALYSIS_COLUMNS)].copy()
    frame["start_time"] = pd.to_datetime(
        frame["start_time"], errors="coerce", format="mixed"
    )
    frame["user_type"] = (
        frame["user_type"].astype("string").str.strip().replace("", pd.NA)
    )
    frame["trip_duration_s"] = pd.to_numeric(
        frame["trip_duration_s"], errors="coerce"
    )
    frame["distance_km"] = pd.to_numeric(frame["distance_km"], errors="coerce")
    return frame


def _add_series(left: pd.Series | None, right: pd.Series) -> pd.Series:
    if left is None:
        return right.astype("float64")
    return left.add(right, fill_value=0.0)


def analyze_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    frequency: str = "day",
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
    max_box_samples: int = 50_000,
    random_seed: int = 42,
) -> TemporalAnalysisResult:
    """Aggregate cleaned chunks into the four required temporal analyses.

    ``end_time`` is exclusive. Absent dates and hours are filled with zero
    before calculating 24-hour averages and daily-volume distributions.
    """

    if frequency not in FREQUENCY_LABELS:
        raise ValueError("frequency 必须是 hour、day、week 或 month")
    if max_box_samples <= 0:
        raise ValueError("max_box_samples 必须大于 0")

    requested_start = None if start_time is None else pd.Timestamp(start_time)
    requested_end = None if end_time is None else pd.Timestamp(end_time)
    if requested_start is not None and requested_end is not None:
        if requested_start >= requested_end:
            raise ValueError("start_time 必须早于 end_time")

    distance_series: pd.Series | None = None
    date_hour_counts: pd.Series | None = None
    user_counts: pd.Series | None = None
    user_duration_sums: pd.Series | None = None
    user_distance_sums: pd.Series | None = None
    user_distance_counts: pd.Series | None = None
    duration_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    input_rows = 0
    analysis_rows = 0
    observed_start: pd.Timestamp | None = None
    observed_end: pd.Timestamp | None = None

    for raw_chunk in chunks:
        input_rows += len(raw_chunk)
        frame = _coerce_chunk(raw_chunk)
        valid = (
            frame["start_time"].notna()
            & frame["user_type"].notna()
            & frame["trip_duration_s"].notna()
        )
        if requested_start is not None:
            valid &= frame["start_time"].ge(requested_start)
        if requested_end is not None:
            valid &= frame["start_time"].lt(requested_end)
        frame = frame.loc[valid].copy()
        if frame.empty:
            continue

        analysis_rows += len(frame)
        chunk_start = frame["start_time"].min()
        chunk_end = frame["start_time"].max()
        observed_start = chunk_start if observed_start is None else min(observed_start, chunk_start)
        observed_end = chunk_end if observed_end is None else max(observed_end, chunk_end)

        frame["time_bucket"] = _time_bucket(frame["start_time"], frequency)
        distance_by_period = frame.groupby("time_bucket", observed=True)[
            "distance_km"
        ].sum(min_count=1).fillna(0.0)
        distance_series = _add_series(distance_series, distance_by_period)

        frame["start_date"] = frame["start_time"].dt.normalize()
        frame["start_hour"] = frame["start_time"].dt.hour
        chunk_date_hour = frame.groupby(
            ["start_date", "start_hour"], observed=True
        ).size()
        date_hour_counts = _add_series(date_hour_counts, chunk_date_hour)

        chunk_user_counts = frame.groupby("user_type", observed=True).size()
        chunk_duration_sums = frame.groupby("user_type", observed=True)[
            "trip_duration_s"
        ].sum()
        valid_distances = frame.loc[frame["distance_km"].notna()]
        chunk_distance_sums = valid_distances.groupby("user_type", observed=True)[
            "distance_km"
        ].sum()
        chunk_distance_counts = valid_distances.groupby(
            "user_type", observed=True
        ).size()
        user_counts = _add_series(user_counts, chunk_user_counts)
        user_duration_sums = _add_series(user_duration_sums, chunk_duration_sums)
        user_distance_sums = _add_series(user_distance_sums, chunk_distance_sums)
        user_distance_counts = _add_series(
            user_distance_counts, chunk_distance_counts
        )

        for user_type, values in frame.groupby("user_type", observed=True)[
            "trip_duration_s"
        ]:
            duration_parts[str(user_type)].append(values.to_numpy(dtype="float64"))

    if analysis_rows == 0 or observed_start is None or observed_end is None:
        raise ValueError("指定范围内没有可用于时间分析的清洗记录")

    calendar_start = (
        requested_start.normalize() if requested_start is not None else observed_start.normalize()
    )
    if requested_end is not None:
        calendar_end = (requested_end - pd.Timedelta(nanoseconds=1)).normalize()
        time_index_end = requested_end - pd.Timedelta(nanoseconds=1)
    else:
        calendar_end = observed_end.normalize()
        time_index_end = observed_end
    calendar_dates = pd.date_range(calendar_start, calendar_end, freq="D")
    if calendar_dates.empty:
        raise ValueError("分析日期范围为空")

    time_index_start = requested_start if requested_start is not None else observed_start
    time_index = _complete_time_index(time_index_start, time_index_end, frequency)
    distance_series = distance_series.reindex(time_index, fill_value=0.0).sort_index()
    distance_over_time = pd.DataFrame(
        {
            "period_start": distance_series.index,
            "total_distance_km": distance_series.to_numpy(dtype="float64"),
        }
    )

    full_date_hour_index = pd.MultiIndex.from_product(
        [calendar_dates, range(24)], names=["start_date", "hour"]
    )
    date_hour_counts = date_hour_counts.reindex(
        full_date_hour_index, fill_value=0.0
    ).astype("int64")
    hourly_average = date_hour_counts.groupby(level="hour").mean()
    hourly_totals = date_hour_counts.groupby(level="hour").sum()
    hourly_average_volume = pd.DataFrame(
        {
            "hour": range(24),
            "average_trips_per_day": hourly_average.reindex(range(24)).to_numpy(),
            "total_trips": hourly_totals.reindex(range(24)).to_numpy(dtype="int64"),
        }
    )

    daily_counts = date_hour_counts.groupby(level="start_date").sum().reindex(
        calendar_dates, fill_value=0
    )
    holiday_dates = USFederalHolidayCalendar().holidays(
        start=calendar_dates.min(), end=calendar_dates.max()
    )
    is_weekend = calendar_dates.dayofweek >= 5
    is_holiday = calendar_dates.isin(holiday_dates)
    is_workday = ~is_weekend & ~is_holiday
    daily_volume = pd.DataFrame(
        {
            "date": calendar_dates,
            "trip_count": daily_counts.to_numpy(dtype="int64"),
            "is_weekend": is_weekend,
            "is_federal_holiday": is_holiday,
            "is_workday": is_workday,
            "day_type": np.where(is_workday, "Workday", "Weekend / holiday"),
        }
    )

    number_of_days = len(calendar_dates)
    total_trip_count = int(user_counts.sum())
    user_rows: list[dict[str, float | int | str]] = []
    duration_samples: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(random_seed)
    for user_type in sorted(user_counts.index.astype(str)):
        durations_s = np.concatenate(duration_parts[user_type])
        durations_min = durations_s / 60.0
        if len(durations_min) > max_box_samples:
            sample_indices = rng.choice(
                len(durations_min), size=max_box_samples, replace=False
            )
            duration_samples[user_type] = durations_min[sample_indices]
        else:
            duration_samples[user_type] = durations_min

        count = int(user_counts.get(user_type, 0))
        distance_count = int(user_distance_counts.get(user_type, 0))
        distance_sum = float(user_distance_sums.get(user_type, 0.0))
        user_rows.append(
            {
                "user_type": user_type,
                "trip_count": count,
                "trip_share_pct": 100.0 * count / total_trip_count,
                "average_daily_trips": count / number_of_days,
                "mean_duration_min": float(
                    user_duration_sums.get(user_type, 0.0) / count / 60.0
                ),
                "median_duration_min": float(np.median(durations_min)),
                "p75_duration_min": float(np.percentile(durations_min, 75)),
                "p95_duration_min": float(np.percentile(durations_min, 95)),
                "distance_valid_trips": distance_count,
                "mean_distance_km": (
                    distance_sum / distance_count if distance_count else math.nan
                ),
            }
        )
    user_type_summary = pd.DataFrame(user_rows).sort_values(
        "trip_count", ascending=False
    )

    return TemporalAnalysisResult(
        distance_over_time=distance_over_time,
        hourly_average_volume=hourly_average_volume,
        daily_volume=daily_volume,
        user_type_summary=user_type_summary,
        duration_samples=duration_samples,
        input_rows=input_rows,
        analysis_rows=analysis_rows,
        start_time=observed_start,
        end_time=observed_end,
    )


def analyze_cleaned_files(
    files: list[Path],
    *,
    chunksize: int = 100_000,
    frequency: str = "day",
    start_time: str | None = None,
    end_time: str | None = None,
    max_box_samples: int = 50_000,
) -> TemporalAnalysisResult:
    if not files:
        raise ValueError("没有找到清洗后的月度 CSV/Parquet 文件")
    if chunksize <= 0:
        raise ValueError("chunksize 必须大于 0")

    def chunks() -> Iterator[pd.DataFrame]:
        for path in files:
            print(f"Reading {path.name}")
            yield from _read_cleaned_chunks(path, chunksize)

    return analyze_chunks(
        chunks(),
        frequency=frequency,
        start_time=start_time,
        end_time=end_time,
        max_box_samples=max_box_samples,
    )


def _format_count(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def create_temporal_outputs(
    result: TemporalAnalysisResult,
    output_dir: Path,
    *,
    frequency: str = "day",
    overwrite: bool = False,
) -> list[Path]:
    """Write four figures, four tables, and run metadata."""

    # Some managed runtimes expose a read-only home directory. Keep Matplotlib's
    # font cache in the writable system temp directory to avoid noisy warnings.
    matplotlib_cache = Path(tempfile.gettempdir()) / "capital_bikeshare_mpl_cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        figure_dir / f"01_total_distance_{frequency}.png",
        figure_dir / "02_hourly_average_volume.png",
        figure_dir / "03_workday_weekend_daily_volume.png",
        figure_dir / "04_user_type_behavior.png",
        table_dir / f"distance_over_time_{frequency}.csv",
        table_dir / "hourly_average_volume.csv",
        table_dir / "daily_volume_by_day_type.csv",
        table_dir / "user_type_summary.csv",
        output_dir / "temporal_analysis_metadata.json",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"输出已存在（使用 --overwrite 替换）: {names}")

    plt.style.use("seaborn-v0_8-whitegrid")
    blue = "#2F6BFF"
    orange = "#FF8A34"
    dark = "#25324A"
    green = "#2AA876"

    # 1. Total straight-line distance over time.
    fig, ax = plt.subplots(figsize=(12, 5.5))
    distance = result.distance_over_time
    ax.plot(
        distance["period_start"],
        distance["total_distance_km"],
        color=blue,
        linewidth=1.8,
    )
    ax.fill_between(
        distance["period_start"],
        distance["total_distance_km"],
        color=blue,
        alpha=0.10,
    )
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: _format_count(x)))
    ax.set_title(f"{FREQUENCY_LABELS[frequency]} Total Straight-line Distance")
    ax.set_xlabel("Ride start time")
    ax.set_ylabel("Total distance estimate (km)")
    ax.text(
        0,
        -0.20,
        "Haversine displacement estimate; records flagged by distance/speed checks are excluded.",
        transform=ax.transAxes,
        fontsize=9,
        color="#5C667A",
    )
    fig.tight_layout()
    fig.savefig(targets[0], dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2. Mean hourly trip volume, with zero-volume hours included.
    hourly = result.hourly_average_volume
    hour_colors = [
        dark
        if hour < 6 or hour >= 22
        else orange
        if 7 <= hour < 10
        else "#E95D76"
        if 16 <= hour < 19
        else green
        for hour in hourly["hour"]
    ]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(
        hourly["hour"],
        hourly["average_trips_per_day"],
        color=hour_colors,
        width=0.82,
    )
    ax.set_xticks(range(24))
    ax.set_title("Average Trip Volume by Start Hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average trips per day")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: _format_count(x)))
    peak_index = int(hourly["average_trips_per_day"].idxmax())
    peak_bar = bars[peak_index]
    ax.annotate(
        f"Peak: {int(hourly.loc[peak_index, 'hour']):02d}:00",
        xy=(peak_bar.get_x() + peak_bar.get_width() / 2, peak_bar.get_height()),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(targets[1], dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3. Distribution of daily trip volume by workday status.
    categories = ["Workday", "Weekend / holiday"]
    values = [
        result.daily_volume.loc[
            result.daily_volume["day_type"].eq(category), "trip_count"
        ].to_numpy()
        for category in categories
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    box = ax.boxplot(
        values,
        tick_labels=categories,
        patch_artist=True,
        showfliers=False,
        widths=0.5,
    )
    for patch, color in zip(box["boxes"], [blue, orange]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    rng = np.random.default_rng(42)
    for position, (category_values, color) in enumerate(
        zip(values, [blue, orange]), start=1
    ):
        jitter = rng.normal(position, 0.035, size=len(category_values))
        ax.scatter(
            jitter,
            category_values,
            color=color,
            s=14,
            alpha=0.35,
            edgecolors="none",
        )
    ax.set_title("Daily Trip Volume: Workdays vs Weekends / Holidays")
    ax.set_xlabel("")
    ax.set_ylabel("Trips per day")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: _format_count(x)))
    fig.tight_layout()
    fig.savefig(targets[2], dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 4. User-type frequency and duration distribution.
    summary = result.user_type_summary.reset_index(drop=True)
    user_types = summary["user_type"].tolist()
    palette = [blue, orange, green, dark]
    colors = [palette[index % len(palette)] for index in range(len(user_types))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    bars = axes[0].bar(
        user_types, summary["average_daily_trips"], color=colors, width=0.62
    )
    axes[0].set_title("Usage Frequency by User Type")
    axes[0].set_ylabel("Average trips per day")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: _format_count(x)))
    for bar, share in zip(bars, summary["trip_share_pct"]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{share:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    duration_values = [result.duration_samples[user_type] for user_type in user_types]
    duration_box = axes[1].boxplot(
        duration_values,
        tick_labels=user_types,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        widths=0.55,
    )
    for patch, color in zip(duration_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1].set_title("Trip Duration by User Type")
    axes[1].set_ylabel("Duration (minutes)")
    fig.text(
        0.5,
        0.01,
        "Frequency denotes trips by user type; the dataset contains no individual user IDs. "
        "Duration plot uses a reproducible sample when a group exceeds the configured limit.",
        ha="center",
        fontsize=8.5,
        color="#5C667A",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(targets[3], dpi=180, bbox_inches="tight")
    plt.close(fig)

    result.distance_over_time.to_csv(targets[4], index=False)
    result.hourly_average_volume.to_csv(targets[5], index=False)
    result.daily_volume.to_csv(targets[6], index=False)
    result.user_type_summary.to_csv(targets[7], index=False)
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": result.input_rows,
        "analysis_rows": result.analysis_rows,
        "observed_start_time": result.start_time.isoformat(),
        "observed_end_time": result.end_time.isoformat(),
        "distance_frequency": frequency,
        "distance_definition": "Haversine straight-line displacement estimate",
        "workday_definition": "Monday-Friday excluding U.S. federal holidays",
        "frequency_definition": "Trip count by user type; no individual user IDs available",
    }
    targets[8].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capital Bikeshare 时间维度与用户类型分析"
    )
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/temporal"))
    parser.add_argument("--start-month", help="清洗文件起始月份 YYYYMM（含）")
    parser.add_argument("--end-month", help="清洗文件结束月份 YYYYMM（含）")
    parser.add_argument("--start-time", help="分析起点（含）")
    parser.add_argument("--end-time", help="分析终点（不含）")
    parser.add_argument(
        "--frequency", choices=tuple(FREQUENCY_LABELS), default="day"
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-box-samples", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = discover_cleaned_files(
        args.input_dir,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    result = analyze_cleaned_files(
        files,
        chunksize=args.chunksize,
        frequency=args.frequency,
        start_time=args.start_time,
        end_time=args.end_time,
        max_box_samples=args.max_box_samples,
    )
    outputs = create_temporal_outputs(
        result,
        args.output_dir,
        frequency=args.frequency,
        overwrite=args.overwrite,
    )
    print("\nTemporal analysis complete:")
    print(f"  rows: {result.analysis_rows:,}")
    print(f"  period: {result.start_time} to {result.end_time}")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
