"""Configuration for Capital Bikeshare preprocessing."""

from __future__ import annotations

from dataclasses import dataclass


# Capital Bikeshare's source column names are normalized to the names used in
# the assignment brief.  Keeping the mapping in one place makes later analysis
# independent of the vendor-specific schema.
SOURCE_COLUMN_MAP = {
    "started_at": "start_time",
    "ended_at": "end_time",
    "start_lng": "start_lon",
    "end_lng": "end_lon",
    "member_casual": "user_type",
}

REQUIRED_COLUMNS = (
    "ride_id",
    "start_time",
    "end_time",
    "start_station_id",
    "start_station_name",
    "start_lat",
    "start_lon",
    "end_station_id",
    "end_station_name",
    "end_lat",
    "end_lon",
    "user_type",
)

STRING_COLUMNS = (
    "ride_id",
    "rideable_type",
    "start_station_id",
    "start_station_name",
    "end_station_id",
    "end_station_name",
    "user_type",
)


@dataclass(frozen=True)
class CleaningConfig:
    """Thresholds and feature definitions used by the cleaning pipeline."""

    min_duration_s: float = 60.0
    max_duration_s: float = 4.0 * 60.0 * 60.0

    # A deliberately generous Washington, D.C. metropolitan-area envelope.
    min_latitude: float = 38.7
    max_latitude: float = 39.2
    min_longitude: float = -77.5
    max_longitude: float = -76.7

    max_straight_line_distance_km: float = 30.0
    max_speed_proxy_kmh: float = 35.0

    morning_peak_start: int = 7
    morning_peak_end: int = 10  # exclusive
    evening_peak_start: int = 16
    evening_peak_end: int = 19  # exclusive
    night_end: int = 6  # 00:00-05:59
    night_start: int = 22  # 22:00-23:59

    local_timezone: str = "America/New_York"

