from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import DATA_DIR, discover_trip_files
from src.preprocess import haversine_km, preprocess_dataframe
from src.temporal import analyze_chunks


class PreprocessTests(unittest.TestCase):
    def _raw_frame(self) -> pd.DataFrame:
        rows = [
            {
                "ride_id": "valid",
                "rideable_type": "classic_bike",
                "started_at": "2025-07-03 08:00:00",
                "ended_at": "2025-07-03 08:10:00",
                "start_station_name": "A",
                "start_station_id": "001",
                "end_station_name": "B",
                "end_station_id": "002",
                "start_lat": 38.90,
                "start_lng": -77.00,
                "end_lat": 38.91,
                "end_lng": -77.00,
                "member_casual": "member",
            },
            {
                "ride_id": pd.NA,
                "rideable_type": "classic_bike",
                "started_at": "2025-07-03 09:00:00",
                "ended_at": "2025-07-03 09:10:00",
                "start_station_name": "A",
                "start_station_id": "001",
                "end_station_name": "B",
                "end_station_id": "002",
                "start_lat": 38.90,
                "start_lng": -77.00,
                "end_lat": 38.91,
                "end_lng": -77.00,
                "member_casual": "member",
            },
            {
                "ride_id": "negative",
                "rideable_type": "classic_bike",
                "started_at": "2025-07-03 10:00:00",
                "ended_at": "2025-07-03 09:59:00",
                "start_station_name": "A",
                "start_station_id": "001",
                "end_station_name": "B",
                "end_station_id": "002",
                "start_lat": 38.90,
                "start_lng": -77.00,
                "end_lat": 38.91,
                "end_lng": -77.00,
                "member_casual": "member",
            },
            {
                "ride_id": "short",
                "rideable_type": "classic_bike",
                "started_at": "2025-07-03 10:00:00",
                "ended_at": "2025-07-03 10:00:30",
                "start_station_name": "A",
                "start_station_id": "001",
                "end_station_name": "B",
                "end_station_id": "002",
                "start_lat": 38.90,
                "start_lng": -77.00,
                "end_lat": 38.91,
                "end_lng": -77.00,
                "member_casual": "member",
            },
            {
                "ride_id": "long",
                "rideable_type": "classic_bike",
                "started_at": "2025-07-03 10:00:00",
                "ended_at": "2025-07-03 14:00:01",
                "start_station_name": "A",
                "start_station_id": "001",
                "end_station_name": "B",
                "end_station_id": "002",
                "start_lat": 38.90,
                "start_lng": -77.00,
                "end_lat": 38.91,
                "end_lng": -77.00,
                "member_casual": "member",
            },
            {
                "ride_id": "bad-coordinate",
                "rideable_type": "electric_bike",
                "started_at": "2025-07-04 17:00:00",
                "ended_at": "2025-07-04 17:15:00",
                "start_station_name": pd.NA,
                "start_station_id": pd.NA,
                "end_station_name": "B",
                "end_station_id": "002",
                "start_lat": 0.0,
                "start_lng": 0.0,
                "end_lat": 38.91,
                "end_lng": -77.00,
                "member_casual": "casual",
            },
        ]
        return pd.DataFrame(rows)

    def test_cleaning_and_features(self) -> None:
        cleaned, report = preprocess_dataframe(self._raw_frame())

        self.assertEqual(set(cleaned["ride_id"]), {"valid", "bad-coordinate"})
        self.assertEqual(report["missing_ride_id_rows"], 1)
        self.assertEqual(report["negative_duration_rows"], 1)
        self.assertEqual(report["too_short_duration_rows"], 1)
        self.assertEqual(report["too_long_duration_rows"], 1)

        valid = cleaned.set_index("ride_id").loc["valid"]
        self.assertEqual(valid["trip_duration_s"], 600.0)
        self.assertEqual(valid["start_hour"], 8)
        self.assertEqual(valid["start_weekday"], 3)
        self.assertEqual(valid["start_month"], 7)
        self.assertTrue(valid["is_workday"])
        self.assertEqual(valid["traffic_period"], "morning_peak")
        self.assertAlmostEqual(valid["distance_km"], 1.11195, places=3)

        bad_coordinate = cleaned.set_index("ride_id").loc["bad-coordinate"]
        self.assertFalse(bad_coordinate["has_valid_start_coords"])
        self.assertTrue(np.isnan(bad_coordinate["start_lat"]))
        self.assertTrue(np.isnan(bad_coordinate["distance_km"]))
        self.assertFalse(bad_coordinate["eligible_start_station_analysis"])
        self.assertTrue(bad_coordinate["is_federal_holiday"])
        self.assertFalse(bad_coordinate["is_workday"])
        self.assertEqual(bad_coordinate["traffic_period"], "evening_peak")

    def test_haversine_zero(self) -> None:
        result = haversine_km(
            pd.Series([-77.0]),
            pd.Series([38.9]),
            pd.Series([-77.0]),
            pd.Series([38.9]),
        )
        self.assertAlmostEqual(result.iloc[0], 0.0, places=12)

    def test_cross_chunk_duplicate_is_removed(self) -> None:
        raw = self._raw_frame().iloc[[0]].copy()
        seen: set[str] = set()
        first, first_report = preprocess_dataframe(raw, seen_ride_ids=seen)
        second, second_report = preprocess_dataframe(raw, seen_ride_ids=seen)

        self.assertEqual(len(first), 1)
        self.assertEqual(first_report["duplicate_ride_id_previous_chunk_rows"], 0)
        self.assertTrue(second.empty)
        self.assertEqual(second_report["duplicate_ride_id_previous_chunk_rows"], 1)

    def test_strict_file_discovery(self) -> None:
        files = discover_trip_files(DATA_DIR, "202501", "202606")
        self.assertEqual(len(files), 18)
        self.assertTrue(all("__MACOSX" not in str(path) for path in files))
        self.assertTrue(all(Path(path).stem == Path(path).parent.name for path in files))

    def test_temporal_aggregations_fill_zero_hours(self) -> None:
        cleaned = pd.DataFrame(
            {
                "start_time": [
                    "2025-07-03 08:00:00",
                    "2025-07-03 08:30:00",
                    "2025-07-05 09:00:00",
                ],
                "user_type": ["member", "member", "casual"],
                "trip_duration_s": [600.0, 1_200.0, 1_800.0],
                "distance_km": [1.0, 2.0, 3.0],
            }
        )
        result = analyze_chunks([cleaned], frequency="day")

        # July 4 lies inside the observed range and is retained as a zero-volume
        # day, so the hourly averages use all three calendar days.
        hourly = result.hourly_average_volume.set_index("hour")
        self.assertAlmostEqual(hourly.loc[8, "average_trips_per_day"], 2 / 3)
        self.assertAlmostEqual(hourly.loc[9, "average_trips_per_day"], 1 / 3)
        self.assertEqual(hourly.loc[10, "average_trips_per_day"], 0)

        distance = result.distance_over_time.set_index("period_start")
        self.assertEqual(
            distance.loc[pd.Timestamp("2025-07-04"), "total_distance_km"], 0
        )

        daily = result.daily_volume.set_index("date")
        self.assertEqual(daily.loc[pd.Timestamp("2025-07-04"), "trip_count"], 0)
        self.assertFalse(daily.loc[pd.Timestamp("2025-07-04"), "is_workday"])

        users = result.user_type_summary.set_index("user_type")
        self.assertEqual(users.loc["member", "trip_count"], 2)
        self.assertAlmostEqual(users.loc["member", "median_duration_min"], 15.0)
        self.assertAlmostEqual(
            users.loc["casual", "average_daily_trips"], 1 / 3
        )

if __name__ == "__main__":
    unittest.main()
