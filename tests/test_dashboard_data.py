from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.dashboard_data import aggregate_dashboard_chunks


class DashboardDataTests(unittest.TestCase):
    def test_aggregates_preserve_temporal_spatial_and_od_counts(self) -> None:
        frame = pd.DataFrame(
            {
                "start_time": [
                    "2025-01-02 08:00:00",
                    "2025-01-02 08:30:00",
                    "2025-01-03 17:00:00",
                    "2025-01-03 18:00:00",
                ],
                "user_type": ["member", "member", "casual", "member"],
                "trip_duration_s": [600, 1_200, 900, 600],
                "distance_km": [1.0, 2.0, np.nan, 1.5],
                "start_station_id": ["A", "A", "B", "A"],
                "end_station_id": ["B", "B", "A", "A"],
                "start_lat": [38.90, 38.90, 38.91, 38.90],
                "start_lon": [-77.00, -77.00, -77.01, -77.00],
                "end_lat": [38.91, 38.91, 38.90, 38.90],
                "end_lon": [-77.01, -77.01, -77.00, -77.00],
                "has_valid_start_coords": [True, True, True, True],
                "has_valid_end_coords": [True, True, True, True],
            }
        )
        station_reference = pd.DataFrame(
            {
                "station_id": ["A", "B"],
                "station_name": ["Station A", "Station B"],
                "station_lat": [38.90, 38.91],
                "station_lon": [-77.00, -77.01],
            }
        )
        result = aggregate_dashboard_chunks(
            [frame.iloc[:2], frame.iloc[2:]],
            station_reference,
            station_reference,
            grid_size_m=500,
        )

        daily = result.daily_user_summary.set_index(["date", "user_type"])
        key = (pd.Timestamp("2025-01-02"), "member")
        self.assertEqual(daily.loc[key, "trip_count"], 2)
        self.assertAlmostEqual(daily.loc[key, "total_distance_km"], 3.0)
        self.assertEqual(daily.loc[key, "distance_valid_trips"], 2)

        hourly = result.hourly_user_volume
        self.assertEqual(
            int(hourly.loc[hourly["hour"].eq(8), "trip_count"].sum()), 2
        )
        station = result.station_daily_user
        self.assertEqual(int(station["departures"].sum()), 4)
        self.assertEqual(int(station["arrivals"].sum()), 4)
        self.assertEqual(int(result.grid_daily_user["departures"].sum()), 4)
        self.assertEqual(int(result.grid_daily_user["arrivals"].sum()), 4)

        # Three cross-station trips enter OD; the A->A self-loop is excluded.
        self.assertEqual(int(result.od_daily_user["trip_count"].sum()), 3)
        self.assertFalse(
            result.od_daily_user["source_station_id"].eq(
                result.od_daily_user["target_station_id"]
            ).any()
        )


if __name__ == "__main__":
    unittest.main()
