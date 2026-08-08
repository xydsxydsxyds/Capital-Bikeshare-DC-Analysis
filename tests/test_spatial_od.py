from __future__ import annotations

import unittest

import pandas as pd

from src.spatial_od import analyze_spatial_chunks


class SpatialODTests(unittest.TestCase):
    def test_station_inference_od_and_grid_counts(self) -> None:
        rows = [
            ("A", "Station A", "B", "Station B", 38.9000, -77.0000, 38.9100, -77.0100, "morning_peak"),
            ("A", "Station A", "B", "Station B", 38.9000, -77.0000, 38.9100, -77.0100, "morning_peak"),
            ("B", "Station B", "A", "Station A", 38.9100, -77.0100, 38.9000, -77.0000, "evening_peak"),
            (pd.NA, pd.NA, "B", "Station B", 38.90005, -77.0000, 38.9100, -77.0100, "evening_peak"),
            ("A", "Station A", "A", "Station A", 38.9000, -77.0000, 38.9000, -77.0000, "off_peak"),
        ]
        frame = pd.DataFrame(
            rows,
            columns=[
                "start_station_id",
                "start_station_name",
                "end_station_id",
                "end_station_name",
                "start_lat",
                "start_lon",
                "end_lat",
                "end_lon",
                "traffic_period",
            ],
        )
        frame["has_valid_start_coords"] = True
        frame["has_valid_end_coords"] = True

        def chunks():
            return [frame.iloc[:3].copy(), frame.iloc[3:].copy()]

        result = analyze_spatial_chunks(chunks, grid_size_m=500)
        stations = result.station_stats.set_index("station_id")

        self.assertEqual(result.input_rows, 5)
        self.assertEqual(stations.loc["A", "observed_departures"], 3)
        self.assertEqual(stations.loc["A", "inferred_departures_100m"], 1)
        self.assertEqual(stations.loc["A", "observed_plus_inferred_100m"], 4)
        self.assertEqual(result.inference_summary.loc[0, "high_confidence_100m_rows"], 1)

        od = result.top20_od_pairs
        self.assertEqual(int(od["trip_count"].sum()), 3)
        self.assertFalse(
            od["source_station_id"].eq(od["target_station_id"]).any()
        )
        self.assertEqual(int(result.grid_stats["total_activity"].sum()), 10)


if __name__ == "__main__":
    unittest.main()

