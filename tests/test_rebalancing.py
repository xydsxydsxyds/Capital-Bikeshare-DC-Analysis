import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.grid import grid_centers, lonlat_to_grid
from src.realtime_gbfs import discover_feed_urls, normalize_station_snapshot
from src.rebalancing import (
    SimulationConfig,
    build_historical_grid_flow_profiles,
    build_profile_demand,
    run_rebalancing_simulation,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


class RebalancingTests(unittest.TestCase):
    def _payload(self, name: str) -> dict:
        return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))

    def _top_grids(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "demand_rank": [1, 2],
                "grid_id": ["E649_N8615", "E647_N8617"],
                "grid_x": [649, 647],
                "grid_y": [8615, 8617],
                "center_lat": [38.90116295222856, 38.90996863593391],
                "center_lon": [-77.02096651012789, -77.03274858509862],
            }
        )

    def _snapshot(self) -> pd.DataFrame:
        return normalize_station_snapshot(
            self._payload("gbfs_station_information.json"),
            self._payload("gbfs_station_status.json"),
            fetched_at=pd.Timestamp("2026-08-08 00:01:00", tz="UTC"),
        )

    def test_discovery_snapshot_and_grid_mapping(self) -> None:
        discovery = {
            "data": {
                "en": {
                    "feeds": [
                        {"name": "station_information", "url": "https://example/info"},
                        {"name": "station_status", "url": "https://example/status"},
                    ]
                }
            }
        }
        feeds = discover_feed_urls(discovery)
        self.assertEqual(feeds["station_status"], "https://example/status")

        snapshot = self._snapshot()
        self.assertEqual(len(snapshot), 2)
        self.assertEqual(snapshot["vehicles_available"].sum(), 10)
        mapped = lonlat_to_grid(snapshot["lon"], snapshot["lat"])
        self.assertEqual(mapped["grid_id"].tolist(), ["E649_N8615", "E647_N8617"])

        centers = grid_centers(pd.Series([649]), pd.Series([8615]))
        self.assertEqual(centers.loc[0, "grid_id"], "E649_N8615")
        self.assertAlmostEqual(centers.loc[0, "center_lat"], 38.90116295, places=6)

    def test_historical_flow_profile_includes_zero_complete_hours(self) -> None:
        trips = pd.DataFrame(
            {
                "start_time": ["2025-01-01 08:00:00", "2025-01-01 08:30:00"],
                "end_time": ["2025-01-01 09:00:00", "2025-01-01 09:30:00"],
                "start_lat": [38.90116295222856, 38.90116295222856],
                "start_lon": [-77.02096651012789, -77.02096651012789],
                "end_lat": [38.90116295222856, 38.90116295222856],
                "end_lon": [-77.02096651012789, -77.02096651012789],
                "has_valid_start_coords": [True, True],
                "has_valid_end_coords": [True, True],
            }
        )
        top = self._top_grids().iloc[[0]].copy()
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "trips.parquet"
            trips.to_parquet(path, index=False)
            profiles = build_historical_grid_flow_profiles(
                [path],
                top,
                profile_start="2025-01-01",
                profile_end="2025-01-08",
                chunksize=1,
            )
        self.assertEqual(len(profiles), 7 * 24)
        departure = profiles.loc[
            profiles["weekday"].eq(2) & profiles["hour"].eq(8),
            "mean_departures",
        ].iloc[0]
        arrival = profiles.loc[
            profiles["weekday"].eq(2) & profiles["hour"].eq(9),
            "mean_arrivals",
        ].iloc[0]
        self.assertEqual(departure, 2.0)
        self.assertEqual(arrival, 2.0)
        self.assertEqual(
            profiles.loc[profiles["hour"].eq(10), "mean_departures"].sum(), 0
        )

    def test_greedy_strategy_reduces_known_shortage(self) -> None:
        profiles = pd.DataFrame(
            {
                "grid_id": ["E649_N8615", "E647_N8617"],
                "weekday": [3, 3],
                "hour": [8, 8],
                "mean_departures": [0.0, 6.0],
                "mean_arrivals": [0.0, 0.0],
            }
        )
        demand = build_profile_demand(
            profiles,
            self._top_grids()["grid_id"].tolist(),
            start_time="2026-01-01 08:00:00",
            horizon_hours=1,
        )
        config = SimulationConfig(
            horizon_hours=1,
            truck_count=1,
            truck_capacity=10,
            max_relocated_bikes_per_hour=10,
            low_inventory_ratio=0.2,
            high_inventory_ratio=0.8,
        )
        result = run_rebalancing_simulation(
            self._snapshot(),
            self._top_grids(),
            demand,
            config,
            demand_mode="profile",
        )
        metrics = result.metrics.set_index("strategy")
        self.assertLess(
            metrics.loc["greedy_rebalancing", "unmet_departures"],
            metrics.loc["no_rebalancing", "unmet_departures"],
        )
        self.assertEqual(result.dispatch_plan["bikes_moved"].sum(), 2)
        self.assertLessEqual(len(result.dispatch_plan), config.truck_count)
        self.assertTrue(result.strategy_trajectory["end_bikes"].ge(0).all())
        self.assertTrue(
            result.strategy_trajectory["end_bikes"]
            .le(result.strategy_trajectory["capacity"])
            .all()
        )


if __name__ == "__main__":
    unittest.main()
