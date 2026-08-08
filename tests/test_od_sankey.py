from __future__ import annotations

import unittest

import pandas as pd

from src.od_sankey import build_bipartite_od_sankey


class ODSankeyTests(unittest.TestCase):
    def test_two_column_layout_is_directional_and_geographically_ordered(self) -> None:
        stations = pd.DataFrame(
            {
                "station_id": ["N", "M", "S"],
                "station_name": ["North Station", "Middle Station", "South Station"],
                "station_lat": [38.95, 38.91, 38.87],
                "station_lon": [-77.03, -77.02, -77.01],
            }
        )
        od = pd.DataFrame(
            {
                "source_station_id": ["S", "N", "N"],
                "target_station_id": ["N", "S", "M"],
                "source_station_name": [
                    "South Station",
                    "North Station",
                    "North Station",
                ],
                "target_station_name": [
                    "North Station",
                    "South Station",
                    "Middle Station",
                ],
                "trip_count": [20, 10, 5],
            }
        )

        figure = build_bipartite_od_sankey(od, stations)
        sankey = figure.data[0]

        self.assertEqual(sankey.arrangement, "fixed")
        self.assertEqual(list(sankey.node.x), [0.02, 0.02, 0.98, 0.98, 0.98])
        self.assertIn("North Station", sankey.node.label[0])
        self.assertIn("South Station", sankey.node.label[1])
        self.assertIn("North Station", sankey.node.label[2])
        self.assertIn("Middle Station", sankey.node.label[3])
        self.assertIn("South Station", sankey.node.label[4])
        self.assertTrue(all(value < 2 for value in sankey.link.source))
        self.assertTrue(all(value >= 2 for value in sankey.link.target))
        self.assertEqual(sum(sankey.link.value), 35)

    def test_invalid_or_self_loop_flows_are_rejected(self) -> None:
        stations = pd.DataFrame(
            {
                "station_id": ["A"],
                "station_name": ["Station A"],
                "station_lat": [38.9],
                "station_lon": [-77.0],
            }
        )
        od = pd.DataFrame(
            {
                "source_station_id": ["A"],
                "target_station_id": ["A"],
                "source_station_name": ["Station A"],
                "target_station_name": ["Station A"],
                "trip_count": [5],
            }
        )

        with self.assertRaisesRegex(ValueError, "没有可绘制"):
            build_bipartite_od_sankey(od, stations)


if __name__ == "__main__":
    unittest.main()
