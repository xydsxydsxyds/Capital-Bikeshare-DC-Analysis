from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.forecast import (
    _month_time_bounds,
    build_hourly_panel,
    engineer_features,
    fit_historical_mean,
    predict_historical_mean,
    regression_metrics,
    select_top_grids,
)


class ForecastTests(unittest.TestCase):
    def test_month_time_bounds_include_complete_end_month(self) -> None:
        start, end = _month_time_bounds("202601", "202607")
        self.assertEqual(start, pd.Timestamp("2026-01-01"))
        self.assertEqual(end, pd.Timestamp("2026-08-01"))

    def _counts(self) -> pd.Series:
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2025-01-01 00:00"), 1, 2),
                (pd.Timestamp("2025-01-01 01:00"), 1, 2),
                (pd.Timestamp("2025-01-01 00:00"), 3, 4),
            ],
            names=["timestamp", "grid_x", "grid_y"],
        )
        return pd.Series([10, 5, 8], index=index, dtype="int64")

    def test_top_grid_selection_and_zero_filled_panel(self) -> None:
        counts = self._counts()
        top = select_top_grids(counts, top_n=2, grid_size_m=500)
        self.assertEqual(top.iloc[0]["grid_id"], "E1_N2")
        self.assertEqual(top.iloc[0]["training_departures"], 15)

        panel = build_hourly_panel(
            counts,
            top,
            start_time="2025-01-01 00:00",
            end_time="2025-01-01 03:00",
        )
        self.assertEqual(len(panel), 6)
        second_grid = panel.loc[panel["grid_id"].eq("E3_N4")]
        self.assertEqual(second_grid["demand"].tolist(), [8, 0, 0])

    def test_current_target_does_not_enter_its_features(self) -> None:
        hours = pd.date_range("2025-01-01", periods=200, freq="h")
        panel = pd.DataFrame(
            {
                "grid_id": "E1_N2",
                "timestamp": hours,
                "demand": np.arange(200, dtype="int32"),
                "grid_x": 1,
                "grid_y": 2,
            }
        )
        changed = panel.copy()
        changed.loc[180, "demand"] = 100_000
        first, columns = engineer_features(panel, ["E1_N2"])
        second, _ = engineer_features(changed, ["E1_N2"])

        self.assertEqual(first.loc[180, "lag_1"], 179)
        self.assertEqual(first.loc[180, "lag_168"], 12)
        self.assertAlmostEqual(first.loc[180, "rolling_mean_3"], 178.0)
        pd.testing.assert_series_equal(
            first.loc[180, columns], second.loc[180, columns], check_names=False
        )

    def test_metrics_and_historical_mean_fallback(self) -> None:
        metrics = regression_metrics([0, 2, 4], [0, 1, 5])
        self.assertAlmostEqual(metrics["mae"], 2 / 3)
        self.assertAlmostEqual(metrics["mape_pct"], 37.5)
        self.assertEqual(metrics["positive_actual_n"], 2)

        train = pd.DataFrame(
            {
                "grid_id": ["A", "A", "B"],
                "weekday": [0, 0, 1],
                "hour": [8, 8, 9],
                "demand": [2, 4, 8],
            }
        )
        fitted = fit_historical_mean(train)
        rows = pd.DataFrame(
            {
                "grid_id": ["A", "A", "unknown"],
                "weekday": [0, 6, 6],
                "hour": [8, 20, 20],
            }
        )
        prediction = predict_historical_mean(rows, fitted)
        self.assertAlmostEqual(prediction[0], 3.0)
        self.assertAlmostEqual(prediction[1], 3.0)
        self.assertAlmostEqual(prediction[2], 14 / 3)


if __name__ == "__main__":
    unittest.main()
