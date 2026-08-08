"""Shared exact-grid utilities for spatial, forecasting, and live analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pyproj import Transformer


WGS84_EPSG = 4326
UTM18N_EPSG = 32618
DEFAULT_GRID_SIZE_M = 500


def validate_grid_size(grid_size_m: int) -> None:
    if grid_size_m <= 0:
        raise ValueError("grid_size_m 必须大于 0")


def grid_ids(
    grid_x: Sequence[int] | np.ndarray | pd.Series,
    grid_y: Sequence[int] | np.ndarray | pd.Series,
) -> pd.Series:
    """Return stable ``E{x}_N{y}`` identifiers for projected grid indices."""

    x = pd.Series(grid_x, copy=False).astype("int32")
    y = pd.Series(grid_y, copy=False).astype("int32")
    return "E" + x.astype(str) + "_N" + y.astype(str)


def lonlat_to_grid(
    longitude: Sequence[float] | np.ndarray | pd.Series,
    latitude: Sequence[float] | np.ndarray | pd.Series,
    *,
    grid_size_m: int = DEFAULT_GRID_SIZE_M,
) -> pd.DataFrame:
    """Project WGS84 coordinates to the exact EPSG:32618 grid used by the project."""

    validate_grid_size(grid_size_m)
    lon = pd.to_numeric(pd.Series(longitude, copy=False), errors="coerce")
    lat = pd.to_numeric(pd.Series(latitude, copy=False), errors="coerce")
    valid = lon.notna() & lat.notna()
    result = pd.DataFrame(
        {
            "grid_x": pd.Series(pd.NA, index=lon.index, dtype="Int32"),
            "grid_y": pd.Series(pd.NA, index=lon.index, dtype="Int32"),
            "grid_id": pd.Series(pd.NA, index=lon.index, dtype="string"),
        }
    )
    if not valid.any():
        return result

    transformer = Transformer.from_crs(
        WGS84_EPSG, UTM18N_EPSG, always_xy=True
    )
    eastings, northings = transformer.transform(
        lon.loc[valid].to_numpy(dtype="float64").tolist(),
        lat.loc[valid].to_numpy(dtype="float64").tolist(),
    )
    grid_x = np.floor(np.asarray(eastings) / grid_size_m).astype("int32")
    grid_y = np.floor(np.asarray(northings) / grid_size_m).astype("int32")
    result.loc[valid, "grid_x"] = grid_x
    result.loc[valid, "grid_y"] = grid_y
    result.loc[valid, "grid_id"] = grid_ids(grid_x, grid_y).to_numpy()
    return result


def grid_centers(
    grid_x: Sequence[int] | np.ndarray | pd.Series,
    grid_y: Sequence[int] | np.ndarray | pd.Series,
    *,
    grid_size_m: int = DEFAULT_GRID_SIZE_M,
) -> pd.DataFrame:
    """Return WGS84 center coordinates for projected grid indices."""

    validate_grid_size(grid_size_m)
    x = pd.to_numeric(pd.Series(grid_x, copy=False), errors="raise").astype("int32")
    y = pd.to_numeric(pd.Series(grid_y, copy=False), errors="raise").astype("int32")
    center_easting = x.to_numpy(dtype="float64") * grid_size_m + grid_size_m / 2
    center_northing = y.to_numpy(dtype="float64") * grid_size_m + grid_size_m / 2
    inverse = Transformer.from_crs(
        UTM18N_EPSG, WGS84_EPSG, always_xy=True
    )
    center_lon, center_lat = inverse.transform(
        center_easting.tolist(), center_northing.tolist()
    )
    return pd.DataFrame(
        {
            "grid_id": grid_ids(x, y).to_numpy(),
            "grid_x": x.to_numpy(),
            "grid_y": y.to_numpy(),
            "center_lat": center_lat,
            "center_lon": center_lon,
        }
    )
