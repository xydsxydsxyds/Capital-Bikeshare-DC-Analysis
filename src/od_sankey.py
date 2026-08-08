"""Reusable, geographically ordered OD Sankey construction."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


OD_COLUMNS = (
    "source_station_id",
    "target_station_id",
    "source_station_name",
    "target_station_name",
    "trip_count",
)
STATION_COLUMNS = (
    "station_id",
    "station_name",
    "station_lat",
    "station_lon",
)


def _short_label(value: object, limit: int = 30) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.strip()
    if value.startswith("rgb"):
        channels = [int(channel) for channel in re.findall(r"\d+", value)[:3]]
        if len(channels) != 3:
            raise ValueError(f"无法解析颜色: {hex_color}")
        red, green, blue = channels
    else:
        value = value.lstrip("#")
        if len(value) != 6:
            raise ValueError(f"无法解析颜色: {hex_color}")
        red, green, blue = (
            int(value[index : index + 2], 16) for index in (0, 2, 4)
        )
    return f"rgba({red},{green},{blue},{alpha})"


def _vertical_positions(count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [0.5]
    return np.linspace(0.03, 0.97, count).tolist()


def _north_to_south(
    station_ids: Sequence[str], station_lookup: pd.DataFrame
) -> list[str]:
    def sort_key(station_id: str) -> tuple[float, str, str]:
        row = station_lookup.loc[station_id]
        latitude = pd.to_numeric(row["station_lat"], errors="coerce")
        north_first = -float(latitude) if pd.notna(latitude) else float("inf")
        return north_first, str(row["station_name"]), station_id

    return sorted(set(station_ids), key=sort_key)


def build_bipartite_od_sankey(
    od: pd.DataFrame,
    station_reference: pd.DataFrame,
    *,
    title: str = "Top directed OD flows",
) -> go.Figure:
    """Build a fixed two-column Sankey with stations ordered north to south.

    A station is intentionally represented twice when it is both an origin and
    a destination. This removes cyclic links from the layout and makes travel
    direction unambiguous.
    """

    missing_od = sorted(set(OD_COLUMNS) - set(od.columns))
    if missing_od:
        raise ValueError("OD数据缺少字段: " + ", ".join(missing_od))
    missing_station = sorted(set(STATION_COLUMNS) - set(station_reference.columns))
    if missing_station:
        raise ValueError("站点参考表缺少字段: " + ", ".join(missing_station))

    flows = od.loc[:, list(OD_COLUMNS)].copy()
    flows["source_station_id"] = flows["source_station_id"].astype(str)
    flows["target_station_id"] = flows["target_station_id"].astype(str)
    flows["trip_count"] = pd.to_numeric(flows["trip_count"], errors="coerce")
    flows = flows.loc[
        flows["trip_count"].gt(0)
        & flows["source_station_id"].ne(flows["target_station_id"])
    ].copy()
    if flows.empty:
        raise ValueError("没有可绘制的正向非自环OD流量")
    flows = (
        flows.groupby(
            [
                "source_station_id",
                "target_station_id",
                "source_station_name",
                "target_station_name",
            ],
            as_index=False,
            dropna=False,
        )["trip_count"]
        .sum()
        .sort_values("trip_count", ascending=False)
        .reset_index(drop=True)
    )

    stations = station_reference.loc[:, list(STATION_COLUMNS)].copy()
    stations["station_id"] = stations["station_id"].astype(str)
    stations["station_lat"] = pd.to_numeric(
        stations["station_lat"], errors="coerce"
    )
    stations["station_lon"] = pd.to_numeric(
        stations["station_lon"], errors="coerce"
    )
    stations = stations.drop_duplicates("station_id").set_index("station_id")
    participating_ids = set(flows["source_station_id"]) | set(
        flows["target_station_id"]
    )
    missing_ids = sorted(participating_ids - set(stations.index))
    if missing_ids:
        raise ValueError("站点参考表缺少OD节点: " + ", ".join(missing_ids[:5]))

    source_ids = _north_to_south(flows["source_station_id"].tolist(), stations)
    target_ids = _north_to_south(flows["target_station_id"].tolist(), stations)
    source_index = {station_id: index for index, station_id in enumerate(source_ids)}
    target_index = {
        station_id: len(source_ids) + index
        for index, station_id in enumerate(target_ids)
    }

    source_totals = flows.groupby("source_station_id")["trip_count"].sum()
    target_totals = flows.groupby("target_station_id")["trip_count"].sum()
    all_ids = _north_to_south(list(participating_ids), stations)
    palette = px.colors.qualitative.Safe
    station_colors = {
        station_id: palette[index % len(palette)]
        for index, station_id in enumerate(all_ids)
    }

    node_ids = source_ids + target_ids
    node_roles = ["Origin"] * len(source_ids) + ["Destination"] * len(target_ids)
    node_totals = [float(source_totals[value]) for value in source_ids] + [
        float(target_totals[value]) for value in target_ids
    ]
    node_labels = [
        f"{_short_label(stations.loc[station_id, 'station_name'])} · {total:,.0f}"
        for station_id, total in zip(node_ids, node_totals)
    ]
    node_customdata = [
        [
            role,
            str(stations.loc[station_id, "station_name"]),
            station_id,
            float(stations.loc[station_id, "station_lat"]),
            float(stations.loc[station_id, "station_lon"]),
        ]
        for station_id, role in zip(node_ids, node_roles)
    ]
    node_colors = [
        _rgba(station_colors[station_id], 0.92 if role == "Origin" else 0.66)
        for station_id, role in zip(node_ids, node_roles)
    ]

    link_source = [source_index[value] for value in flows["source_station_id"]]
    link_target = [target_index[value] for value in flows["target_station_id"]]
    link_colors = [
        _rgba(station_colors[value], 0.30) for value in flows["source_station_id"]
    ]
    link_customdata = [
        [source_name, target_name, source_id, target_id]
        for source_name, target_name, source_id, target_id in zip(
            flows["source_station_name"],
            flows["target_station_name"],
            flows["source_station_id"],
            flows["target_station_id"],
        )
    ]

    figure = go.Figure(
        go.Sankey(
            arrangement="fixed",
            orientation="h",
            valueformat=",.0f",
            valuesuffix=" trips",
            node=dict(
                pad=22,
                thickness=17,
                line=dict(color="rgba(70,80,95,0.45)", width=0.7),
                label=node_labels,
                color=node_colors,
                customdata=node_customdata,
                x=[0.02] * len(source_ids) + [0.98] * len(target_ids),
                y=_vertical_positions(len(source_ids))
                + _vertical_positions(len(target_ids)),
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "%{customdata[0]} station · ID %{customdata[2]}<br>"
                    "Coordinates: %{customdata[3]:.4f}, %{customdata[4]:.4f}<br>"
                    "Displayed flow: %{value:,.0f} trips<extra></extra>"
                ),
            ),
            link=dict(
                source=link_source,
                target=link_target,
                value=flows["trip_count"].tolist(),
                color=link_colors,
                customdata=link_customdata,
                hovertemplate=(
                    "<b>%{customdata[0]} → %{customdata[1]}</b><br>"
                    "Station IDs: %{customdata[2]} → %{customdata[3]}<br>"
                    "%{value:,.0f} trips<extra></extra>"
                ),
            ),
        )
    )
    figure.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        font=dict(size=11),
        height=max(640, 74 * max(len(source_ids), len(target_ids))),
        margin=dict(l=25, r=25, t=105, b=30),
        annotations=[
            dict(
                x=0.0,
                y=1.055,
                xref="paper",
                yref="paper",
                text="<b>ORIGINS</b><br><span style='font-size:10px'>north → south</span>",
                showarrow=False,
                xanchor="left",
                align="left",
            ),
            dict(
                x=1.0,
                y=1.055,
                xref="paper",
                yref="paper",
                text="<b>DESTINATIONS</b><br><span style='font-size:10px'>north → south</span>",
                showarrow=False,
                xanchor="right",
                align="right",
            ),
        ],
    )
    return figure
