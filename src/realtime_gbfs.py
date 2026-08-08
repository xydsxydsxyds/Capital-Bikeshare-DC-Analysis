"""Fetch, validate, persist, and reload one Capital Bikeshare GBFS snapshot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISCOVERY_URL = "https://gbfs.capitalbikeshare.com/gbfs/2.3/gbfs.json"
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "realtime"


def _request_session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "capital-bikeshare-dc-course-project/1.0"}
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_json(
    url: str,
    *,
    timeout_s: float = 10.0,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    if timeout_s <= 0:
        raise ValueError("timeout_s 必须大于 0")
    active_session = session or _request_session()
    response = active_session.get(url, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(f"GBFS端点返回的JSON结构无效: {url}")
    return payload


def discover_feed_urls(
    discovery_payload: Mapping[str, Any], *, language: str = "en"
) -> dict[str, str]:
    """Parse both GBFS 2.x localized and GBFS 3.x discovery structures."""

    data = discovery_payload.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("GBFS discovery缺少data对象")
    if isinstance(data.get("feeds"), list):
        feed_rows = data["feeds"]
    else:
        localized = data.get(language)
        if not isinstance(localized, Mapping):
            localized = next(
                (
                    value
                    for value in data.values()
                    if isinstance(value, Mapping)
                    and isinstance(value.get("feeds"), list)
                ),
                None,
            )
        if not isinstance(localized, Mapping):
            raise ValueError("GBFS discovery中找不到feeds列表")
        feed_rows = localized["feeds"]

    feeds: dict[str, str] = {}
    for row in feed_rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name", "")).strip()
        url = str(row.get("url", "")).strip()
        if name and url:
            feeds[name] = url
    required = {"station_information", "station_status"}
    missing = sorted(required - set(feeds))
    if missing:
        raise ValueError("GBFS discovery缺少端点: " + ", ".join(missing))
    return feeds


def _localized_text(value: Any, language: str = "en") -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        candidates = [item for item in value if isinstance(item, Mapping)]
        preferred = next(
            (item for item in candidates if item.get("language") == language),
            candidates[0] if candidates else None,
        )
        if preferred is not None:
            text = str(preferred.get("text", "")).strip()
            return text or None
    return str(value).strip() or None


def _stations(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, Mapping) or not isinstance(data.get("stations"), list):
        raise ValueError("GBFS端点缺少data.stations列表")
    return [row for row in data["stations"] if isinstance(row, Mapping)]


def _parse_updated_at(value: Any) -> pd.Timestamp:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return pd.to_datetime(value, unit="s", utc=True)
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        raise ValueError(f"无法解析GBFS更新时间: {value!r}")
    return timestamp


def _coerce_service_flag(value: Any) -> bool:
    """Normalize GBFS 2.x integer flags and GBFS 3.x boolean values."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    return bool(value)


def normalize_station_snapshot(
    information_payload: Mapping[str, Any],
    status_payload: Mapping[str, Any],
    *,
    fetched_at: datetime | pd.Timestamp | None = None,
    language: str = "en",
) -> pd.DataFrame:
    """Join station metadata with current station supply status."""

    info_rows: list[dict[str, Any]] = []
    for row in _stations(information_payload):
        info_rows.append(
            {
                "station_id": str(row.get("station_id", "")).strip(),
                "station_name": _localized_text(row.get("name"), language),
                "lat": row.get("lat"),
                "lon": row.get("lon"),
                "published_capacity": row.get("capacity"),
            }
        )
    status_rows: list[dict[str, Any]] = []
    for row in _stations(status_payload):
        status_rows.append(
            {
                "station_id": str(row.get("station_id", "")).strip(),
                "vehicles_available": row.get("num_bikes_available", row.get("num_vehicles_available")),
                "docks_available": row.get("num_docks_available"),
                "vehicles_disabled": row.get("num_bikes_disabled", row.get("num_vehicles_disabled", 0)),
                "docks_disabled": row.get("num_docks_disabled", 0),
                "is_installed": row.get("is_installed", True),
                "is_renting": row.get("is_renting", True),
                "is_returning": row.get("is_returning", True),
                "last_reported": row.get("last_reported"),
            }
        )

    information = pd.DataFrame(info_rows)
    status = pd.DataFrame(status_rows)
    if information.empty or status.empty:
        raise ValueError("GBFS站点信息或状态为空")
    if information["station_id"].eq("").any() or status["station_id"].eq("").any():
        raise ValueError("GBFS站点记录包含空station_id")
    if information["station_id"].duplicated().any():
        raise ValueError("station_information包含重复station_id")
    if status["station_id"].duplicated().any():
        raise ValueError("station_status包含重复station_id")

    snapshot = status.merge(
        information,
        on="station_id",
        how="left",
        validate="one_to_one",
    )
    for column in (
        "lat",
        "lon",
        "published_capacity",
        "vehicles_available",
        "docks_available",
        "vehicles_disabled",
        "docks_disabled",
    ):
        snapshot[column] = pd.to_numeric(snapshot[column], errors="coerce")
    for column in ("is_installed", "is_renting", "is_returning"):
        snapshot[column] = snapshot[column].map(_coerce_service_flag).astype(bool)

    numeric_status = [
        "vehicles_available",
        "docks_available",
        "vehicles_disabled",
        "docks_disabled",
    ]
    if snapshot[numeric_status].isna().any().any():
        missing = snapshot.loc[
            snapshot[numeric_status].isna().any(axis=1), "station_id"
        ].tolist()
        raise ValueError(f"GBFS状态字段缺失，示例站点: {missing[:5]}")
    if snapshot[numeric_status].lt(0).any().any():
        raise ValueError("GBFS状态包含负车辆数或负车桩数")

    snapshot_time = _parse_updated_at(status_payload.get("last_updated"))
    fetched_timestamp = pd.Timestamp(
        fetched_at if fetched_at is not None else datetime.now(timezone.utc)
    )
    if fetched_timestamp.tzinfo is None:
        fetched_timestamp = fetched_timestamp.tz_localize("UTC")
    else:
        fetched_timestamp = fetched_timestamp.tz_convert("UTC")
    snapshot["snapshot_time"] = snapshot_time
    snapshot["fetched_at"] = fetched_timestamp
    snapshot["data_age_seconds"] = max(
        0.0, float((fetched_timestamp - snapshot_time).total_seconds())
    )
    numeric_last_reported = pd.to_numeric(
        snapshot["last_reported"], errors="coerce"
    )
    epoch_last_reported = pd.to_datetime(
        numeric_last_reported, unit="s", errors="coerce", utc=True
    )
    text_last_reported = pd.to_datetime(
        snapshot["last_reported"].where(numeric_last_reported.isna()),
        errors="coerce",
        utc=True,
    )
    snapshot["last_reported"] = epoch_last_reported.fillna(text_last_reported)
    snapshot["functional_capacity"] = (
        snapshot["vehicles_available"] + snapshot["docks_available"]
    )
    capacity_fallback = snapshot["functional_capacity"]
    snapshot["capacity"] = snapshot["published_capacity"].where(
        snapshot["published_capacity"].gt(0), capacity_fallback
    )
    snapshot["capacity"] = np.maximum(
        snapshot["capacity"], snapshot["functional_capacity"]
    )
    snapshot["station_name"] = snapshot["station_name"].astype("string")
    snapshot["station_id"] = snapshot["station_id"].astype("string")
    return snapshot[
        [
            "snapshot_time",
            "fetched_at",
            "data_age_seconds",
            "station_id",
            "station_name",
            "lat",
            "lon",
            "capacity",
            "functional_capacity",
            "vehicles_available",
            "docks_available",
            "vehicles_disabled",
            "docks_disabled",
            "is_installed",
            "is_renting",
            "is_returning",
            "last_reported",
        ]
    ].sort_values("station_id").reset_index(drop=True)


def fetch_capital_bikeshare_snapshot(
    *,
    discovery_url: str = DEFAULT_DISCOVERY_URL,
    language: str = "en",
    timeout_s: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    session = _request_session()
    fetched_at = datetime.now(timezone.utc)
    discovery = fetch_json(discovery_url, timeout_s=timeout_s, session=session)
    feeds = discover_feed_urls(discovery, language=language)
    information = fetch_json(
        feeds["station_information"], timeout_s=timeout_s, session=session
    )
    status = fetch_json(feeds["station_status"], timeout_s=timeout_s, session=session)
    snapshot = normalize_station_snapshot(
        information, status, fetched_at=fetched_at, language=language
    )
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "discovery_url": discovery_url,
        "station_information_url": feeds["station_information"],
        "station_status_url": feeds["station_status"],
        "gbfs_version": status.get("version", discovery.get("version")),
        "snapshot_time": snapshot["snapshot_time"].iloc[0].isoformat(),
        "fetched_at": snapshot["fetched_at"].iloc[0].isoformat(),
        "station_rows": int(len(snapshot)),
        "data_age_seconds": float(snapshot["data_age_seconds"].iloc[0]),
    }
    return snapshot, metadata


def save_snapshot(
    snapshot: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    output_dir: Path = DEFAULT_SNAPSHOT_DIR,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_time = pd.Timestamp(snapshot["snapshot_time"].iloc[0])
    if snapshot_time.tzinfo is None:
        snapshot_time = snapshot_time.tz_localize("UTC")
    else:
        snapshot_time = snapshot_time.tz_convert("UTC")
    stamp = snapshot_time.strftime("%Y%m%d_%H%M%S")
    parquet_path = output_dir / f"station_snapshot_{stamp}.parquet"
    metadata_path = output_dir / f"station_snapshot_{stamp}.json"
    existing = [path for path in (parquet_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "快照输出已存在（使用 --overwrite 替换）: "
            + ", ".join(path.name for path in existing)
        )
    snapshot.to_parquet(parquet_path, index=False)
    metadata_path.write_text(
        json.dumps(dict(metadata), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return parquet_path, metadata_path


def find_latest_snapshot(
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> Path | None:
    if not snapshot_dir.exists():
        return None
    candidates = sorted(snapshot_dir.glob("station_snapshot_*.parquet"))
    return candidates[-1] if candidates else None


def load_latest_snapshot(
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> tuple[pd.DataFrame, Path]:
    path = find_latest_snapshot(snapshot_dir)
    if path is None:
        raise FileNotFoundError(f"没有本地GBFS快照: {snapshot_dir}")
    return pd.read_parquet(path), path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="获取并保存一次Capital Bikeshare GBFS实时站点快照"
    )
    parser.add_argument("--discovery-url", default=DEFAULT_DISCOVERY_URL)
    parser.add_argument("--language", default="en")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot, metadata = fetch_capital_bikeshare_snapshot(
        discovery_url=args.discovery_url,
        language=args.language,
        timeout_s=args.timeout,
    )
    paths = save_snapshot(
        snapshot, metadata, output_dir=args.output_dir, overwrite=args.overwrite
    )
    print("\nGBFS snapshot complete:")
    print(f"  stations: {len(snapshot):,}")
    print(f"  snapshot time: {snapshot['snapshot_time'].iloc[0]}")
    print(f"  data age: {snapshot['data_age_seconds'].iloc[0]:.0f} seconds")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
