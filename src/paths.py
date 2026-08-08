"""Project paths and strict discovery of monthly trip-data files."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

MONTH_PATTERN = re.compile(r"^(?P<month>\d{6})-capitalbikeshare-tripdata$")


def _validate_month(month: str | None) -> str | None:
    if month is None:
        return None
    if not re.fullmatch(r"\d{6}", month):
        raise ValueError(f"月份必须使用 YYYYMM 格式，收到: {month!r}")
    year = int(month[:4])
    number = int(month[4:])
    if year < 1900 or not 1 <= number <= 12:
        raise ValueError(f"无效月份: {month!r}")
    return month


def discover_trip_files(
    data_dir: Path = DATA_DIR,
    start_month: str | None = None,
    end_month: str | None = None,
) -> list[Path]:
    """Return only valid ``data/YYYYMM-.../YYYYMM-....csv`` inputs.

    This intentionally excludes ``__MACOSX``, temporary files, processed data,
    and any CSV whose filename does not agree with its parent month directory.
    Both month bounds are inclusive.
    """

    start_month = _validate_month(start_month)
    end_month = _validate_month(end_month)
    if start_month and end_month and start_month > end_month:
        raise ValueError("start_month 不能晚于 end_month")

    files: list[Path] = []
    if not data_dir.exists():
        return files

    for directory in data_dir.iterdir():
        if not directory.is_dir():
            continue
        match = MONTH_PATTERN.fullmatch(directory.name)
        if not match:
            continue
        month = match.group("month")
        if start_month and month < start_month:
            continue
        if end_month and month > end_month:
            continue

        candidate = directory / f"{directory.name}.csv"
        if candidate.is_file():
            files.append(candidate)

    return sorted(files, key=lambda path: path.parent.name[:6])


def month_from_trip_file(path: Path) -> str:
    """Extract and validate the YYYYMM key from a discovered trip file."""

    match = MONTH_PATTERN.fullmatch(path.parent.name)
    expected_name = f"{path.parent.name}.csv"
    if not match or path.name != expected_name:
        raise ValueError(f"不是有效的月度 Capital Bikeshare 文件: {path}")
    return match.group("month")

