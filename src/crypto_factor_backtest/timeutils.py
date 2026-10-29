from __future__ import annotations

import re

import pandas as pd

_UNITS = {
    "m": 60,
    "h": 60 * 60,
    "d": 24 * 60 * 60,
    "w": 7 * 24 * 60 * 60,
}


def frequency_seconds(value: str) -> int:
    if not isinstance(value, str):
        raise ValueError("Frequency must be a string such as 1h or 4h")
    # Uppercase M means calendar months on exchanges, never minutes.
    match = re.fullmatch(r"([1-9][0-9]*)([mhdw])", value.strip())
    if not match:
        raise ValueError(f"Unsupported frequency '{value}'. Use values such as 5m, 1h, 4h, or 1d.")
    return int(match.group(1)) * _UNITS[match.group(2)]


def periods_per_year(frequency: str) -> float:
    return 365.0 * 24 * 60 * 60 / frequency_seconds(frequency)


def utc_timestamp(value: str) -> pd.Timestamp:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Date boundaries must be non-empty timestamp strings")
    try:
        timestamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid timestamp: {value!r}") from exc
    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp: {value!r}")
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
