from __future__ import annotations

import re


_UNITS = {
    "m": 60,
    "h": 60 * 60,
    "d": 24 * 60 * 60,
    "w": 7 * 24 * 60 * 60,
}


def frequency_seconds(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([mhdw])", value.strip().lower())
    if not match:
        raise ValueError(f"Unsupported frequency '{value}'. Use values such as 5m, 1h, 4h, or 1d.")
    return int(match.group(1)) * _UNITS[match.group(2)]


def periods_per_year(frequency: str) -> float:
    return 365.0 * 24 * 60 * 60 / frequency_seconds(frequency)
