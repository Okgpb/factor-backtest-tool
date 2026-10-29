from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .timeutils import utc_timestamp

BAR_COLUMNS = {"timestamp", "symbol", "open", "high", "low", "close", "volume", "quote_volume"}
FACTOR_KEYS = {"timestamp", "symbol"}
FUNDING_COLUMNS = {"funding_time", "symbol", "funding_rate"}


@dataclass(slots=True)
class FrequencyData:
    bars: pd.DataFrame
    factors: pd.DataFrame
    funding: pd.DataFrame = field(default_factory=pd.DataFrame)
    universe: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


def read_table(path: Path) -> pd.DataFrame:
    if path.is_dir():
        return pd.read_parquet(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv" or path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data format: {path}")


def _find_table(root: Path, names: list[str], *, required: bool) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(f"None of the expected inputs exist under {root}: {', '.join(names)}")
    return None


def _utc(series: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().any():
        examples = series[parsed.isna()].head(3).tolist()
        raise ValueError(f"{label} contains invalid timestamps, examples: {examples}")
    return parsed


def _symbols(series: pd.Series, label: str) -> pd.Series:
    if series.isna().any() or not series.map(lambda value: isinstance(value, str)).all():
        raise ValueError(f"{label} must contain non-empty symbol strings")
    symbols = series.str.strip().str.upper()
    if symbols.eq("").any():
        raise ValueError(f"{label} must contain non-empty symbol strings")
    return symbols


def _normalise_bars(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "timestamp" not in frame and "close_time" in frame:
        frame = frame.rename(columns={"close_time": "timestamp"})
    missing = BAR_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"bars data is missing columns: {sorted(missing)}")
    frame["timestamp"] = _utc(frame["timestamp"], "bars.timestamp")
    frame["symbol"] = _symbols(frame["symbol"], "bars.symbol")
    numeric = ["open", "high", "low", "close", "volume", "quote_volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("bars data contains non-finite, non-numeric or missing OHLCV values")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("bars data contains non-positive prices")
    if (frame[["volume", "quote_volume"]] < 0).any().any():
        raise ValueError("bars data contains negative volume")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any() or (
        frame["low"] > frame[["open", "high", "close"]].min(axis=1)
    ).any():
        raise ValueError("bars data contains inconsistent OHLC prices")
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("bars data contains duplicate (timestamp, symbol) rows")
    return frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _normalise_factors(frame: pd.DataFrame, factor_column: str) -> pd.DataFrame:
    missing = (FACTOR_KEYS | {factor_column}) - set(frame.columns)
    if missing:
        raise ValueError(f"factor data is missing columns: {sorted(missing)}")
    frame = frame[["timestamp", "symbol", factor_column]].copy()
    frame["timestamp"] = _utc(frame["timestamp"], "factors.timestamp")
    frame["symbol"] = _symbols(frame["symbol"], "factors.symbol")
    frame[factor_column] = pd.to_numeric(frame[factor_column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[factor_column])
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("factor data contains duplicate (timestamp, symbol) rows")
    return frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _normalise_funding(frame: pd.DataFrame) -> pd.DataFrame:
    missing = FUNDING_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"funding data is missing columns: {sorted(missing)}")
    frame = frame[["funding_time", "symbol", "funding_rate"]].copy()
    frame["funding_time"] = _utc(frame["funding_time"], "funding.funding_time")
    frame["symbol"] = _symbols(frame["symbol"], "funding.symbol")
    frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="coerce")
    if not np.isfinite(frame["funding_rate"].to_numpy(dtype=float)).all():
        raise ValueError("funding data contains non-finite or invalid rates")
    if frame.duplicated(["funding_time", "symbol"]).any():
        raise ValueError("funding data contains duplicate (funding_time, symbol) rows")
    return frame.sort_values(["funding_time", "symbol"]).reset_index(drop=True)


def _normalise_universe(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "symbol", "tradable"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"universe data is missing columns: {sorted(missing)}")
    frame = frame[["timestamp", "symbol", "tradable"]].copy()
    frame["timestamp"] = _utc(frame["timestamp"], "universe.timestamp")
    frame["symbol"] = _symbols(frame["symbol"], "universe.symbol")
    if frame["tradable"].isna().any():
        raise ValueError("universe.tradable contains missing boolean values")
    if pd.api.types.is_bool_dtype(frame["tradable"]):
        frame["tradable"] = frame["tradable"].astype(bool)
    elif pd.api.types.is_numeric_dtype(frame["tradable"]):
        if not frame["tradable"].isin([0, 1]).all():
            raise ValueError("universe.tradable numeric values must be 0 or 1")
        frame["tradable"] = frame["tradable"].ne(0)
    else:
        values = frame["tradable"].astype(str).str.strip().str.lower()
        mapping = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
        unknown = sorted(set(values) - set(mapping))
        if unknown:
            raise ValueError(f"universe.tradable contains invalid boolean values: {unknown[:3]}")
        frame["tradable"] = values.map(mapping)
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("universe data contains duplicate (timestamp, symbol) rows")
    return frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def load_frequency_data(
    data_root: Path,
    factor_root: Path,
    frequency: str,
    factor_column: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> FrequencyData:
    market_dir = data_root / frequency
    if not market_dir.exists():
        raise FileNotFoundError(f"Market-data frequency directory does not exist: {market_dir}")
    bars_path = _find_table(market_dir, ["bars.parquet", "bars.csv", "bars"], required=True)
    factor_path = _find_table(
        factor_root,
        [f"{frequency}.parquet", f"{frequency}.csv", frequency],
        required=True,
    )
    funding_path = _find_table(market_dir, ["funding.parquet", "funding.csv"], required=False)
    universe_path = _find_table(market_dir, ["universe.parquet", "universe.csv"], required=False)

    bars = _normalise_bars(read_table(bars_path))  # type: ignore[arg-type]
    factors = _normalise_factors(read_table(factor_path), factor_column)  # type: ignore[arg-type]
    funding = _normalise_funding(read_table(funding_path)) if funding_path else pd.DataFrame()
    universe = _normalise_universe(read_table(universe_path)) if universe_path else pd.DataFrame()

    start_ts = utc_timestamp(start) if start is not None else None
    end_ts = utc_timestamp(end) if end is not None else None
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("end must be at or after start")
    if start_ts is not None:
        bars = bars[bars["timestamp"] >= start_ts]
        factors = factors[factors["timestamp"] >= start_ts]
        if not funding.empty:
            funding = funding[funding["funding_time"] >= start_ts]
        # Universe rows are state updates: pre-start observations remain necessary
        # to establish membership at the beginning of the requested window.
        if not universe.empty:
            prior = universe[universe["timestamp"] < start_ts].groupby("symbol", sort=False).tail(1)
            universe = pd.concat([prior, universe[universe["timestamp"] >= start_ts]], ignore_index=True)
    if end_ts is not None:
        bars = bars[bars["timestamp"] <= end_ts]
        factors = factors[factors["timestamp"] <= end_ts]
        if not funding.empty:
            funding = funding[funding["funding_time"] <= end_ts]
        if not universe.empty:
            universe = universe[universe["timestamp"] <= end_ts]

    if bars.empty or factors.empty:
        raise ValueError(f"No bars or factors remain for {frequency} after date filtering")
    bar_symbols = set(bars["symbol"])
    factor_symbols = set(factors["symbol"])
    common = bar_symbols & factor_symbols
    if not common:
        raise ValueError(f"No common symbols between bars and factors for {frequency}")
    warnings: list[str] = []
    missing_factor_symbols = factor_symbols - bar_symbols
    if missing_factor_symbols:
        warnings.append(f"Ignored {len(missing_factor_symbols)} factor symbols without market data")
    if funding.empty:
        warnings.append("No funding data found; funding contribution is set to zero")
    if universe.empty:
        warnings.append("No point-in-time universe file found; survivorship bias may be present")
    return FrequencyData(bars=bars, factors=factors, funding=funding, universe=universe, warnings=warnings)
