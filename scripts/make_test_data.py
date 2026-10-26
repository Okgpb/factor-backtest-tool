#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "test_data"
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
]


def _base_bars() -> pd.DataFrame:
    rng = np.random.default_rng(20261026)
    timestamps = pd.date_range("2025-01-01 00:59:59", periods=720, freq="h", tz="UTC")
    frames = []
    for number, symbol in enumerate(SYMBOLS):
        shocks = rng.normal(0.00012 + number * 0.000005, 0.006 + number * 0.00008, len(timestamps))
        close = (100 + 17 * number) * np.exp(np.cumsum(shocks))
        open_price = np.r_[close[0] / np.exp(shocks[0]), close[:-1]]
        spread = rng.uniform(0.0002, 0.004, len(timestamps))
        high = np.maximum(open_price, close) * (1 + spread)
        low = np.minimum(open_price, close) * (1 - spread)
        quote_volume = rng.lognormal(15.5 - number * 0.02, 0.35, len(timestamps))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "symbol": symbol,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": quote_volume / close,
                    "quote_volume": quote_volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])


def _resample_bars(bars: pd.DataFrame, frequency: str) -> pd.DataFrame:
    output = []
    for symbol, frame in bars.groupby("symbol"):
        values = frame.set_index("timestamp").sort_index()
        resampled = values.resample(frequency, label="right", closed="right").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "quote_volume": "sum",
            }
        )
        resampled["symbol"] = symbol
        output.append(resampled.dropna().reset_index())
    return pd.concat(output, ignore_index=True).sort_values(["timestamp", "symbol"])


def _factor(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars.pivot(index="timestamp", columns="symbol", values="close")
    momentum = close.pct_change(6).shift(1)
    reversal = -close.pct_change(1).shift(1)
    factor = momentum + 0.2 * reversal
    return (
        factor.rename_axis(columns="symbol")
        .stack()
        .rename("momentum_factor")
        .reset_index()
        .dropna(subset=["momentum_factor"])
    )


def _funding(index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(1234)
    funding_times = index[index.hour.isin([0, 8, 16])]
    rows = []
    for symbol_number, symbol in enumerate(SYMBOLS):
        for i, timestamp in enumerate(funding_times):
            rows.append(
                {
                    "funding_time": timestamp,
                    "symbol": symbol,
                    "funding_rate": 0.00005 + 0.00001 * np.sin(i + symbol_number) + rng.normal(0, 0.000002),
                }
            )
    return pd.DataFrame(rows)


def _universe(bars: pd.DataFrame) -> pd.DataFrame:
    rows = bars[["timestamp", "symbol"]].copy()
    rows["tradable"] = True
    first_times = sorted(rows["timestamp"].unique())
    rows.loc[
        (rows["symbol"] == "BCHUSDT") & (rows["timestamp"] < first_times[min(48, len(first_times) - 1)]),
        "tradable",
    ] = False
    rows.loc[
        (rows["symbol"] == "LTCUSDT") & (rows["timestamp"] >= first_times[max(0, len(first_times) - 24)]),
        "tradable",
    ] = False
    return rows


def build_fixture(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    base = _base_bars()
    frequencies = {"1h": base, "4h": _resample_bars(base, "4h")}
    factor_root = root / "factors"
    factor_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source": "deterministic synthetic fixture",
        "seed": 20261026,
        "factor_column": "momentum_factor",
        "frequencies": {},
    }
    for frequency, bars in frequencies.items():
        market_dir = root / "market" / frequency
        market_dir.mkdir(parents=True, exist_ok=True)
        bars.to_parquet(market_dir / "bars.parquet", index=False)
        funding = _funding(pd.DatetimeIndex(sorted(bars["timestamp"].unique())))
        funding.to_parquet(market_dir / "funding.parquet", index=False)
        universe = _universe(bars)
        universe.to_parquet(market_dir / "universe.parquet", index=False)
        factors = _factor(bars)
        factors.to_parquet(factor_root / f"{frequency}.parquet", index=False)
        manifest["frequencies"][frequency] = {
            "bars": len(bars), "factors": len(factors), "funding": len(funding), "universe": len(universe)
        }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(build_fixture(), ensure_ascii=False, indent=2))
