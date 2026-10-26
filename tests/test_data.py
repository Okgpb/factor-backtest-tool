from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crypto_factor_backtest.data import load_frequency_data


def test_duplicate_bars_fail_fast(tmp_path: Path) -> None:
    market = tmp_path / "market" / "1h"
    factors = tmp_path / "factors"
    market.mkdir(parents=True)
    factors.mkdir()
    row = {
        "timestamp": "2025-01-01T00:00:00Z",
        "symbol": "BTCUSDT",
        "open": 1,
        "high": 1,
        "low": 1,
        "close": 1,
        "volume": 1,
        "quote_volume": 1,
    }
    pd.DataFrame([row, row]).to_parquet(market / "bars.parquet", index=False)
    pd.DataFrame([{"timestamp": row["timestamp"], "symbol": "BTCUSDT", "factor": 1}]).to_parquet(
        factors / "1h.parquet", index=False
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_frequency_data(tmp_path / "market", factors, "1h", "factor")


def test_string_false_universe_value_stays_false(tmp_path: Path) -> None:
    market = tmp_path / "market" / "1h"
    factors = tmp_path / "factors"
    market.mkdir(parents=True)
    factors.mkdir()
    timestamp = "2025-01-01T00:00:00Z"
    pd.DataFrame(
        [{
            "timestamp": timestamp, "symbol": "BTCUSDT", "open": 1, "high": 1,
            "low": 1, "close": 1, "volume": 1, "quote_volume": 1,
        }]
    ).to_parquet(market / "bars.parquet", index=False)
    pd.DataFrame([{"timestamp": timestamp, "symbol": "BTCUSDT", "factor": 1}]).to_parquet(
        factors / "1h.parquet", index=False
    )
    pd.DataFrame([{"timestamp": timestamp, "symbol": "BTCUSDT", "tradable": "false"}]).to_parquet(
        market / "universe.parquet", index=False
    )
    loaded = load_frequency_data(tmp_path / "market", factors, "1h", "factor")
    assert bool(loaded.universe.iloc[0]["tradable"]) is False
