from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crypto_factor_backtest.data import (
    _normalise_bars,
    _normalise_factors,
    _normalise_funding,
    _normalise_universe,
    load_frequency_data,
)


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
        [
            {
                "timestamp": timestamp,
                "symbol": "BTCUSDT",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
                "quote_volume": 1,
            }
        ]
    ).to_parquet(market / "bars.parquet", index=False)
    pd.DataFrame([{"timestamp": timestamp, "symbol": "BTCUSDT", "factor": 1}]).to_parquet(
        factors / "1h.parquet", index=False
    )
    pd.DataFrame([{"timestamp": timestamp, "symbol": "BTCUSDT", "tradable": "false"}]).to_parquet(
        market / "universe.parquet", index=False
    )
    loaded = load_frequency_data(tmp_path / "market", factors, "1h", "factor")
    assert bool(loaded.universe.iloc[0]["tradable"]) is False


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("open", float("inf")),
        ("volume", -1),
        ("quote_volume", float("inf")),
        ("symbol", None),
        ("symbol", 123),
        ("symbol", "  "),
        ("high", 0.5),
        ("low", 1000),
    ],
)
def test_invalid_market_values_rejected(small_data, column, value) -> None:
    bars = small_data.bars.copy()
    bars[column] = value
    with pytest.raises(ValueError):
        _normalise_bars(bars)


@pytest.mark.parametrize("value", [None, float("nan"), 2, -1, "unknown"])
def test_invalid_universe_state_rejected(value) -> None:
    frame = pd.DataFrame([{"timestamp": "2025-01-01", "symbol": "BTCUSDT", "tradable": value}])
    with pytest.raises(ValueError):
        _normalise_universe(frame)


def test_duplicate_and_nonfinite_funding_rejected() -> None:
    row = {"funding_time": "2025-01-01", "symbol": "BTCUSDT", "funding_rate": 0.01}
    with pytest.raises(ValueError, match="duplicate"):
        _normalise_funding(pd.DataFrame([row, row]))
    row["funding_rate"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        _normalise_funding(pd.DataFrame([row]))


def test_date_window_preserves_prior_universe_state(tmp_path, small_data) -> None:
    market = tmp_path / "market" / "1h"
    factors = tmp_path / "factors"
    market.mkdir(parents=True)
    factors.mkdir()
    small_data.bars.to_parquet(market / "bars.parquet", index=False)
    small_data.factors.to_parquet(factors / "1h.parquet", index=False)
    pd.DataFrame([{"timestamp": "2024-12-31", "symbol": "BTCUSDT", "tradable": True}]).to_parquet(
        market / "universe.parquet",
        index=False,
    )
    loaded = load_frequency_data(tmp_path / "market", factors, "1h", "factor", start="2025-01-01 03:00")
    assert loaded.bars["timestamp"].min() == pd.Timestamp("2025-01-01 03:00", tz="UTC")
    assert len(loaded.universe) == 1


def test_symbols_normalised_without_dropping_string_identity(small_data) -> None:
    factors = small_data.factors.copy()
    factors["symbol"] = " 1000pepeusdt "
    factors = factors.iloc[:1]
    assert _normalise_factors(factors, "factor").iloc[0]["symbol"] == "1000PEPEUSDT"
