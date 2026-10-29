from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_factor_backtest.config import BacktestConfig, ExecutionConfig, PortfolioConfig, UniverseConfig
from crypto_factor_backtest.data import FrequencyData
from crypto_factor_backtest.engine import _funding_panel, _ic_series, _universe_panel, run_backtest


def _config(**execution: float | int | bool) -> BacktestConfig:
    return BacktestConfig(
        factor_column="factor",
        frequencies=["1h"],
        portfolio=PortfolioConfig(min_assets=8),
        execution=ExecutionConfig(**execution),
        universe=UniverseConfig(min_history_bars=1, liquidity_lookback=1),
    )


def test_signal_is_delayed_one_bar(small_data: FrequencyData) -> None:
    result = run_backtest(small_data, _config(delay_bars=1, fee_bps=0, slippage_bps=0), "1h")
    assert result.weights.iloc[0].abs().sum() == 0
    assert result.weights.iloc[1].abs().sum() == 1


def test_future_factor_change_cannot_change_past_weights(small_data: FrequencyData) -> None:
    original = run_backtest(small_data, _config(delay_bars=1, fee_bps=0, slippage_bps=0), "1h")
    changed_factors = small_data.factors.copy()
    cutoff = changed_factors["timestamp"].sort_values().unique()[6]
    changed_factors.loc[changed_factors["timestamp"] >= cutoff, "factor"] *= -100
    changed = run_backtest(
        FrequencyData(bars=small_data.bars, factors=changed_factors),
        _config(delay_bars=1, fee_bps=0, slippage_bps=0),
        "1h",
    )
    pd.testing.assert_frame_equal(original.weights.iloc[:6], changed.weights.iloc[:6])


def test_cost_is_charged_on_absolute_turnover(small_data: FrequencyData) -> None:
    result = run_backtest(small_data, _config(delay_bars=1, fee_bps=10, slippage_bps=5), "1h")
    expected = result.stats["turnover"] * 10 / 10_000
    pd.testing.assert_series_equal(result.stats["fee_return"], expected, check_names=False)
    expected_slippage = result.stats["turnover"] * 5 / 10_000
    pd.testing.assert_series_equal(result.stats["slippage_return"], expected_slippage, check_names=False)


def test_non_rebalance_bars_do_not_create_trades(small_data: FrequencyData) -> None:
    config = _config(delay_bars=1, fee_bps=0, slippage_bps=0)
    config.portfolio.rebalance_every = 2
    result = run_backtest(small_data, config, "1h")
    assert (result.stats["turnover"].iloc[2::2] == 0).all()


def test_positive_funding_costs_longs_and_rewards_shorts(small_data: FrequencyData) -> None:
    baseline = run_backtest(small_data, _config(delay_bars=1, fee_bps=0, slippage_bps=0), "1h")
    timestamp = baseline.weights.index[2]
    long_symbol = baseline.weights.loc[timestamp].idxmax()
    short_symbol = baseline.weights.loc[timestamp].idxmin()

    positive_long = pd.DataFrame([{"funding_time": timestamp, "symbol": long_symbol, "funding_rate": 0.01}])
    long_result = run_backtest(
        FrequencyData(bars=small_data.bars, factors=small_data.factors, funding=positive_long),
        _config(delay_bars=1, fee_bps=0, slippage_bps=0),
        "1h",
    )
    assert long_result.stats.loc[timestamp, "funding_return"] < 0

    positive_short = pd.DataFrame([{"funding_time": timestamp, "symbol": short_symbol, "funding_rate": 0.01}])
    short_result = run_backtest(
        FrequencyData(bars=small_data.bars, factors=small_data.factors, funding=positive_short),
        _config(delay_bars=1, fee_bps=0, slippage_bps=0),
        "1h",
    )
    assert short_result.stats.loc[timestamp, "funding_return"] > 0


def test_benchmark_does_not_require_a_factor(small_data) -> None:
    factors = small_data.factors[small_data.factors["symbol"] != "BTCUSDT"]
    expected = small_data.bars[small_data.bars["symbol"].eq("BTCUSDT")].set_index("timestamp")["close"]
    # Give BTC a changing price so a zero-filled benchmark cannot pass.
    bars = small_data.bars.copy()
    bars.loc[bars["symbol"].eq("BTCUSDT"), "close"] = expected.to_numpy() * np.arange(1, len(expected) + 1)
    result = run_backtest(FrequencyData(bars=bars, factors=factors), _config(), "1h")
    expected = bars[bars["symbol"].eq("BTCUSDT")].set_index("timestamp")["close"]
    pd.testing.assert_series_equal(
        result.stats["benchmark_return"],
        expected.pct_change(fill_method=None).fillna(0),
        check_names=False,
    )
    assert "BTCUSDT" not in result.weights
    assert not any("Benchmark" in warning for warning in result.warnings)


def test_universe_updates_between_bars_are_carried_forward() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    universe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-12-31T23:00Z", "2025-01-01T00:30Z", "2025-01-01T01:30Z"]),
            "symbol": ["A"] * 3,
            "tradable": [True, False, True],
        }
    )
    panel = _universe_panel(universe, index, ["A", "B"])
    assert panel["A"].tolist() == [True, False, True]
    assert not panel["B"].any()


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
def test_funding_mapping_handles_timestamp_resolutions(unit) -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    if hasattr(index, "as_unit"):
        index = index.as_unit(unit)
    funding = pd.DataFrame(
        {
            "funding_time": pd.to_datetime(["2025-01-01T00:30Z", "2025-01-01T01:00Z", "2025-01-01T04:00Z"]),
            "symbol": ["A"] * 3,
            "funding_rate": [0.01, 0.02, 0.05],
        }
    )
    assert _funding_panel(funding, index, ["A"])["A"].tolist() == [0, 0.03, 0, 0]


def test_ic_uses_common_nonmissing_pairs_and_execution_delay() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    factors = pd.DataFrame([[1, 2, 3, 4]] * 4, index=index, columns=list("ABCD"), dtype=float)
    opening = pd.DataFrame(100.0, index=index, columns=factors.columns)
    closing = opening.copy()
    closing.iloc[1] = [120, np.nan, 110, 130]
    closing.iloc[2] = [140, 130, 120, 110]
    eligible = pd.DataFrame(True, index=index, columns=factors.columns)
    ic, _ = _ic_series(factors, opening, closing, eligible, [1])
    assert ic.iloc[0, 0] == pytest.approx(0.5)
    delayed_ic, forward = _ic_series(factors, opening, closing, eligible, [1], delay_bars=2)
    assert delayed_ic.iloc[0, 0] == pytest.approx(-1)
    assert forward[1].iloc[0, 0] == pytest.approx(0.4)
    assert forward[1].iloc[-2:].isna().all().all()


def test_missing_held_prices_cannot_hide_losses(small_data) -> None:
    baseline = run_backtest(small_data, _config(), "1h")
    timestamp = baseline.weights.index[2]
    symbol = baseline.weights.loc[timestamp].idxmax()
    bars = small_data.bars[
        ~(small_data.bars["timestamp"].eq(timestamp) & small_data.bars["symbol"].eq(symbol))
    ]
    with pytest.raises(ValueError, match="Missing execution/valuation prices"):
        run_backtest(FrequencyData(bars=bars, factors=small_data.factors), _config(), "1h")


def test_missing_entire_bar_cannot_compress_time(small_data) -> None:
    bars = small_data.bars[small_data.bars["timestamp"] != small_data.bars["timestamp"].unique()[3]]
    with pytest.raises(ValueError, match="regular 1h grid"):
        run_backtest(FrequencyData(bars=bars, factors=small_data.factors), _config(), "1h")


def test_unaligned_factor_timestamps_cannot_produce_a_silent_empty_strategy(small_data) -> None:
    factors = small_data.factors.copy()
    factors["timestamp"] += pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match="No finite factors align"):
        run_backtest(FrequencyData(bars=small_data.bars, factors=factors), _config(), "1h")


def test_return_attribution_reconciles_with_gaps_and_costs(small_data) -> None:
    config = _config(fee_bps=10, slippage_bps=5)
    config.portfolio.rebalance_every = 2
    result = run_backtest(small_data, config, "1h")
    stats = result.stats
    np.testing.assert_allclose(
        stats["long_return"] + stats["short_return"], stats["gross_return"], atol=1e-14
    )
    np.testing.assert_allclose(
        stats["gross_return"] + stats["funding_return"] - stats["fee_return"] - stats["slippage_return"],
        stats["net_return"],
        atol=1e-14,
    )


def test_mean_one_bar_ic_does_not_depend_on_horizon_order(small_data) -> None:
    bars = small_data.bars.copy()
    odd_hour = bars["timestamp"].dt.hour.mod(2).eq(1)
    bars.loc[odd_hour, "close"] = 2 * bars.loc[odd_hour, "open"] - bars.loc[odd_hour, "close"]
    data = FrequencyData(bars=bars, factors=small_data.factors)
    config = _config()
    config.analysis.ic_horizons = [2, 1]
    result = run_backtest(data, config, "1h")
    assert not np.isclose(result.ic_series["1_bar"].mean(), result.ic_series["2_bar"].mean())
    assert result.metrics["mean_rank_ic_1"] == pytest.approx(result.ic_series["1_bar"].mean())
    config.analysis.ic_horizons = [2]
    result = run_backtest(data, config, "1h")
    assert np.isnan(result.metrics["mean_rank_ic_1"])
