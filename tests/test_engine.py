from __future__ import annotations

import pandas as pd

from crypto_factor_backtest.config import BacktestConfig, ExecutionConfig, PortfolioConfig, UniverseConfig
from crypto_factor_backtest.data import FrequencyData
from crypto_factor_backtest.engine import run_backtest


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

    positive_long = pd.DataFrame(
        [{"funding_time": timestamp, "symbol": long_symbol, "funding_rate": 0.01}]
    )
    long_result = run_backtest(
        FrequencyData(bars=small_data.bars, factors=small_data.factors, funding=positive_long),
        _config(delay_bars=1, fee_bps=0, slippage_bps=0),
        "1h",
    )
    assert long_result.stats.loc[timestamp, "funding_return"] < 0

    positive_short = pd.DataFrame(
        [{"funding_time": timestamp, "symbol": short_symbol, "funding_rate": 0.01}]
    )
    short_result = run_backtest(
        FrequencyData(bars=small_data.bars, factors=small_data.factors, funding=positive_short),
        _config(delay_bars=1, fee_bps=0, slippage_bps=0),
        "1h",
    )
    assert short_result.stats.loc[timestamp, "funding_return"] > 0
