from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from crypto_factor_backtest import cli
from crypto_factor_backtest.config import BacktestConfig, ExecutionConfig, PortfolioConfig, UniverseConfig
from crypto_factor_backtest.data import (
    _normalise_factors,
    _normalise_funding,
    _normalise_universe,
    _utc,
    load_frequency_data,
)
from crypto_factor_backtest.engine import _simulate_portfolio, _universe_panel, run_backtest
from crypto_factor_backtest.portfolio import calculate_group_returns
from crypto_factor_backtest.runner import run_batch


def _config():
    return BacktestConfig(
        portfolio=PortfolioConfig(min_assets=8),
        universe=UniverseConfig(min_history_bars=1, liquidity_lookback=1),
    )


@pytest.mark.parametrize("with_future_update", [False, True])
def test_present_but_empty_universe_never_enables_trading(tmp_path, small_data, with_future_update):
    market = tmp_path / "market" / "1h"
    factors = tmp_path / "factors"
    market.mkdir(parents=True)
    factors.mkdir()
    small_data.bars.to_parquet(market / "bars.parquet", index=False)
    small_data.factors.to_parquet(factors / "1h.parquet", index=False)
    universe = pd.DataFrame(columns=["timestamp", "symbol", "tradable"])
    if with_future_update:
        universe = pd.DataFrame([{"timestamp": "2025-02-01", "symbol": "BTCUSDT", "tradable": True}])
    universe.to_parquet(market / "universe.parquet", index=False)
    loaded = load_frequency_data(tmp_path / "market", factors, "1h", "factor", end="2025-01-02")
    result = run_backtest(loaded, _config(), "1h")
    assert result.weights.eq(0).all().all()
    assert not any("No point-in-time universe file found" in warning for warning in result.warnings)
    assert _universe_panel(pd.DataFrame(), result.weights.index, ["A"]).all().all()


@pytest.mark.parametrize(("assets", "groups"), [(25, 25), (50, 25), (100, 10), (49, 7), (10, 3)])
def test_group_boundaries_use_exact_integer_ranks(assets, groups):
    factors = pd.DataFrame([np.arange(1, assets + 1)], dtype=float)
    returns = factors / assets
    eligible = pd.DataFrame(True, index=factors.index, columns=factors.columns)
    actual = calculate_group_returns(factors, returns, eligible, groups)
    for group in range(1, groups + 1):
        members = [rank for rank in range(1, assets + 1) if (rank * groups + assets - 1) // assets == group]
        assert actual.iloc[0][f"group_{group}"] == pytest.approx(np.mean(members) / assets)


def test_benchmark_retains_price_change_across_missing_bar(small_data):
    # BTC is only a benchmark here, so a missing BTC bar must not stop strategy valuation.
    small_data.factors = small_data.factors[small_data.factors["symbol"] != "BTCUSDT"]
    timestamps = sorted(small_data.bars["timestamp"].unique())
    btc = small_data.bars["symbol"].eq("BTCUSDT")
    small_data.bars.loc[btc, "close"] = np.linspace(100, 190, len(timestamps))
    small_data.bars = small_data.bars[~(btc & small_data.bars["timestamp"].eq(timestamps[1]))]
    result = run_backtest(small_data, _config(), "1h")
    assert result.stats["benchmark_nav"].iloc[-1] == pytest.approx(1.9)
    assert any("Benchmark" in warning and "missing" in warning for warning in result.warnings)


def test_mixed_valid_iso_timestamps_are_not_rejected():
    values = pd.Series(
        [
            "2025-01-01T00:00:00Z",
            "2025-01-01T01:00:00.123Z",
            "2025-01-01T03:00:00+01:00",
            "2025-01-01T04:00:00.123456789Z",
        ]
    )
    actual = _utc(values, "test.timestamp")
    assert actual.iloc[1] == pd.Timestamp("2025-01-01T01:00:00.123Z")
    assert actual.iloc[2] == pd.Timestamp("2025-01-01T02:00:00Z")
    assert actual.iloc[3] == pd.Timestamp("2025-01-01T04:00:00.123456789Z")


@pytest.mark.parametrize("value", [1735689600000, 1735689600.0, True, "1735689600000"])
def test_epoch_timestamps_require_explicit_unit_conversion(value):
    with pytest.raises(ValueError, match="timestamp|epoch"):
        _utc(pd.Series([value]), "test.timestamp")


@pytest.mark.parametrize(
    ("normalise", "columns"),
    [
        (_normalise_universe, ["timestamp", "symbol", "tradable"]),
        (_normalise_funding, ["funding_time", "symbol", "funding_rate"]),
    ],
)
def test_empty_optional_tables_accept_unspecified_numeric_column_dtypes(normalise, columns):
    frame = pd.DataFrame({column: pd.Series([], dtype=float) for column in columns})
    result = normalise(frame)
    assert result.empty
    assert list(result.columns) == columns


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, "invalid"])
def test_duplicate_factor_keys_are_rejected_before_value_filtering(invalid_value):
    factors = pd.DataFrame(
        {
            "timestamp": ["2025-01-01"] * 2,
            "symbol": ["BTCUSDT"] * 2,
            "factor": [1, invalid_value],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        _normalise_factors(factors, "factor")


def test_reusing_nonempty_output_directory_fails_before_writes(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    previous = output / "run_summary.json"
    previous.write_text('{"previous": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="empty|non-empty"):
        run_batch(_config(), tmp_path / "missing_market", tmp_path / "missing_factors", output)
    assert json.loads(previous.read_text(encoding="utf-8")) == {"previous": True}
    assert list(output.iterdir()) == [previous]


def test_cli_default_output_is_under_working_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    captured = {}

    def run(config, data, factors, output):
        captured["output"] = output
        return {"output_dir": str(output)}

    monkeypatch.setattr(cli, "run_batch", run)
    assert cli.main(["--data-root", str(tmp_path), "--factor-root", str(tmp_path)]) == 0
    assert captured["output"].parent == tmp_path / "output"
    assert json.loads(capsys.readouterr().out)["output_dir"] == str(captured["output"])


@pytest.mark.parametrize(
    ("rebalance", "fee_bps", "include_funding"), [(1, 0, False), (2, 7, True), (3, 10, True)]
)
def test_simulator_matches_independent_contract_cash_ledger(rebalance, fee_bps, include_funding):
    index = pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC")
    rng = np.random.default_rng(520)
    opening = pd.DataFrame(100 * np.exp(rng.normal(0, 0.04, (8, 3))), index=index, columns=list("ABC"))
    closing = opening * np.exp(rng.normal(0, 0.03, opening.shape))
    funding = pd.DataFrame(rng.normal(0, 0.001, opening.shape), index=index, columns=opening.columns)
    targets = pd.DataFrame([[0.4, -0.5, 0.1], [-0.3, 0.4, -0.1]] * 4, index=index, columns=opening.columns)
    config = BacktestConfig(
        portfolio=PortfolioConfig(rebalance_every=rebalance),
        execution=ExecutionConfig(fee_bps=fee_bps, slippage_bps=3, include_funding=include_funding),
    )
    stats, _, deltas = _simulate_portfolio(targets, opening, closing, funding, config)
    quantity = np.zeros(3)
    equity = 1.0
    previous_close = opening.iloc[0].to_numpy()
    nav = []
    for number in range(len(index)):
        prices = opening.iloc[number].to_numpy()
        closes = closing.iloc[number].to_numpy()
        equity += np.dot(quantity, prices - previous_close)
        source = number - config.execution.delay_bars
        if source >= 0 and source % rebalance == 0:
            new_quantity = targets.iloc[number].to_numpy() * equity / prices
            equity -= np.dot(np.abs(new_quantity - quantity), prices) * (fee_bps + 3) / 10_000
            quantity = new_quantity
        else:
            assert deltas.iloc[number].eq(0).all()
        equity += np.dot(quantity, closes - prices)
        if include_funding:
            equity -= np.dot(quantity * prices, funding.iloc[number].to_numpy())
        nav.append(equity)
        previous_close = closes
    np.testing.assert_allclose((1 + stats["net_return"]).cumprod(), nav, rtol=1e-12, atol=1e-12)
