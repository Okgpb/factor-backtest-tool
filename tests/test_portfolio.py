from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_factor_backtest.config import PortfolioConfig
from crypto_factor_backtest.portfolio import build_target_weights, calculate_group_returns


def test_quantile_weights_are_dollar_neutral() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    columns = [f"S{i}" for i in range(10)]
    factors = pd.DataFrame([range(10), range(10)], index=index, columns=columns, dtype=float)
    eligible = pd.DataFrame(True, index=index, columns=columns)
    weights = build_target_weights(
        factors,
        eligible,
        PortfolioConfig(long_quantile=0.2, short_quantile=0.2, min_assets=8),
    )
    assert (weights.sum(axis=1).abs() < 1e-12).all()
    assert ((weights.abs().sum(axis=1) - 1.0).abs() < 1e-12).all()
    assert (weights.gt(0).sum(axis=1) == 2).all()
    assert (weights.lt(0).sum(axis=1) == 2).all()


def test_vectorized_weights_match_stable_reference_with_ties_and_missing_values() -> None:
    rng = np.random.default_rng(42)
    factors = pd.DataFrame(rng.integers(0, 5, size=(40, 20)), dtype=float)
    factors = factors.mask(rng.random(factors.shape) < 0.2)
    eligible = pd.DataFrame(rng.random(factors.shape) > 0.2)
    config = PortfolioConfig(long_quantile=0.3, short_quantile=0.1, min_assets=8, rebalance_every=3)
    expected = pd.DataFrame(0.0, index=factors.index, columns=factors.columns)
    for number, row in factors.where(eligible).iterrows():
        if number % config.rebalance_every:
            expected.loc[number] = np.nan
            continue
        signal = row.dropna().sort_values(kind="stable")
        if len(signal) < config.min_assets:
            continue
        longs = max(1, int(len(signal) * config.long_quantile))
        shorts = max(1, int(len(signal) * config.short_quantile))
        expected.loc[number, signal.index[-longs:]] = config.gross_exposure / 2 / longs
        expected.loc[number, signal.index[:shorts]] = -config.gross_exposure / 2 / shorts
    pd.testing.assert_frame_equal(build_target_weights(factors, eligible, config), expected.ffill().fillna(0))


def test_missing_future_returns_do_not_change_group_membership() -> None:
    factors = pd.DataFrame([[1, 2, 3, 4]], columns=list("ABCD"), dtype=float)
    returns = pd.DataFrame([[0.1, np.nan, 0.3, 0.4]], columns=factors.columns)
    eligible = pd.DataFrame(True, index=factors.index, columns=factors.columns)
    groups = calculate_group_returns(factors, returns, eligible, 2)
    assert groups.loc[0, "group_1"] == 0.1
    assert groups.loc[0, "group_2"] == 0.35
