from __future__ import annotations

import pandas as pd

from crypto_factor_backtest.config import PortfolioConfig
from crypto_factor_backtest.portfolio import build_target_weights


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
