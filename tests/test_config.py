from __future__ import annotations

import pytest

from crypto_factor_backtest.config import BacktestConfig, ExecutionConfig, PortfolioConfig
from crypto_factor_backtest.timeutils import periods_per_year


def test_delay_must_prevent_lookahead() -> None:
    config = BacktestConfig(execution=ExecutionConfig(delay_bars=0))
    with pytest.raises(ValueError, match="look-ahead"):
        config.validate()


def test_quantiles_must_not_overlap() -> None:
    config = BacktestConfig(portfolio=PortfolioConfig(long_quantile=0.6, short_quantile=0.5))
    with pytest.raises(ValueError, match="cannot exceed"):
        config.validate()


def test_crypto_annualization_is_24_7() -> None:
    assert periods_per_year("1h") == 365 * 24
    assert periods_per_year("4h") == 365 * 6
