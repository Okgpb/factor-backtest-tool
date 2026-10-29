from __future__ import annotations

import pytest

from crypto_factor_backtest.config import (
    BacktestConfig,
    ExecutionConfig,
    PortfolioConfig,
    apply_overrides,
    load_config,
)
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


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("frequencies", "1h"),
        ("frequencies", ["../1h"]),
        ("frequencies", ["1h", "60m"]),
        ("frequencies", [None]),
        ("frequencies", ["1M"]),
        ("factor_column", "symbol"),
        ("reverse", "false"),
        ("execution.include_funding", "false"),
        ("execution.delay_bars", 1.5),
        ("portfolio.groups", True),
        ("portfolio.gross_exposure", float("inf")),
        ("execution.fee_bps", float("nan")),
        ("universe.min_quote_volume", -1),
        ("analysis.ic_horizons", []),
        ("analysis.ic_horizons", [1, 1]),
        ("analysis.ic_horizons", [1.5]),
        ("analysis.ic_horizons", [True]),
        ("analysis.benchmark_symbol", ""),
        ("start", "NaT"),
        ("end", 123),
    ],
)
def test_invalid_overrides_fail_with_value_error(key, value) -> None:
    with pytest.raises(ValueError):
        apply_overrides(BacktestConfig(), {key: value})


@pytest.mark.parametrize("source", ["execution: null", "execution: []", "execution: {typo: 1}", "[", "false"])
def test_invalid_yaml_fails_cleanly(tmp_path, source) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_reversed_date_window_rejected() -> None:
    with pytest.raises(ValueError, match="end must"):
        BacktestConfig(start="2025-01-02", end="2025-01-01").validate()
