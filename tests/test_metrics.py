from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_factor_backtest.metrics import max_drawdown, performance_metrics


def _stats(returns):
    frame = pd.DataFrame({"net_return": returns})
    frame["nav"] = (1 + frame["net_return"]).cumprod()
    for column in [
        "turnover",
        "fee_return",
        "slippage_return",
        "funding_return",
        "gross_exposure",
        "net_exposure",
    ]:
        frame[column] = 0.0
    return frame


def test_drawdown_includes_initial_capital() -> None:
    maximum, drawdown = max_drawdown(pd.Series([0.9, 0.8, 1.1, 0.99]))
    assert maximum == pytest.approx(-0.2)
    assert drawdown.iloc[0] == pytest.approx(-0.1)


def test_sortino_uses_downside_deviation_over_all_periods() -> None:
    returns = [0.02, -0.01, 0.03, -0.01]
    metrics = performance_metrics(_stats(returns), "1h")
    downside = np.sqrt((0.01**2 + 0.01**2) / 4)
    assert metrics["sortino"] == pytest.approx(np.mean(returns) / downside * np.sqrt(365 * 24))


def test_empty_metrics_do_not_index_missing_nav() -> None:
    metrics = performance_metrics(_stats([]), "1h")
    assert metrics["periods"] == 0
    assert np.isnan(metrics["annualized_return"])
