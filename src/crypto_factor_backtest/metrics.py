from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .timeutils import periods_per_year


def max_drawdown(nav: pd.Series) -> tuple[float, pd.Series]:
    drawdown = nav / nav.cummax() - 1.0
    return float(drawdown.min()) if not drawdown.empty else float("nan"), drawdown


def performance_metrics(stats: pd.DataFrame, frequency: str) -> dict[str, float | int]:
    returns = stats["net_return"].fillna(0.0)
    nav = stats["nav"].dropna()
    ppy = periods_per_year(frequency)
    periods = len(returns)
    total_return = float(nav.iloc[-1] - 1.0) if not nav.empty else float("nan")
    annual_return = float((nav.iloc[-1] ** (ppy / periods)) - 1.0) if periods and nav.iloc[-1] > 0 else float("nan")
    annual_volatility = float(returns.std(ddof=1) * math.sqrt(ppy)) if periods > 1 else float("nan")
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * math.sqrt(ppy))
        if periods > 1 and returns.std(ddof=1) > 0
        else float("nan")
    )
    downside = returns[returns < 0].std(ddof=1)
    sortino = float(returns.mean() / downside * math.sqrt(ppy)) if downside and downside > 0 else float("nan")
    mdd, _ = max_drawdown(nav)
    calmar = float(annual_return / abs(mdd)) if mdd < 0 and np.isfinite(annual_return) else float("nan")
    benchmark = stats.get("benchmark_return", pd.Series(index=stats.index, dtype=float)).fillna(0.0)
    benchmark_variance = benchmark.var()
    return_std = returns.std(ddof=1)
    benchmark_std = benchmark.std(ddof=1)
    beta = float(returns.cov(benchmark) / benchmark_variance) if benchmark_variance > 1e-30 else float("nan")
    correlation = (
        float(returns.corr(benchmark))
        if len(returns) > 1 and return_std > 1e-15 and benchmark_std > 1e-15
        else float("nan")
    )
    return {
        "periods": periods,
        "total_return": total_return,
        "annualized_return": annual_return,
        "annualized_volatility": annual_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "win_rate": float((returns > 0).mean()),
        "average_turnover": float(stats["turnover"].mean()),
        "total_fee": float(stats["fee_return"].sum()),
        "total_slippage": float(stats["slippage_return"].sum()),
        "total_funding": float(stats["funding_return"].sum()),
        "average_gross_exposure": float(stats["gross_exposure"].mean()),
        "average_net_exposure": float(stats["net_exposure"].mean()),
        "btc_beta": beta,
        "btc_correlation": correlation,
    }


def ic_summary(ic_series: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ic_series.columns:
        values = ic_series[column].dropna()
        std = values.std(ddof=1)
        rows.append(
            {
                "horizon": column,
                "observations": len(values),
                "mean_ic": values.mean(),
                "std_ic": std,
                "icir": values.mean() / std if std and std > 0 else np.nan,
                "positive_rate": (values > 0).mean() if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)
