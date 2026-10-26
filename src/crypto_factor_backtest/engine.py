from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .data import FrequencyData
from .metrics import ic_summary, max_drawdown, performance_metrics
from .portfolio import build_target_weights, calculate_group_returns


@dataclass(slots=True)
class BacktestResult:
    frequency: str
    stats: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    ic_series: pd.DataFrame
    ic_summary: pd.DataFrame
    group_returns: pd.DataFrame
    metrics: dict[str, float | int | str]
    warnings: list[str]


def _panel(frame: pd.DataFrame, column: str, index: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    result = frame.pivot(index="timestamp", columns="symbol", values=column).reindex(
        index=index, columns=symbols
    )
    result.index.name = "timestamp"
    result.columns.name = "symbol"
    return result


def _funding_panel(
    funding: pd.DataFrame,
    index: pd.DatetimeIndex,
    symbols: list[str],
) -> pd.DataFrame:
    result = pd.DataFrame(0.0, index=index, columns=symbols)
    if funding.empty:
        return result
    index_values = index.view("i8")
    event_values = pd.DatetimeIndex(funding["funding_time"]).view("i8")
    locations = np.searchsorted(index_values, event_values, side="left")
    valid = locations < len(index)
    mapped = funding.loc[valid, ["symbol", "funding_rate"]].copy()
    mapped["timestamp"] = index[locations[valid]]
    mapped = mapped[mapped["symbol"].isin(symbols)]
    if mapped.empty:
        return result
    values = mapped.groupby(["timestamp", "symbol"])["funding_rate"].sum().unstack(fill_value=0.0)
    return result.add(values.reindex(index=index, columns=symbols).fillna(0.0), fill_value=0.0)


def _universe_panel(
    universe: pd.DataFrame,
    index: pd.DatetimeIndex,
    symbols: list[str],
) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame(True, index=index, columns=symbols)
    values = universe.pivot(index="timestamp", columns="symbol", values="tradable")
    return values.reindex(index=index, columns=symbols).ffill().fillna(False).astype(bool)


def _ic_series(
    factors: pd.DataFrame,
    open_price: pd.DataFrame,
    close_price: pd.DataFrame,
    eligible: pd.DataFrame,
    horizons: list[int],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    output: dict[str, pd.Series] = {}
    forward: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        future = close_price.shift(-horizon) / open_price.shift(-1) - 1.0
        future = future.where(eligible)
        forward[horizon] = future
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            output[f"{horizon}_bar"] = factors.rank(axis=1).corrwith(future.rank(axis=1), axis=1)
    return pd.DataFrame(output), forward


def _trades_from_deltas(
    deltas: pd.DataFrame,
    execution_price: pd.DataFrame,
    total_cost_bps: float,
) -> pd.DataFrame:
    deltas = deltas.copy()
    deltas.index.name = "timestamp"
    deltas.columns.name = "symbol"
    rows = deltas.stack()
    rows = rows[rows.abs() > 1e-15].rename("weight_change").reset_index()
    if rows.empty:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "side", "weight_change", "execution_price", "estimated_cost"]
        )
    rows["side"] = np.where(rows["weight_change"] > 0, "buy", "sell")
    execution_price = execution_price.copy()
    execution_price.index.name = "timestamp"
    execution_price.columns.name = "symbol"
    prices = execution_price.stack().rename("execution_price").reset_index()
    rows = rows.merge(prices, on=["timestamp", "symbol"], how="left")
    rows["estimated_cost"] = rows["weight_change"].abs() * total_cost_bps / 10_000.0
    return rows[["timestamp", "symbol", "side", "weight_change", "execution_price", "estimated_cost"]]


def _simulate_portfolio(
    targets: pd.DataFrame,
    open_price: pd.DataFrame,
    close_price: pd.DataFrame,
    funding_rates: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index, columns = targets.index, targets.columns
    active_weights = pd.DataFrame(0.0, index=index, columns=columns)
    trade_deltas = pd.DataFrame(0.0, index=index, columns=columns)
    records: list[dict[str, float | int | pd.Timestamp]] = []
    end_weights = pd.Series(0.0, index=columns)
    previous_close: pd.Series | None = None

    for row_number, timestamp in enumerate(index):
        opening = open_price.loc[timestamp]
        closing = close_price.loc[timestamp]
        if previous_close is None:
            gap_asset_return = pd.Series(0.0, index=columns)
        else:
            gap_asset_return = (opening / previous_close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gap_return = float((end_weights * gap_asset_return).sum())
        if 1.0 + gap_return <= 0:
            raise RuntimeError(f"Portfolio equity became non-positive before rebalance at {timestamp}")
        pretrade = end_weights.mul(1.0 + gap_asset_return).div(1.0 + gap_return)

        source_row = row_number - config.execution.delay_bars
        scheduled = source_row >= 0 and source_row % config.portfolio.rebalance_every == 0
        if scheduled:
            active = targets.loc[timestamp].fillna(0.0)
            delta = active - pretrade
        else:
            active = pretrade
            delta = pd.Series(0.0, index=columns)
        active_weights.loc[timestamp] = active
        trade_deltas.loc[timestamp] = delta

        turnover = float(delta.abs().sum())
        fee_return = turnover * config.execution.fee_bps / 10_000.0
        slippage_return = turnover * config.execution.slippage_bps / 10_000.0
        intrabar_asset_return = (closing / opening - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        intrabar_return = float((active * intrabar_asset_return).sum())
        funding_return = (
            -float((active * funding_rates.loc[timestamp]).sum())
            if config.execution.include_funding
            else 0.0
        )
        post_open_return = intrabar_return + funding_return - fee_return - slippage_return
        net_return = (1.0 + gap_return) * (1.0 + post_open_return) - 1.0
        gross_return = (1.0 + gap_return) * (1.0 + intrabar_return) - 1.0
        long_return = float((end_weights.clip(lower=0.0) * gap_asset_return).sum())
        long_return += (1.0 + gap_return) * float(
            (active.clip(lower=0.0) * intrabar_asset_return).sum()
        )
        short_return = float((end_weights.clip(upper=0.0) * gap_asset_return).sum())
        short_return += (1.0 + gap_return) * float(
            (active.clip(upper=0.0) * intrabar_asset_return).sum()
        )
        denominator = 1.0 + post_open_return
        if denominator <= 0:
            raise RuntimeError(f"Portfolio equity became non-positive during bar ending {timestamp}")
        end_weights = active.mul(1.0 + intrabar_asset_return).div(denominator)
        previous_close = closing
        records.append(
            {
                "timestamp": timestamp,
                "gross_return": gross_return,
                "long_return": long_return,
                "short_return": short_return,
                "funding_return": (1.0 + gap_return) * funding_return,
                "fee_return": (1.0 + gap_return) * fee_return,
                "slippage_return": (1.0 + gap_return) * slippage_return,
                "net_return": net_return,
                "turnover": turnover,
                "gross_exposure": float(active.abs().sum()),
                "net_exposure": float(active.sum()),
                "long_count": int((active > 0).sum()),
                "short_count": int((active < 0).sum()),
            }
        )
    stats = pd.DataFrame(records).set_index("timestamp")
    stats.index.name = "timestamp"
    return stats, active_weights, trade_deltas


def run_backtest(data: FrequencyData, config: BacktestConfig, frequency: str) -> BacktestResult:
    config.validate()
    bar_index = pd.DatetimeIndex(sorted(data.bars["timestamp"].unique()))
    symbols = sorted(set(data.bars["symbol"]) & set(data.factors["symbol"]))
    open_price = _panel(data.bars, "open", bar_index, symbols)
    close_price = _panel(data.bars, "close", bar_index, symbols)
    quote_volume = _panel(data.bars, "quote_volume", bar_index, symbols)
    factors = data.factors.pivot(index="timestamp", columns="symbol", values=config.factor_column).reindex(
        index=bar_index, columns=symbols
    )
    if config.reverse:
        factors = -factors

    history_ok = close_price.notna().cumsum() >= config.universe.min_history_bars
    liquidity = quote_volume.rolling(
        config.universe.liquidity_lookback,
        min_periods=config.universe.liquidity_lookback,
    ).mean()
    liquidity_ok = liquidity >= config.universe.min_quote_volume
    point_in_time = _universe_panel(data.universe, bar_index, symbols)
    eligible = history_ok & liquidity_ok & point_in_time & open_price.notna() & close_price.notna()

    ic_series, forward_returns = _ic_series(
        factors,
        open_price,
        close_price,
        eligible,
        config.analysis.ic_horizons,
    )
    group_returns = calculate_group_returns(
        factors,
        forward_returns[config.analysis.ic_horizons[0]],
        eligible,
        config.portfolio.groups,
    )

    signal_weights = build_target_weights(factors, eligible, config.portfolio)
    target_weights = signal_weights.shift(config.execution.delay_bars).fillna(0.0)
    asset_returns = close_price / open_price - 1.0
    unavailable = (target_weights.abs() > 0) & asset_returns.isna()
    warnings = list(data.warnings)
    if unavailable.any().any():
        warnings.append(
            f"{int(unavailable.sum().sum())} held symbol-bars lacked returns and were valued at zero return"
        )
    funding_rates = _funding_panel(data.funding, bar_index, symbols)
    stats, weights, trade_deltas = _simulate_portfolio(
        target_weights,
        open_price,
        close_price,
        funding_rates,
        config,
    )

    benchmark_symbol = config.analysis.benchmark_symbol.upper()
    if benchmark_symbol in open_price:
        benchmark_return = close_price[benchmark_symbol].pct_change(fill_method=None).fillna(0.0)
    else:
        benchmark_return = pd.Series(0.0, index=bar_index)
        warnings.append(f"Benchmark {benchmark_symbol} not found; benchmark return is set to zero")

    stats["benchmark_return"] = benchmark_return
    stats["nav"] = (1.0 + stats["net_return"]).cumprod()
    stats["benchmark_nav"] = (1.0 + stats["benchmark_return"]).cumprod()
    _, stats["drawdown"] = max_drawdown(stats["nav"])
    metrics: dict[str, float | int | str] = performance_metrics(stats, frequency)
    metrics["frequency"] = frequency
    metrics["symbols"] = len(symbols)
    metrics["mean_rank_ic_1"] = float(ic_series.iloc[:, 0].mean()) if not ic_series.empty else float("nan")

    weights.index.name = "timestamp"
    trades = _trades_from_deltas(
        trade_deltas,
        open_price,
        config.execution.fee_bps + config.execution.slippage_bps,
    )
    return BacktestResult(
        frequency=frequency,
        stats=stats,
        weights=weights,
        trades=trades,
        ic_series=ic_series,
        ic_summary=ic_summary(ic_series),
        group_returns=group_returns,
        metrics=metrics,
        warnings=warnings,
    )
