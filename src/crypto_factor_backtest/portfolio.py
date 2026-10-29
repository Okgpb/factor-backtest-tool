from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PortfolioConfig


def build_target_weights(
    factors: pd.DataFrame,
    eligible: pd.DataFrame,
    config: PortfolioConfig,
) -> pd.DataFrame:
    factors, eligible = factors.align(eligible, join="outer", axis=None)
    factors = factors.where(eligible.fillna(False)).replace([np.inf, -np.inf], np.nan)
    half_gross = config.gross_exposure / 2.0
    count = factors.notna().sum(axis=1)
    long_count = np.floor(count * config.long_quantile).clip(lower=1)
    short_count = np.floor(count * config.short_quantile).clip(lower=1)
    valid = (count >= config.min_assets) & (long_count + short_count <= count)
    # method='first' preserves the stable column-order tie break of the original
    # per-row sort while ranking the whole panel in one operation.
    rank = factors.rank(axis=1, method="first")
    longs = rank.gt(count - long_count, axis=0).mul(half_gross / long_count, axis=0)
    shorts = rank.le(short_count, axis=0).mul(half_gross / short_count, axis=0)
    target = (longs - shorts).where(valid, 0.0, axis=0)
    target.iloc[np.arange(len(target)) % config.rebalance_every != 0] = np.nan
    return target.ffill().fillna(0.0)


def calculate_group_returns(
    factors: pd.DataFrame,
    forward_returns: pd.DataFrame,
    eligible: pd.DataFrame,
    groups: int,
) -> pd.DataFrame:
    factors, forward_returns = factors.align(forward_returns, join="inner")
    signals = factors.where(eligible.reindex_like(factors).fillna(False)).replace([np.inf, -np.inf], np.nan)
    # Form groups using information available at signal time. A missing future
    # return must never move another asset to a different quantile.
    rank = signals.rank(axis=1, method="first", pct=True)
    assignments = np.ceil(rank * groups).clip(upper=groups)
    assignments = assignments.where(signals.notna().sum(axis=1) >= groups, axis=0)
    returns = pd.DataFrame(
        {
            f"group_{group}": forward_returns.where(assignments.eq(group)).mean(axis=1)
            for group in range(1, groups + 1)
        }
    )
    return returns.dropna(how="all").sort_index().rename_axis("timestamp")
