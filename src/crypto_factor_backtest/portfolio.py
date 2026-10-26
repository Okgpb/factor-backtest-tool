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
    factors = factors.where(eligible.fillna(False))
    target = pd.DataFrame(0.0, index=factors.index, columns=factors.columns)
    half_gross = config.gross_exposure / 2.0

    for row_number, (timestamp, row) in enumerate(factors.iterrows()):
        if row_number % config.rebalance_every:
            target.loc[timestamp] = np.nan
            continue
        signal = row.dropna().sort_values(kind="stable")
        count = len(signal)
        if count < config.min_assets:
            continue
        long_count = max(1, int(np.floor(count * config.long_quantile)))
        short_count = max(1, int(np.floor(count * config.short_quantile)))
        if long_count + short_count > count:
            continue
        short_symbols = signal.index[:short_count]
        long_symbols = signal.index[-long_count:]
        target.loc[timestamp, long_symbols] = half_gross / long_count
        target.loc[timestamp, short_symbols] = -half_gross / short_count

    return target.ffill().fillna(0.0)


def calculate_group_returns(
    factors: pd.DataFrame,
    forward_returns: pd.DataFrame,
    eligible: pd.DataFrame,
    groups: int,
) -> pd.DataFrame:
    values: list[dict[str, float | pd.Timestamp]] = []
    common_index = factors.index.intersection(forward_returns.index)
    for timestamp in common_index:
        frame = pd.concat(
            [
                factors.loc[timestamp].rename("factor"),
                forward_returns.loc[timestamp].rename("return"),
                eligible.loc[timestamp].rename("eligible"),
            ],
            axis=1,
        )
        frame = frame[frame["eligible"].fillna(False)].dropna(subset=["factor", "return"])
        if len(frame) < groups:
            continue
        rank = frame["factor"].rank(method="first", pct=True)
        frame["group"] = np.minimum((rank * groups).apply(np.ceil).astype(int), groups)
        record: dict[str, float | pd.Timestamp] = {"timestamp": timestamp}
        for group, value in frame.groupby("group")["return"].mean().items():
            record[f"group_{int(group)}"] = float(value)
        values.append(record)
    if not values:
        return pd.DataFrame(columns=[f"group_{i}" for i in range(1, groups + 1)])
    return pd.DataFrame(values).set_index("timestamp").sort_index().reindex(
        columns=[f"group_{i}" for i in range(1, groups + 1)]
    )
