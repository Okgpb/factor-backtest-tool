# Backtest Methodology

## Portfolio formation

For each timestamp, eligible symbols are ranked by factor value. The highest configured fraction is
held long and the lowest fraction short. Each leg is equal-weighted and receives half of configured
gross exposure. If the cross-section has fewer than `min_assets`, no new portfolio is opened.

Eligibility combines observed prices, minimum history, rolling quote-volume threshold and optional
point-in-time universe membership. Existing targets are carried between configured rebalance bars.

## No-look-ahead convention

The factor timestamp is the bar close at which it becomes available. Target weights are shifted by
`delay_bars`, whose minimum accepted value is one. The default return assigned to active weights is
the next bar's open-to-close return. For delay `d` and horizon `h`, IC uses
`close[t+d+h-1] / open[t+d] - 1`, matching the configured execution delay. Rank IC ranks only
the common non-missing factor/forward-return pairs in each eligible cross-section.

Quantile membership is determined from eligible factors at signal time, before observing future
return availability. Missing future returns are excluded from group means without reassigning other
assets. Group curves use the first configured IC horizon. When that horizon exceeds one bar,
forward returns overlap; their cumulative curve is descriptive and is not an executable portfolio NAV.
For rank `k` among `n` assets, the group is `ceil(k * groups / n)`, computed with integer-valued
arithmetic so floating-point percentage ranks cannot move boundary assets into the next group.

Bars must form a regular grid at the requested frequency. A missing entire bar raises an error rather
than silently compressing execution delays, horizons and annualization. Missing open/close prices
for held or targeted assets also raise an error; the engine cannot infer delisting settlement prices.
Missing prices for assets with no position or target are allowed.

`start` and `end` filter bars and factors inclusively. The portfolio starts flat and history/liquidity
lookbacks warm up within that window. Universe state is the last update at or before each signal
timestamp, including updates before `start` and between bar timestamps. State before a symbol's
first observation is false; future updates never affect earlier signals.

## Attribution

For every bar:

```text
gap_return     = sum(previous_close_weight * close_to_open_return)
intrabar       = sum(active_weight * open_to_close_return)
funding        = -sum(active_weight * funding_rate)
gross_return   = (1 + gap_return) * (1 + intrabar) - 1
net_return     = (1 + gap_return) * (1 + intrabar + funding - fees - slippage) - 1
```

Outputs retain long-leg, short-leg, funding, fee and slippage contributions separately.
Between rebalances, signed position weights drift with asset prices. At a scheduled rebalance, turnover
is measured against those drifted pre-trade weights rather than the previous target weights.

The benchmark uses market data independently of factor coverage; BTC does not need a factor to
serve as a benchmark. Drawdown includes the initial NAV of 1.0. Sortino uses zero as the minimum
acceptable return and downside deviation `sqrt(mean(min(net_return, 0)^2))` over all periods.
Missing benchmark bars retain the last known close until the next quote, when the full intervening
price change is recognized. The benchmark starts at its first observed close and the report warns
about incomplete coverage; no future prices are backfilled into earlier timestamps.

## Limitations

The engine is intended for factor screening. It does not model order-book depth, partial fills,
contract quantity steps, margin tiers, liquidation, insurance funds or ADL. Funding is modeled at
the supplied settlement rate and weights, not from mark-price cash amounts. Results therefore need
event-driven execution validation before live trading.
