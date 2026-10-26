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
the next bar's open-to-close return. IC uses the next open as entry and a future close as exit.

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

## Limitations

The engine is intended for factor screening. It does not model order-book depth, partial fills,
contract quantity steps, margin tiers, liquidation, insurance funds or ADL. Funding is modeled at
the supplied settlement rate and weights, not from mark-price cash amounts. Results therefore need
event-driven execution validation before live trading.
