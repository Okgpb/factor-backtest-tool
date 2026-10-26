# Data Contract

All timestamps must be parseable as timezone-aware UTC values. Symbols are uppercase strings such
as `BTCUSDT`; they are never converted to integers.

## Layout

```text
market/
├── 1h/
│   ├── bars.parquet
│   ├── funding.parquet       # optional, strongly recommended
│   └── universe.parquet      # optional, strongly recommended
└── 4h/
    └── ...

factors/
├── 1h.parquet
└── 4h.parquet
```

CSV can be used instead of Parquet. A partitioned Parquet dataset directory named `bars` is also
accepted. Each requested frequency needs its own factor file. The engine never implicitly converts
a high-frequency factor into a low-frequency factor.

## Bars

| Column | Meaning |
| --- | --- |
| `timestamp` | Bar close timestamp in UTC. `close_time` is accepted as an alias. |
| `symbol` | Contract symbol. |
| `open`, `high`, `low`, `close` | Positive numeric prices. |
| `volume` | Base-asset volume. |
| `quote_volume` | Quote-asset notional volume. |

Keys `(timestamp, symbol)` must be unique. A signal stamped at time `t` is assumed to become known
after that bar closes. With the mandatory default delay of one bar, it is first applied to the
`open → close` return of bar `t+1`.

## Factors

Required columns are `timestamp`, `symbol`, and the configured factor column. Higher factor values
are preferred. Use `--reverse` or `reverse: true` when lower values are preferred. Non-finite factor
values are discarded.

## Funding

| Column | Meaning |
| --- | --- |
| `funding_time` | Actual funding settlement time in UTC. |
| `symbol` | Contract symbol. |
| `funding_rate` | Decimal rate, e.g. `0.0001` for one basis point. |

Events are assigned to the first bar ending at or after the settlement time. The portfolio funding
return is `-sum(signed_weight * funding_rate)`: a long pays a positive rate and a short receives it.

## Point-in-time universe

Optional columns are `timestamp`, `symbol`, `tradable`. State is forward-filled after the first
observation. Without this file, the engine uses available bars and emits a survivorship-bias warning.

## Costs and returns

- Target weights are shifted by at least one bar.
- Turnover is `sum(abs(current_weight - previous_weight))`.
- Fee and slippage costs are one-way bps multiplied by turnover.
- Default gross exposure is 1.0: long weights sum to `+0.5`, short weights to `-0.5`.
- Annualization uses a 365-day, 24-hour market rather than 252 equity trading days.
