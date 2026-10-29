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

OHLCV values must be finite, prices positive, and volumes non-negative. High/low must bound both
open and close. Symbols must be non-empty strings; whitespace is stripped and letters uppercased.
The combined bar timeline must be regularly spaced at the requested frequency. Missing prices for
held or targeted assets are errors rather than zero-return assumptions.

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
Keys `(funding_time, symbol)` must be unique and rates finite. Nanosecond, microsecond and
millisecond datetime storage resolutions are supported without changing event placement.

## Point-in-time universe

The optional universe file requires `timestamp`, `symbol`, `tradable`, with unique `(timestamp, symbol)`
keys. `tradable` accepts booleans, numeric 0/1, or strings true/false, yes/no, 1/0; missing values are
rejected. Updates may fall between bar timestamps. The last known state is carried forward,
including updates before the requested start date. A symbol is ineligible until its first observation.
Without this file, the engine uses available bars and emits a survivorship-bias warning.

## Costs and returns

- Target weights are shifted by at least one bar.
- Turnover is `sum(abs(target_weight - drifted_pretrade_weight))` at scheduled rebalances and zero otherwise.
- Fee and slippage costs are one-way bps multiplied by turnover.
- Default gross exposure is 1.0: long weights sum to `+0.5`, short weights to `-0.5`.
- Annualization uses a 365-day, 24-hour market rather than 252 equity trading days.
- Frequencies use lowercase fixed-duration units `m`, `h`, `d`, `w`; calendar-month `1M` is rejected.
