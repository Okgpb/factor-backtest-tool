# Crypto Factor Backtest Tool

A multi-timeframe cross-sectional factor research framework for USDT perpetual futures. It provides
rank IC, quantile returns, dollar-neutral long-short portfolios, trading costs, funding attribution,
a BTC benchmark, and self-contained HTML reports.

## Quick check

```bash
python -m pip install -e .
python scripts/run_crypto_factor_backtest.py --test-data
```

The bundled fixture is deterministic synthetic data and is not research evidence.

## User-data run

```bash
python scripts/run_crypto_factor_backtest.py \
  --config configs/default.yaml \
  --data-root /path/to/market \
  --factor-root /path/to/factors \
  --factor-column my_factor \
  --frequencies 1h 4h 1d
```

Higher factor values are preferred by default. Pass `--reverse` for a low-is-good factor.

Each frequency uses its own factor file. See
[`references/data-contract.md`](references/data-contract.md) for schemas and timestamp semantics.

This is a vectorized factor-screening engine. It does not simulate order books, partial fills,
margin tiers or liquidation. Validate promising signals in an execution simulator before trading.

## Development and validation

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
```

Configuration validates boolean/integer types and finite numeric values. The market timeline must
be regularly spaced; held or targeted assets require execution and valuation prices. Date filtering
restarts history/liquidity warm-up within the requested window while preserving the last known
universe state before the start date. See the [methodology](references/backtest-methodology.md)
and [code review record](references/code-review.md) for details.

Licensed under MIT.
