---
name: crypto-factor-backtest
description: >-
  Run multi-timeframe cross-sectional factor backtests for USDT perpetual futures.
  Use for crypto factor IC, quantile returns, dollar-neutral long-short portfolios,
  funding fees, trading costs, turnover, drawdown, Binance data preparation, and
  HTML reports. Also use for 加密因子回测、永续合约、多空组合、资金费率或多周期扫描.
license: MIT
metadata:
  short-description: Backtest crypto perpetual cross-sectional factors
  repository: factor-backtest-tool
  repository_url: https://github.com/Okgpb/factor-backtest-tool
  creator: Okgpb
---

# Crypto Factor Backtest

Use this skill to evaluate a cross-sectional factor on Binance-style USDT perpetual data.
It ranks string symbols, builds equal-weight top/bottom quantile portfolios, accounts for
fees, slippage and funding, and compares multiple bar frequencies.

## Required inputs

For real research require all of the following:

- `--data-root`: market root with `<frequency>/bars.parquet`; funding is optional but recommended.
- `--factor-root`: one independent `<frequency>.parquet` factor file per requested frequency.
- `--factor-column`: numeric factor column, provided through CLI or YAML config.
- `--frequencies`: one or more frequencies, provided through CLI or YAML config.

Never use `--test-data` for research conclusions. It contains deterministic synthetic data only.

Read `references/data-contract.md` before preparing user data. All timestamps are UTC. A signal at
bar close is traded no earlier than the next bar open (`execution.delay_bars >= 1`). Do not silently
resample one factor file to other frequencies.

## Workflow

1. Validate that the requested market and factor files exist.
2. Validate UTC timestamps, unique `(timestamp, symbol)` keys and positive prices.
3. Run `scripts/run_crypto_factor_backtest.py`.
4. Inspect the printed JSON and verify every `expected_outputs` path.
5. Report warnings, especially missing funding or point-in-time universe data.
6. State factor direction, frequencies, quantiles, gross exposure, fees, slippage and delay.

## Commands

Real run:

```bash
python scripts/run_crypto_factor_backtest.py \
  --config configs/default.yaml \
  --data-root /path/to/market \
  --factor-root /path/to/factors \
  --factor-column my_factor \
  --frequencies 1h 4h 1d
```

Pass `--reverse` when lower factor values are preferred.

Offline pipeline self-test:

```bash
python scripts/run_crypto_factor_backtest.py --test-data
```

Download public Binance USD-M data:

```bash
python scripts/download_binance_usdm.py \
  --output-root /path/to/market \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --frequencies 1h 4h \
  --start 2024-01-01 --end 2025-01-01
```

Auto-discovering only currently active symbols can cause survivorship bias. Prefer a researched
historical symbol list and provide point-in-time `universe.parquet` files.

## Interpretation limits

This is a vectorized factor-research engine, not an exchange matching engine. It does not simulate
order-book depth, partial fills, maintenance-margin tiers, liquidation or ADL. Keep gross exposure
conservative and use an execution simulator before production deployment.
