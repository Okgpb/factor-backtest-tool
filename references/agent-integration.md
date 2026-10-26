# Agent Integration

Run the deterministic offline check from the repository root:

```bash
python scripts/run_crypto_factor_backtest.py --test-data
```

Success requires exit code zero and every path in printed `expected_outputs` to exist. Test fixture
results verify only the software pipeline and must never be presented as investment evidence.

For a user run, require `--data-root`, `--factor-root`, factor column and frequencies. Missing
funding data is accepted with a warning. Missing point-in-time universe data is accepted with a
survivorship-bias warning.
