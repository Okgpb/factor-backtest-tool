from __future__ import annotations

import pandas as pd
import pytest

from crypto_factor_backtest.data import FrequencyData


@pytest.fixture
def small_data() -> FrequencyData:
    timestamps = pd.date_range("2025-01-01 00:00", periods=10, freq="h", tz="UTC")
    symbols = [f"S{i}USDT" for i in range(8)] + ["BTCUSDT"]
    bar_rows = []
    factor_rows = []
    for t_number, timestamp in enumerate(timestamps):
        for s_number, symbol in enumerate(symbols):
            open_price = 100.0 + s_number
            asset_return = (s_number - 4) * 0.001
            bar_rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": open_price,
                    "high": open_price * 1.01,
                    "low": open_price * 0.99,
                    "close": open_price * (1 + asset_return),
                    "volume": 1000.0,
                    "quote_volume": 1_000_000.0,
                }
            )
            factor_rows.append(
                {"timestamp": timestamp, "symbol": symbol, "factor": float(s_number + t_number * 0.01)}
            )
    return FrequencyData(bars=pd.DataFrame(bar_rows), factors=pd.DataFrame(factor_rows))
