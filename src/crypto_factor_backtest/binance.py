from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .timeutils import utc_timestamp

DEFAULT_BASE_URL = "https://fapi.binance.com"
# Fixed-duration intervals supported by both Binance and this engine. Calendar
# months are deliberately excluded because annualization requires a fixed period.
BINANCE_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"}
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


class BinanceUSDMClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self.session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
                if response.status_code in {418, 429}:
                    last_error = RuntimeError(f"HTTP {response.status_code}: rate limited")
                    if attempt == 4:
                        break
                    delay = float(response.headers.get("Retry-After", 2 ** (attempt + 1)))
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                if (
                    isinstance(exc, requests.HTTPError)
                    and exc.response is not None
                    and 400 <= exc.response.status_code < 500
                ):
                    raise
                last_error = exc
                if attempt == 4:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"Binance request failed for {path}: {last_error}")

    def active_usdt_perpetuals(self) -> list[str]:
        payload = self.get("/fapi/v1/exchangeInfo")
        return sorted(
            item["symbol"]
            for item in payload.get("symbols", [])
            if item.get("contractType") == "PERPETUAL"
            and item.get("quoteAsset") == "USDT"
            and item.get("status") == "TRADING"
        )

    def klines(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        if interval not in BINANCE_INTERVALS:
            raise ValueError(f"Unsupported Binance interval: {interval}")
        rows: list[list[Any]] = []
        cursor = start_ms
        while cursor <= end_ms:
            batch = self.get(
                "/fapi/v1/klines",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1500,
                },
            )
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][6]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
        if not rows:
            return pd.DataFrame(columns=["timestamp", "symbol", *KLINE_COLUMNS[1:6], "quote_volume"])
        frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        frame = frame[pd.to_numeric(frame["close_time"], errors="coerce") <= end_ms]
        frame["timestamp"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
        frame["symbol"] = symbol
        numeric = ["open", "high", "low", "close", "volume", "quote_volume"]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
        return frame[["timestamp", "symbol", *numeric]].drop_duplicates(["timestamp", "symbol"])

    def funding_rates(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        cursor = start_ms
        while cursor <= end_ms:
            batch = self.get(
                "/fapi/v1/fundingRate",
                {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000},
            )
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1]["fundingTime"]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < 1000:
                break
        if not rows:
            return pd.DataFrame(columns=["funding_time", "symbol", "funding_rate"])
        frame = pd.DataFrame(rows)
        frame["funding_time"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
        frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
        frame["symbol"] = symbol
        return frame[["funding_time", "symbol", "funding_rate"]].drop_duplicates(["funding_time", "symbol"])


def _timestamp_ms(value: str) -> int:
    timestamp = utc_timestamp(value)
    return int(timestamp.timestamp() * 1000)


def download_dataset(
    output_root: Path,
    symbols: list[str],
    frequencies: list[str],
    start: str,
    end: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    start_ms, end_ms = _timestamp_ms(start), _timestamp_ms(end)
    if end_ms <= start_ms:
        raise ValueError("end must be later than start")
    if not frequencies or len(set(frequencies)) != len(frequencies):
        raise ValueError("frequencies must be non-empty and distinct")
    for frequency in frequencies:
        if frequency not in BINANCE_INTERVALS:
            raise ValueError(f"Unsupported Binance interval: {frequency}")
    client = BinanceUSDMClient(base_url=base_url)
    try:
        return _download_dataset(client, output_root, symbols, frequencies, start, end, start_ms, end_ms)
    finally:
        client.session.close()


def _download_dataset(
    client: BinanceUSDMClient,
    output_root: Path,
    symbols: list[str],
    frequencies: list[str],
    start: str,
    end: str,
    start_ms: int,
    end_ms: int,
) -> dict[str, Any]:
    symbols = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
    if not symbols:
        symbols = client.active_usdt_perpetuals()
    if not symbols:
        raise ValueError("No symbols were supplied or returned by Binance")
    all_funding = []
    for symbol in symbols:
        funding = client.funding_rates(symbol, start_ms, end_ms)
        if not funding.empty:
            all_funding.append(funding)
    funding_frame = (
        pd.concat(all_funding, ignore_index=True)
        if all_funding
        else pd.DataFrame(columns=["funding_time", "symbol", "funding_rate"])
    )
    counts: dict[str, int] = {}
    for frequency in frequencies:
        frames = []
        for symbol in symbols:
            bars = client.klines(symbol, frequency, start_ms, end_ms)
            if not bars.empty:
                frames.append(bars)
        if not frames:
            raise RuntimeError(f"Binance returned no {frequency} bars")
        frequency_dir = output_root / frequency
        frequency_dir.mkdir(parents=True, exist_ok=True)
        bars_frame = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
        bars_frame.to_parquet(frequency_dir / "bars.parquet", index=False)
        funding_frame.to_parquet(frequency_dir / "funding.parquet", index=False)
        counts[frequency] = len(bars_frame)
    manifest = {
        "source": "Binance USD-M Futures public REST API",
        "base_url": client.base_url,
        "start": start,
        "end": end,
        "symbols": symbols,
        "frequencies": frequencies,
        "rows": counts,
        "warning": "Auto-discovered active symbols can introduce survivorship bias.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Binance USD-M perpetual bars and funding rates.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--symbols-file", type=Path)
    parser.add_argument("--frequencies", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = list(args.symbols)
    try:
        if args.symbols_file:
            symbols.extend(
                line.strip() for line in args.symbols_file.read_text(encoding="utf-8").splitlines()
            )
        manifest = download_dataset(
            args.output_root.resolve(),
            symbols,
            args.frequencies,
            args.start,
            args.end,
            base_url=args.base_url,
        )
    except (OSError, RuntimeError, ValueError, requests.RequestException) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
