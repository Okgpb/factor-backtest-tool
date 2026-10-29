from __future__ import annotations

import json
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from crypto_factor_backtest.binance import BinanceUSDMClient, download_dataset, main


def _response(status, payload):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode()
    return response


@pytest.mark.parametrize(
    ("frequencies", "start", "end"),
    [
        (["2m"], "2025-01-01", "2025-01-02"),
        (["1M"], "2025-01-01", "2025-01-02"),
        ([], "2025-01-01", "2025-01-02"),
        (["1h", "1h"], "2025-01-01", "2025-01-02"),
        (["1h"], "2025-01-02", "2025-01-01"),
        (["1h"], "NaT", "2025-01-02"),
    ],
)
def test_invalid_download_arguments_fail_before_network(monkeypatch, tmp_path, frequencies, start, end):
    get = Mock(side_effect=AssertionError("Unexpected network request"))
    monkeypatch.setattr(requests.Session, "get", get)
    with pytest.raises(ValueError):
        download_dataset(tmp_path, [], frequencies, start, end)
    get.assert_not_called()


def test_permanent_http_errors_are_not_retried(monkeypatch):
    client = BinanceUSDMClient()
    get = Mock(return_value=_response(400, {"code": -1121, "msg": "Invalid symbol"}))
    monkeypatch.setattr(client.session, "get", get)
    with pytest.raises(requests.HTTPError):
        client.get("/fapi/v1/klines")
    assert get.call_count == 1
    client.session.close()


def test_rate_limit_exhaustion_reports_http_status(monkeypatch):
    client = BinanceUSDMClient()
    response = _response(429, {})
    response.headers["Retry-After"] = "0"
    get = Mock(return_value=response)
    monkeypatch.setattr(client.session, "get", get)
    monkeypatch.setattr("crypto_factor_backtest.binance.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        client.get("/fapi/v1/klines")
    assert get.call_count == 5
    client.session.close()


def test_transient_errors_retry_and_can_recover(monkeypatch):
    client = BinanceUSDMClient()
    get = Mock(side_effect=[_response(503, {}), _response(200, {"ok": True})])
    monkeypatch.setattr(client.session, "get", get)
    monkeypatch.setattr("crypto_factor_backtest.binance.time.sleep", lambda _: None)
    assert client.get("/fapi/v1/klines") == {"ok": True}
    assert get.call_count == 2
    client.session.close()


def test_klines_paginate_and_exclude_unfinished_bar(monkeypatch):
    client = BinanceUSDMClient()

    def row(open_time):
        return [open_time, "100", "102", "99", "101", "10", open_time + 59_999, "1000", 1, "5", "500", "0"]

    get = Mock(side_effect=[[row(0), row(60_000)], [row(120_000)], []])
    monkeypatch.setattr(client, "get", get)
    bars = client.klines("BTCUSDT", "1m", 0, 130_000)
    assert len(bars) == 2
    assert get.call_args_list[1].args[1]["startTime"] == 120_000
    assert bars["timestamp"].max() == pd.Timestamp(119_999, unit="ms", tz="UTC")
    client.session.close()


@pytest.mark.parametrize("fail", [False, True])
def test_download_closes_session_on_success_and_failure(monkeypatch, tmp_path, fail):
    client = BinanceUSDMClient()
    close = Mock()
    monkeypatch.setattr(client.session, "close", close)
    monkeypatch.setattr("crypto_factor_backtest.binance.BinanceUSDMClient", lambda **_: client)
    row = {
        "timestamp": pd.Timestamp("2025-01-01T01:00Z"),
        "symbol": "BTCUSDT",
        "open": 1,
        "high": 1,
        "low": 1,
        "close": 1,
        "volume": 1,
        "quote_volume": 1,
    }
    monkeypatch.setattr(client, "funding_rates", Mock(return_value=pd.DataFrame()))
    monkeypatch.setattr(
        client,
        "klines",
        Mock(side_effect=RuntimeError("network failed")) if fail else Mock(return_value=pd.DataFrame([row])),
    )
    if fail:
        with pytest.raises(RuntimeError, match="network failed"):
            download_dataset(tmp_path, ["BTCUSDT"], ["1h"], "2025-01-01", "2025-01-02")
    else:
        manifest = download_dataset(tmp_path, ["BTCUSDT"], ["1h"], "2025-01-01", "2025-01-02")
        assert manifest["rows"] == {"1h": 1}
        assert (tmp_path / "1h" / "bars.parquet").exists()
    close.assert_called_once()


def test_missing_symbols_file_produces_cli_error(tmp_path, capsys):
    code = main(
        [
            "--output-root",
            str(tmp_path),
            "--symbols-file",
            str(tmp_path / "missing.txt"),
            "--frequencies",
            "1h",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-02",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "ERROR:" in captured.err
