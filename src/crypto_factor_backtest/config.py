from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .timeutils import frequency_seconds, utc_timestamp


@dataclass(slots=True)
class PortfolioConfig:
    long_quantile: float = 0.20
    short_quantile: float = 0.20
    groups: int = 5
    gross_exposure: float = 1.0
    min_assets: int = 8
    rebalance_every: int = 1


@dataclass(slots=True)
class ExecutionConfig:
    delay_bars: int = 1
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    include_funding: bool = True


@dataclass(slots=True)
class UniverseConfig:
    min_history_bars: int = 24
    min_quote_volume: float = 0.0
    liquidity_lookback: int = 24


@dataclass(slots=True)
class AnalysisConfig:
    ic_horizons: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 24])
    benchmark_symbol: str = "BTCUSDT"


@dataclass(slots=True)
class BacktestConfig:
    factor_column: str = "factor"
    reverse: bool = False
    frequencies: list[str] = field(default_factory=lambda: ["1h"])
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    start: str | None = None
    end: str | None = None

    def validate(self) -> None:
        p = self.portfolio
        e = self.execution
        u = self.universe
        if not isinstance(self.factor_column, str) or not self.factor_column.strip():
            raise ValueError("factor_column cannot be empty")
        if self.factor_column in {"timestamp", "symbol"}:
            raise ValueError("factor_column cannot be timestamp or symbol")
        if not isinstance(self.frequencies, list) or not self.frequencies:
            raise ValueError("frequencies must be a non-empty list without duplicates")
        for frequency in self.frequencies:
            frequency_seconds(frequency)
        if len({frequency_seconds(f) for f in self.frequencies}) != len(self.frequencies):
            raise ValueError("frequencies must be a non-empty list without duplicates")
        for label, value in {
            "reverse": self.reverse,
            "execution.include_funding": e.include_funding,
        }.items():
            if not isinstance(value, bool):
                raise ValueError(f"{label} must be a boolean")
        for label, value in {
            "portfolio.groups": p.groups,
            "portfolio.min_assets": p.min_assets,
            "portfolio.rebalance_every": p.rebalance_every,
            "execution.delay_bars": e.delay_bars,
            "universe.min_history_bars": u.min_history_bars,
            "universe.liquidity_lookback": u.liquidity_lookback,
        }.items():
            if type(value) is not int:
                raise ValueError(f"{label} must be an integer")
        for label, value in {
            "portfolio.long_quantile": p.long_quantile,
            "portfolio.short_quantile": p.short_quantile,
            "portfolio.gross_exposure": p.gross_exposure,
            "execution.fee_bps": e.fee_bps,
            "execution.slippage_bps": e.slippage_bps,
            "universe.min_quote_volume": u.min_quote_volume,
        }.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{label} must be a finite number")
        if not 0 < p.long_quantile < 1 or not 0 < p.short_quantile < 1:
            raise ValueError("long_quantile and short_quantile must be between 0 and 1")
        if p.long_quantile + p.short_quantile > 1:
            raise ValueError("long_quantile + short_quantile cannot exceed 1")
        if p.groups < 2 or p.min_assets < 2 or p.rebalance_every < 1:
            raise ValueError("groups/min_assets/rebalance_every are invalid")
        if p.gross_exposure <= 0:
            raise ValueError("gross_exposure must be positive")
        if e.delay_bars < 1:
            raise ValueError("delay_bars must be at least 1 to prevent look-ahead")
        if e.fee_bps < 0 or e.slippage_bps < 0:
            raise ValueError("fee_bps and slippage_bps cannot be negative")
        if u.min_history_bars < 1 or u.liquidity_lookback < 1:
            raise ValueError("universe lookbacks must be positive")
        if u.min_quote_volume < 0:
            raise ValueError("universe.min_quote_volume cannot be negative")
        horizons = self.analysis.ic_horizons
        if (
            not isinstance(horizons, list)
            or not horizons
            or any(type(h) is not int or h < 1 for h in horizons)
            or len(set(horizons)) != len(horizons)
        ):
            raise ValueError("IC horizons must be a non-empty list of distinct positive integers")
        if not isinstance(self.analysis.benchmark_symbol, str) or not self.analysis.benchmark_symbol.strip():
            raise ValueError("benchmark_symbol must be a non-empty string")
        start = utc_timestamp(self.start) if self.start is not None else None
        end = utc_timestamp(self.end) if self.end is not None else None
        if start is not None and end is not None and start > end:
            raise ValueError("end must be at or after start")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_config(raw: dict[str, Any]) -> BacktestConfig:
    known = {"factor_column", "reverse", "frequencies", "start", "end"}
    allowed = known | {"portfolio", "execution", "universe", "analysis"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    sections = {
        "portfolio": PortfolioConfig,
        "execution": ExecutionConfig,
        "universe": UniverseConfig,
        "analysis": AnalysisConfig,
    }
    built = {}
    for name, constructor in sections.items():
        values = raw.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f"{name} config must be a mapping")
        try:
            built[name] = constructor(**values)
        except TypeError as exc:
            raise ValueError(f"Invalid {name} config: {exc}") from exc
    cfg = BacktestConfig(
        **{k: raw[k] for k in known if k in raw},
        **built,
    )
    cfg.validate()
    return cfg


def load_config(path: Path | None = None) -> BacktestConfig:
    if path is None:
        cfg = BacktestConfig()
        cfg.validate()
        return cfg
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config: {path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return _build_config(raw)


def apply_overrides(config: BacktestConfig, overrides: dict[str, Any]) -> BacktestConfig:
    raw = config.to_dict()
    for dotted_key, value in overrides.items():
        cursor = raw
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise ValueError(f"Unknown config key: {dotted_key}")
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise ValueError(f"Unknown config key: {dotted_key}")
        cursor[parts[-1]] = value
    return _build_config(raw)
