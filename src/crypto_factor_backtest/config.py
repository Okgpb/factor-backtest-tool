from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
        if not self.factor_column:
            raise ValueError("factor_column cannot be empty")
        if not self.frequencies or len(set(self.frequencies)) != len(self.frequencies):
            raise ValueError("frequencies must be a non-empty list without duplicates")
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
        if any(h < 1 for h in self.analysis.ic_horizons):
            raise ValueError("all IC horizons must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_config(raw: dict[str, Any]) -> BacktestConfig:
    known = {"factor_column", "reverse", "frequencies", "start", "end"}
    allowed = known | {"portfolio", "execution", "universe", "analysis"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    cfg = BacktestConfig(
        **{k: raw[k] for k in known if k in raw},
        portfolio=PortfolioConfig(**raw.get("portfolio", {})),
        execution=ExecutionConfig(**raw.get("execution", {})),
        universe=UniverseConfig(**raw.get("universe", {})),
        analysis=AnalysisConfig(**raw.get("analysis", {})),
    )
    cfg.validate()
    return cfg


def load_config(path: Path | None = None) -> BacktestConfig:
    if path is None:
        cfg = BacktestConfig()
        cfg.validate()
        return cfg
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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
