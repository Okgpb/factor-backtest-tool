from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import BacktestConfig
from .data import load_frequency_data
from .engine import BacktestResult, run_backtest
from .report import write_comparison, write_result
from .timeutils import frequency_seconds


def default_output_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate = root / f"run_{stamp}"
    number = 1
    while candidate.exists():
        candidate = root / f"run_{stamp}_{number:02d}"
        number += 1
    return candidate


def run_batch(
    config: BacktestConfig,
    data_root: Path,
    factor_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config.validate()
    for frequency in config.frequencies:
        frequency_seconds(frequency)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[BacktestResult] = []
    reports: dict[str, str] = {}
    for frequency in config.frequencies:
        dataset = load_frequency_data(
            data_root,
            factor_root,
            frequency,
            config.factor_column,
            start=config.start,
            end=config.end,
        )
        result = run_backtest(dataset, config, frequency)
        result_dir = output_dir / frequency
        reports[frequency] = str(write_result(result, result_dir, config))
        results.append(result)
    comparison_csv, report_html = write_comparison(results, output_dir)
    summary: dict[str, Any] = {
        "data_root": str(data_root.resolve()),
        "factor_root": str(factor_root.resolve()),
        "factor_column": config.factor_column,
        "frequencies": config.frequencies,
        "output_dir": str(output_dir.resolve()),
        "reports": reports,
        "comparison_csv": str(comparison_csv.resolve()),
        "report_html": str(report_html.resolve()),
        "expected_outputs": [
            "run_summary.json",
            "frequency_comparison.csv",
            "report.html",
            *[f"{frequency}/summary.json" for frequency in config.frequencies],
            *[f"{frequency}/stats.csv" for frequency in config.frequencies],
            *[f"{frequency}/ic_series.csv" for frequency in config.frequencies],
            *[f"{frequency}/group_returns.csv" for frequency in config.frequencies],
            *[f"{frequency}/weights.parquet" for frequency in config.frequencies],
            *[f"{frequency}/trades.csv" for frequency in config.frequencies],
            *[f"{frequency}/funding_pnl.csv" for frequency in config.frequencies],
            *[f"{frequency}/nav.png" for frequency in config.frequencies],
            *[f"{frequency}/report.html" for frequency in config.frequencies],
        ],
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    missing = [name for name in summary["expected_outputs"] if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Backtest finished with missing outputs: {missing}")
    return summary
