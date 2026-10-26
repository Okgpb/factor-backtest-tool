from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import BacktestConfig
from .engine import BacktestResult


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot JSON-encode {type(value)!r}")


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _metric_text(value: Any) -> str:
    if isinstance(value, float):
        return "—" if not np.isfinite(value) else f"{value:.6f}"
    return str(value)


def _save_figure(fig: plt.Figure, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _charts(result: BacktestResult, output_dir: Path) -> dict[str, str]:
    images: dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    result.stats[["nav", "benchmark_nav"]].plot(ax=ax)
    ax.set_title(f"NAV — {result.frequency}")
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.25)
    images["nav"] = _save_figure(fig, output_dir / "nav.png")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    result.stats["drawdown"].plot(ax=ax, color="#c0392b")
    ax.fill_between(result.stats.index, result.stats["drawdown"], 0, color="#e74c3c", alpha=0.25)
    ax.set_title("Drawdown")
    ax.grid(alpha=0.25)
    images["drawdown"] = _save_figure(fig, output_dir / "drawdown.png")

    if not result.group_returns.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        (1.0 + result.group_returns.fillna(0.0)).cumprod().plot(ax=ax, cmap="coolwarm")
        ax.set_title("Quantile NAV")
        ax.grid(alpha=0.25)
        images["groups"] = _save_figure(fig, output_dir / "group_returns.png")
    return images


def write_result(result: BacktestResult, output_dir: Path, config: BacktestConfig) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.stats.to_csv(output_dir / "stats.csv")
    result.ic_series.to_csv(output_dir / "ic_series.csv")
    result.ic_summary.to_csv(output_dir / "ic_summary.csv", index=False)
    result.group_returns.to_csv(output_dir / "group_returns.csv")
    result.weights.stack().rename("weight").reset_index().to_parquet(output_dir / "weights.parquet", index=False)
    result.trades.to_parquet(output_dir / "trades.parquet", index=False)
    result.trades.to_csv(output_dir / "trades.csv", index=False)
    result.stats[["turnover"]].to_csv(output_dir / "turnover.csv")
    result.stats[["funding_return"]].to_csv(output_dir / "funding_pnl.csv")
    result.stats[["fee_return", "slippage_return", "funding_return"]].to_csv(
        output_dir / "cost_breakdown.csv"
    )
    payload = _clean_json({"metrics": result.metrics, "warnings": result.warnings})
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    images = _charts(result, output_dir)
    report_path = output_dir / "report.html"
    report_path.write_text(_frequency_html(result, images), encoding="utf-8")
    return report_path


def _frequency_html(result: BacktestResult, images: dict[str, str]) -> str:
    metric_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(_metric_text(value))}</td></tr>"
        for key, value in result.metrics.items()
    )
    warnings = "".join(f"<li>{html.escape(item)}</li>" for item in result.warnings) or "<li>None</li>"
    charts = "".join(
        f'<section><h2>{name.title()}</h2><img src="data:image/png;base64,{encoded}" alt="{name}"></section>'
        for name, encoded in images.items()
    )
    ic_table = result.ic_summary.to_html(index=False, border=0, float_format=lambda x: f"{x:.6f}")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Crypto factor report — {html.escape(result.frequency)}</title>
<style>
body{{font:14px system-ui,sans-serif;margin:0;background:#f5f7fa;color:#17202a}}
main{{max-width:1100px;margin:auto;padding:28px}} section{{background:white;padding:20px;margin:18px 0;border-radius:10px}}
table{{border-collapse:collapse;width:100%}} th,td{{padding:8px;border-bottom:1px solid #e5e7eb;text-align:left}}
img{{max-width:100%;height:auto}} .warning{{border-left:5px solid #f39c12}}
</style></head><body><main><h1>Crypto factor report — {html.escape(result.frequency)}</h1>
<section><h2>Performance</h2><table>{metric_rows}</table></section>
<section><h2>IC summary</h2>{ic_table}</section>
<section class="warning"><h2>Data warnings</h2><ul>{warnings}</ul></section>{charts}
</main></body></html>"""


def write_comparison(results: list[BacktestResult], output_dir: Path) -> tuple[Path, Path]:
    rows = [result.metrics for result in results]
    comparison = pd.DataFrame(rows).set_index("frequency")
    csv_path = output_dir / "frequency_comparison.csv"
    comparison.to_csv(csv_path)
    table = comparison.reset_index().to_html(index=False, border=0, float_format=lambda x: f"{x:.6f}")
    links = "".join(
        f'<li><a href="{html.escape(result.frequency)}/report.html">{html.escape(result.frequency)} report</a></li>'
        for result in results
    )
    report = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Multi-timeframe crypto factor comparison</title><style>
body{{font:14px system-ui,sans-serif;background:#f5f7fa;color:#17202a;margin:0}}main{{max-width:1200px;margin:auto;padding:28px}}
section{{background:white;padding:20px;border-radius:10px;overflow:auto}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:8px;border-bottom:1px solid #e5e7eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}
</style></head><body><main><h1>Multi-timeframe comparison</h1><section>{table}<h2>Reports</h2><ul>{links}</ul></section>
</main></body></html>"""
    report_path = output_dir / "report.html"
    report_path.write_text(report, encoding="utf-8")
    return csv_path, report_path
