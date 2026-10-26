from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

from .config import apply_overrides, load_config
from .runner import default_output_dir, run_batch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _value(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return raw


def _overrides(pairs: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override must be key=value: {pair}")
        key, value = pair.split("=", 1)
        result[key.strip()] = _value(value.strip())
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-timeframe crypto perpetual factor backtests.")
    parser.add_argument("--config", type=Path, help="YAML configuration file.")
    parser.add_argument("--data-root", type=Path, help="Market data root containing one directory per frequency.")
    parser.add_argument("--factor-root", type=Path, help="Directory containing <frequency>.parquet factor files.")
    parser.add_argument("--factor-column", help="Numeric factor column.")
    parser.add_argument("--reverse", action="store_true", help="Treat lower factor values as better.")
    parser.add_argument("--frequencies", nargs="+", help="Frequencies to run, such as 1h 4h 1d.")
    parser.add_argument("--start", help="Optional inclusive UTC start timestamp.")
    parser.add_argument("--end", help="Optional inclusive UTC end timestamp.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--override", action="append", default=[], help="Config override, e.g. execution.fee_bps=5.")
    parser.add_argument("--test-data", action="store_true", help="Use the checked-in deterministic fixture.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = args.config
        if args.test_data and config_path is None:
            config_path = PROJECT_ROOT / "configs" / "default.yaml"
        config = load_config(config_path)
        raw_overrides = _overrides(args.override)
        if args.factor_column:
            raw_overrides["factor_column"] = args.factor_column
        if args.reverse:
            raw_overrides["reverse"] = True
        if args.frequencies:
            raw_overrides["frequencies"] = args.frequencies
        if args.start:
            raw_overrides["start"] = args.start
        if args.end:
            raw_overrides["end"] = args.end
        config = apply_overrides(config, raw_overrides) if raw_overrides else config
        if args.test_data:
            fixture = PROJECT_ROOT / "data" / "test_data"
            data_root = args.data_root or fixture / "market"
            factor_root = args.factor_root or fixture / "factors"
        else:
            if args.data_root is None or args.factor_root is None:
                raise ValueError("--data-root and --factor-root are required unless --test-data is used")
            data_root, factor_root = args.data_root, args.factor_root
        if not data_root.exists() or not factor_root.exists():
            raise FileNotFoundError(f"Data or factor root does not exist: {data_root}, {factor_root}")
        output_dir = (args.output_dir or default_output_dir(PROJECT_ROOT / "output")).resolve()
        summary = run_batch(config, data_root.resolve(), factor_root.resolve(), output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
