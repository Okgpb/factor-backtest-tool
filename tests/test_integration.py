from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_fixture_pipeline(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_crypto_factor_backtest.py"),
        "--test-data",
        "--output-dir",
        str(tmp_path / "output"),
    ]
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    output = Path(payload["output_dir"])
    for relative in payload["expected_outputs"]:
        assert (output / relative).exists(), relative
    original_summary = (output / "run_summary.json").read_bytes()
    repeated = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert repeated.returncode == 2
    assert "Output directory must be empty" in repeated.stderr
    assert not repeated.stdout
    assert (output / "run_summary.json").read_bytes() == original_summary
