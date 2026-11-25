"""Tests for run_marketing_ab_test.py CLI arguments and synthetic fallback."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import run_marketing_ab_test as runner


def test_parse_args_csv_flag() -> None:
    """Verify --csv argument is parsed into args.csv."""
    args = runner.parse_args(["--csv", "custom/path.csv"])
    assert args.csv == Path("custom/path.csv")

    default_args = runner.parse_args([])
    assert default_args.csv is None


def test_main_with_explicit_csv(tmp_path: Path, capsys) -> None:
    """Verify main() reads the file passed via --csv without printing synthetic notice."""
    csv_file = tmp_path / "real_marketing.csv"
    df = pd.DataFrame({
        "user id": list(range(1, 41)),
        "test group": ["psa"] * 20 + ["ad"] * 20,
        "converted": [1] * 4 + [0] * 16 + [1] * 8 + [0] * 12,
    })
    df.to_csv(csv_file, index=False)

    runner.REPORTS_DIR = tmp_path / "reports"
    runner.LOG_PATH = runner.REPORTS_DIR / "data_quality_log.md"
    results = runner.main(["--csv", str(csv_file)])

    captured = capsys.readouterr().out
    assert runner.SYNTHETIC_NOTICE not in captured
    assert results["n_a"] == 20
    assert results["n_b"] == 20


def test_main_synthetic_fallback_notice_and_filename(tmp_path: Path, monkeypatch, capsys) -> None:
    """Verify synthetic fallback prints prominent notice at start & end and saves marketing_AB_synthetic.csv."""
    monkeypatch.setattr(runner, "DEFAULT_CANDIDATE_PATHS", [tmp_path / "nonexistent.csv"])
    monkeypatch.setattr(runner, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(runner, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(runner, "LOG_PATH", tmp_path / "reports" / "data_quality_log.md")

    def _small_synth(output_path: Path) -> pd.DataFrame:
        df = pd.DataFrame({
            "user id": list(range(1, 21)),
            "test group": ["psa"] * 10 + ["ad"] * 10,
            "converted": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        })
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    monkeypatch.setattr(runner, "_generate_synthetic_dataset", _small_synth)

    runner.main([])
    captured = capsys.readouterr().out

    assert captured.count(runner.SYNTHETIC_NOTICE) >= 2
    assert (tmp_path / "data" / "marketing_AB_synthetic.csv").exists()
