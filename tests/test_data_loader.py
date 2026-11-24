"""Unit tests for src/data_loader.py.

Validates parameter flexibility, data hygiene logging, mapping confirmation,
and end-to-end integration with the statistical analysis engine.
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
import pandas as pd

from src.data_loader import (
    clean_data,
    inspect_data,
    load_clean_and_test,
    map_groups,
    run_ab_analysis,
    write_data_quality_log,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Fixture providing a DataFrame with intentional duplicates, nulls, and format variants."""
    return pd.DataFrame({
        "client_id": [101, 102, 103, 104, 105, 105, 106, 107, 108],
        "experiment_arm": [" psa ", "ad", "psa", "ad", "ad", "ad", None, "psa", "ad"],
        "has_converted": ["True", "False", True, False, 1, 1, 0, None, "false"],
        "extra_info": ["x", "y", "z", "w", "v", "v", "u", "t", "s"],
    })


def test_inspect_data_summary(sample_raw_df: pd.DataFrame) -> None:
    """Verify inspect_data extracts accurate summary statistics."""
    summary = inspect_data(sample_raw_df, user_id_col="client_id", print_output=False)

    assert summary["row_count"] == 9
    assert "client_id" in summary["columns"]
    assert summary["duplicate_rows"] == 1  # row 4 and row 5 are exact duplicates
    assert summary["duplicate_user_ids"] == 1
    assert summary["null_counts"]["experiment_arm"] == 1
    assert summary["null_counts"]["has_converted"] == 1


def test_inspect_data_invalid_user_col(sample_raw_df: pd.DataFrame) -> None:
    """Verify inspect_data raises KeyError when given a non-existent user column."""
    with pytest.raises(KeyError, match="not found in dataset columns"):
        inspect_data(sample_raw_df, user_id_col="non_existent_column", print_output=False)


def test_clean_data_and_quality_log(sample_raw_df: pd.DataFrame, tmp_path: Path) -> None:
    """Verify clean_data performs all cleaning operations and writes quality log."""
    log_file = tmp_path / "data_quality_log.md"

    cleaned_df, log_records = clean_data(
        sample_raw_df,
        group_col="experiment_arm",
        converted_col="has_converted",
        user_id_col="client_id",
        log_path=log_file,
    )

    # 1. Exact duplicates: 1 duplicate dropped (row 5)
    # 2. Null group dropped (row 6)
    # 3. Null conversion dropped (row 7)
    # Remaining rows: 6
    assert len(cleaned_df) == 6

    # Verify whitespace stripped
    assert all(val in ("psa", "ad") for val in cleaned_df["experiment_arm"].unique())

    # Verify conversions are integer 0 or 1
    assert set(cleaned_df["has_converted"].unique()).issubset({0, 1})

    # Verify log records structure
    assert len(log_records) >= 5
    for record in log_records:
        assert "issue" in record
        assert "count" in record
        assert "action" in record

    # Verify log file written
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "# Data Quality Log" in content
    assert "| Issue | Count | Action |" in content


def test_map_groups_valid(sample_raw_df: pd.DataFrame, tmp_path: Path) -> None:
    """Verify map_groups accurately maps custom names to 'A' and 'B'."""
    cleaned_df, _ = clean_data(
        sample_raw_df,
        group_col="experiment_arm",
        converted_col="has_converted",
        log_path=None,
    )

    mapping = {"psa": "A", "ad": "B"}
    mapped_df = map_groups(
        cleaned_df,
        group_col="experiment_arm",
        group_mapping=mapping,
        variant_col="variant",
        print_confirmation=False,
    )

    assert "variant" in mapped_df.columns
    assert set(mapped_df["variant"].unique()) == {"A", "B"}
    # psa should map to A
    psa_variants = mapped_df[mapped_df["experiment_arm"] == "psa"]["variant"].unique()
    assert list(psa_variants) == ["A"]


def test_map_groups_unmapped_value_error(sample_raw_df: pd.DataFrame) -> None:
    """Verify map_groups raises ValueError when unique labels are not in group_mapping."""
    cleaned_df, _ = clean_data(
        sample_raw_df,
        group_col="experiment_arm",
        converted_col="has_converted",
        log_path=None,
    )

    incomplete_mapping = {"psa": "A"}  # Missing 'ad'
    with pytest.raises(ValueError, match="missing from group_mapping"):
        map_groups(
            cleaned_df,
            group_col="experiment_arm",
            group_mapping=incomplete_mapping,
            print_confirmation=False,
        )


def test_run_ab_analysis_output() -> None:
    """Verify statistical analysis runs and returns plain English interpretation."""
    test_df = pd.DataFrame({
        "variant": ["A"] * 1000 + ["B"] * 1000,
        "converted": [1] * 100 + [0] * 900 + [1] * 140 + [0] * 860,
    })

    results = run_ab_analysis(test_df, variant_col="variant", converted_col="converted", print_output=False)

    assert results["n_a"] == 1000
    assert results["x_a"] == 100
    assert results["n_b"] == 1000
    assert results["x_b"] == 140

    test_res = results["test_result"]
    assert test_res["p_a"] == 0.10
    assert test_res["p_b"] == 0.14
    assert test_res["p_value"] < 0.05

    decision_res = results["decision_result"]
    assert decision_res["decision"] == "SHIP_B"
    assert "Variant B" in decision_res["explanation"]

    # Verify disclaimer is present
    assert "NOT assigned using our deterministic" in results["disclaimer"]
    assert "src/assign.py" in results["disclaimer"]


def test_load_clean_and_test_csv_buffer(tmp_path: Path) -> None:
    """Verify end-to-end load_clean_and_test using an in-memory CSV buffer."""
    csv_data = """custom_id,campaign_group,is_converted
1,control,False
2,treatment,True
3,control,True
4,treatment,False
5,control,0
6,treatment,1
7,control,false
8,treatment,true
"""
    buf = io.StringIO(csv_data)
    log_file = tmp_path / "data_quality_log.md"

    mapped_df, results = load_clean_and_test(
        filepath_or_buffer=buf,
        group_col="campaign_group",
        converted_col="is_converted",
        user_id_col="custom_id",
        group_mapping={"control": "A", "treatment": "B"},
        log_path=log_file,
        print_output=False,
    )

    assert len(mapped_df) == 8
    assert results["n_a"] == 4
    assert results["n_b"] == 4
    assert results["x_a"] == 1
    assert results["x_b"] == 3
    assert log_file.exists()
