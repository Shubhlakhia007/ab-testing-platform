"""Data loading, inspection, cleaning, and A/B test pipeline for marketing datasets.

Designed for flexible analysis of external datasets (such as the Kaggle Marketing A/B Testing
dataset) without assuming fixed column names. Provides data quality logging, group mapping,
and integration with src.stats_engine.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Add project root to sys.path if not already present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stats_engine import decide, sample_ratio_check, two_prop_test


def inspect_data(
    df: pd.DataFrame,
    user_id_col: Optional[str] = None,
    print_output: bool = True,
) -> Dict[str, Any]:
    """Inspect raw dataset properties and display key data hygiene metrics.

    Prints:
        - List of column names
        - Column data types (dtypes)
        - Total row count
        - Null value counts per column
        - Number of exact duplicate rows
        - Number of duplicate user IDs (if user_id_col is specified)

    Args:
        df: Input pandas DataFrame.
        user_id_col: Optional column name representing user/customer ID.
        print_output: Whether to print summary to stdout (default: True).

    Returns:
        Dictionary containing extracted metadata and inspection statistics.
    """
    row_count = len(df)
    columns = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    null_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    duplicate_rows = int(df.duplicated().sum())

    duplicate_user_ids: Optional[int] = None
    if user_id_col is not None:
        if user_id_col in df.columns:
            duplicate_user_ids = int(df[user_id_col].duplicated().sum())
        else:
            raise KeyError(
                f"User ID column '{user_id_col}' not found in dataset columns: {columns}"
            )

    info_summary: Dict[str, Any] = {
        "row_count": row_count,
        "columns": columns,
        "dtypes": dtypes,
        "null_counts": null_counts,
        "duplicate_rows": duplicate_rows,
        "duplicate_user_ids": duplicate_user_ids,
    }

    if print_output:
        sep = "=" * 70
        print(sep)
        print("DATASET INSPECTION REPORT")
        print(sep)
        print(f"Total Rows    : {row_count:,}")
        print(f"Total Columns : {len(columns)}")
        print("\nColumns & Data Types:")
        for col in columns:
            null_count = null_counts[col]
            null_pct = (null_count / row_count * 100.0) if row_count > 0 else 0.0
            print(f"  - {col:<25} | {dtypes[col]:<12} | Nulls: {null_count:>6,} ({null_pct:>5.1f}%)")

        print("\nDuplicates:")
        dup_row_pct = (duplicate_rows / row_count * 100.0) if row_count > 0 else 0.0
        print(f"  - Exact Duplicate Rows : {duplicate_rows:,} ({dup_row_pct:.2f}%)")

        if user_id_col is not None and duplicate_user_ids is not None:
            dup_u_pct = (duplicate_user_ids / row_count * 100.0) if row_count > 0 else 0.0
            print(f"  - Duplicate '{user_id_col}' IDs : {duplicate_user_ids:,} ({dup_u_pct:.2f}%)")
        print(sep + "\n")

    return info_summary


def _parse_conversion_value(value: Any) -> int:
    """Normalize various boolean/string/numeric conversion values to integer 0 or 1.

    Supports:
        - Booleans: True -> 1, False -> 0
        - Numbers: 1/1.0 -> 1, 0/0.0 -> 0
        - Strings: "true", "t", "yes", "y", "1" -> 1
                   "false", "f", "no", "n", "0" -> 0

    Raises:
        ValueError: If value cannot be deterministically mapped to 0 or 1.
    """
    if pd.isna(value):
        raise ValueError("Cannot convert NaN or None value to binary conversion indicator.")

    if isinstance(value, bool):
        return 1 if value else 0

    if isinstance(value, (int, float)):
        if value == 1:
            return 1
        if value == 0:
            return 0

    s = str(value).strip().lower()
    if s in ("true", "t", "yes", "y", "1", "1.0"):
        return 1
    if s in ("false", "f", "no", "n", "0", "0.0"):
        return 0

    raise ValueError(
        f"Unable to parse conversion value '{value}' (type: {type(value).__name__}) to 0/1."
    )


def write_data_quality_log(
    log_records: List[Dict[str, Any]],
    output_path: Union[str, Path] = "reports/data_quality_log.md",
) -> str:
    """Write data quality issues and actions to a Markdown table file.

    Args:
        log_records: List of dictionaries with keys 'issue', 'count', and 'action'.
        output_path: Filepath where the Markdown table will be written.

    Returns:
        Formatted Markdown table string.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    header = "# Data Quality Log\n\n"
    intro = "Audit log documenting all data cleaning and standardization operations applied to the dataset.\n\n"
    table_header = "| Issue | Count | Action |\n|:---|---:|:---|\n"

    rows = []
    for entry in log_records:
        issue = str(entry.get("issue", "N/A")).replace("|", "\\|")
        count = entry.get("count", 0)
        action = str(entry.get("action", "N/A")).replace("|", "\\|")
        count_str = f"{count:,}" if isinstance(count, (int, float)) else str(count)
        rows.append(f"| {issue} | {count_str} | {action} |")

    content = header + intro + table_header + "\n".join(rows) + "\n"

    target.write_text(content, encoding="utf-8")
    return content


def clean_data(
    df: pd.DataFrame,
    group_col: str,
    converted_col: str,
    user_id_col: Optional[str] = None,
    log_path: Optional[Union[str, Path]] = "reports/data_quality_log.md",
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Clean dataset according to data quality specifications and record audit log.

    Operations performed:
        1. Drop exact duplicate rows.
        2. Drop rows with missing group column value.
        3. Drop rows with missing conversion column value.
        4. Strip leading/trailing whitespace from group labels.
        5. Convert conversion column from boolean/text to 0/1 integers.
        6. Check for duplicate user IDs (audited in log).

    Every action is recorded in a data-quality log (`issue`, `count`, `action`)
    and written to `reports/data_quality_log.md` as a Markdown table.

    Args:
        df: Input raw DataFrame.
        group_col: Column containing experiment group labels.
        converted_col: Column indicating conversion status.
        user_id_col: Optional column containing user IDs.
        log_path: Path to write markdown data quality log (or None to skip disk write).

    Returns:
        Tuple of (cleaned_df, log_records).

    Raises:
        KeyError: If group_col or converted_col are missing from df.
    """
    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not found in DataFrame columns: {list(df.columns)}")
    if converted_col not in df.columns:
        raise KeyError(
            f"Converted column '{converted_col}' not found in DataFrame columns: {list(df.columns)}"
        )

    log_records: List[Dict[str, Any]] = []
    cleaned = df.copy()

    # 1. Drop exact duplicate rows
    initial_rows = len(cleaned)
    dup_rows = int(cleaned.duplicated().sum())
    if dup_rows > 0:
        cleaned = cleaned.drop_duplicates()
        log_records.append({
            "issue": "Exact duplicate rows",
            "count": dup_rows,
            "action": f"Dropped {dup_rows:,} exact duplicate rows",
        })
    else:
        log_records.append({
            "issue": "Exact duplicate rows",
            "count": 0,
            "action": "No exact duplicate rows found; no rows dropped",
        })

    # 2. Drop rows with missing group value
    missing_group = int(cleaned[group_col].isna().sum())
    if missing_group > 0:
        cleaned = cleaned.dropna(subset=[group_col])
        log_records.append({
            "issue": f"Missing values in group column '{group_col}'",
            "count": missing_group,
            "action": f"Dropped {missing_group:,} rows with missing group label",
        })
    else:
        log_records.append({
            "issue": f"Missing values in group column '{group_col}'",
            "count": 0,
            "action": "No missing group values detected; no rows dropped",
        })

    # 3. Drop rows with missing conversion value
    missing_conv = int(cleaned[converted_col].isna().sum())
    if missing_conv > 0:
        cleaned = cleaned.dropna(subset=[converted_col])
        log_records.append({
            "issue": f"Missing values in conversion column '{converted_col}'",
            "count": missing_conv,
            "action": f"Dropped {missing_conv:,} rows with missing conversion status",
        })
    else:
        log_records.append({
            "issue": f"Missing values in conversion column '{converted_col}'",
            "count": 0,
            "action": "No missing conversion values detected; no rows dropped",
        })

    # 4. Strip whitespace from group labels
    orig_series = cleaned[group_col].astype(str)
    stripped_series = orig_series.str.strip()
    ws_count = int((orig_series != stripped_series).sum())
    cleaned[group_col] = stripped_series
    if ws_count > 0:
        log_records.append({
            "issue": f"Whitespace in group column '{group_col}'",
            "count": ws_count,
            "action": f"Stripped leading/trailing whitespace from {ws_count:,} values",
        })
    else:
        log_records.append({
            "issue": f"Whitespace in group column '{group_col}'",
            "count": 0,
            "action": "No whitespace padding found in group labels",
        })

    # 5. Convert conversion column to 0/1 integers
    # Count how many non-int rows or standardize all
    cleaned[converted_col] = cleaned[converted_col].apply(_parse_conversion_value).astype(int)
    log_records.append({
        "issue": f"Conversion format standardization for '{converted_col}'",
        "count": len(cleaned),
        "action": "Normalized boolean/text representations to binary integers (0 and 1)",
    })

    # 6. Audit duplicate user IDs if specified
    if user_id_col is not None and user_id_col in cleaned.columns:
        dup_users = int(cleaned[user_id_col].duplicated().sum())
        if dup_users > 0:
            log_records.append({
                "issue": f"Duplicate user IDs in '{user_id_col}'",
                "count": dup_users,
                "action": f"Identified {dup_users:,} duplicate user ID rows (retained for user-level aggregation if needed)",
            })
        else:
            log_records.append({
                "issue": f"Duplicate user IDs in '{user_id_col}'",
                "count": 0,
                "action": "All user IDs are strictly unique",
            })

    # Write log to disk if path given
    if log_path is not None:
        write_data_quality_log(log_records, log_path)

    return cleaned, log_records


def map_groups(
    df: pd.DataFrame,
    group_col: str,
    group_mapping: Dict[str, str],
    variant_col: str = "variant",
    print_confirmation: bool = True,
) -> pd.DataFrame:
    """Map arbitrary dataset group labels to standardized experiment variants 'A' and 'B'.

    Control group must map to 'A' and treatment group to 'B'.

    Args:
        df: Input DataFrame.
        group_col: Column with source group labels.
        group_mapping: Dictionary mapping source label string -> 'A' or 'B'.
                       Example: {"psa": "A", "ad": "B"} or {"control": "A", "treatment": "B"}.
        variant_col: Destination column name to store mapped 'A'/'B' values (default: 'variant').
        print_confirmation: Whether to print confirmation of mapping to stdout.

    Returns:
        DataFrame with new variant_col assigned.

    Raises:
        KeyError: If group_col is missing from df.
        ValueError: If group_mapping does not map to 'A' and 'B', or unmapped values exist.
    """
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in DataFrame.")

    # Validate mapping target values
    mapped_targets = set(group_mapping.values())
    if not (mapped_targets == {"A", "B"} or mapped_targets.issubset({"A", "B"})):
        raise ValueError(
            f"group_mapping values must only map to 'A' (control) and 'B' (treatment). Found: {mapped_targets}"
        )
    if "A" not in mapped_targets:
        raise ValueError("group_mapping must map at least one label to 'A' (control group).")

    unique_vals = set(df[group_col].unique())
    unmapped = unique_vals - set(group_mapping.keys())
    if unmapped:
        raise ValueError(
            f"The following unique values in '{group_col}' are missing from group_mapping: {unmapped}. "
            f"Provided mapping keys: {list(group_mapping.keys())}"
        )

    out_df = df.copy()
    out_df[variant_col] = out_df[group_col].map(group_mapping)

    if print_confirmation:
        sep = "=" * 70
        print(sep)
        print("GROUP MAPPING CONFIRMATION")
        print(sep)
        print(f"Source Column : '{group_col}'")
        print(f"Target Column : '{variant_col}'")
        print("Mapping Definition:")
        for raw_val, mapped_val in group_mapping.items():
            role = "Control (A)" if mapped_val == "A" else "Treatment (B)"
            count = int((out_df[variant_col] == mapped_val).sum())
            print(f"  - '{raw_val}' -> '{mapped_val}' [{role}] (Sample size: {count:,})")
        print(sep + "\n")

    return out_df


def run_ab_analysis(
    df: pd.DataFrame,
    variant_col: str = "variant",
    converted_col: str = "converted",
    alpha: float = 0.05,
    planned_n_per_group: Optional[int] = None,
    print_output: bool = True,
) -> Dict[str, Any]:
    """Execute sample ratio check and two-proportion z-test, printing plain English results.

    Integrates with `src.stats_engine.sample_ratio_check`, `two_prop_test`, and `decide`.
    Includes an explicit notice that user groups were not assigned by the project's SHA-256 hash.

    Args:
        df: DataFrame containing variant_col and converted_col.
        variant_col: Column containing 'A' and 'B' group indicators.
        converted_col: Column containing binary conversion values (0 or 1).
        alpha: Significance level for two-proportion z-test (default: 0.05).
        planned_n_per_group: Optional planned sample size for early stopping checks.
        print_output: Whether to print summary to stdout.

    Returns:
        Dictionary containing sample counts, test results, SRM results, and decisions.
    """
    variants = set(df[variant_col].unique())
    if not ({"A", "B"}.issubset(variants) or variants == {"A", "B"}):
        raise ValueError(
            f"variant_col '{variant_col}' must contain both 'A' and 'B'. Found: {variants}"
        )

    # Compute observed counts
    group_a = df[df[variant_col] == "A"]
    group_b = df[df[variant_col] == "B"]

    n_a = int(len(group_a))
    x_a = int(group_a[converted_col].sum())
    n_b = int(len(group_b))
    x_b = int(group_b[converted_col].sum())

    # Run statistical checks
    srm_result = sample_ratio_check(n_a, n_b)
    test_result = two_prop_test(x_a, n_a, x_b, n_b, alpha=alpha)
    decision_result = decide(test_result, planned_n_per_group, n_a, n_b)

    p_a = test_result["p_a"]
    p_b = test_result["p_b"]
    diff = test_result["diff"]
    relative_lift = test_result["relative_lift"]
    z_stat = test_result["z"]
    p_val = test_result["p_value"]
    ci_low = test_result["ci_low"]
    ci_high = test_result["ci_high"]

    explanation = decision_result["explanation"]
    decision = decision_result["decision"]

    # Important notice string regarding dataset randomization
    disclaimer = (
        "IMPORTANT NOTICE ON RANDOMISATION MECHANISM:\n"
        "The experimental groups in this dataset were NOT assigned using our deterministic\n"
        "SHA-256 hash function (src/assign.py). Instead, we rely entirely on the dataset's\n"
        "own external randomisation procedure. Because allocation in this marketing dataset\n"
        "was deliberately unequal (or subject to external ad server delivery), sample ratio\n"
        "checks evaluated against a theoretical 50/50 split should be interpreted in the context\n"
        "of the original experiment's intentional allocation design."
    )

    if print_output:
        sep = "=" * 70
        print(sep)
        print("MARKETING A/B TEST: STATISTICAL ANALYSIS RESULTS")
        print(sep)
        print(f"Variant A (Control)   : {x_a:,} conversions / {n_a:,} users (CR = {p_a:.4%})")
        print(f"Variant B (Treatment) : {x_b:,} conversions / {n_b:,} users (CR = {p_b:.4%})")
        print("-" * 70)
        print(f"Absolute Difference   : {diff:+.4%} ({p_b - p_a:+.6f})")
        print(f"Relative Lift         : {relative_lift:+.2%}")
        print(f"{int((1 - alpha) * 100)}% Confidence Interval : [{ci_low:+.4%}, {ci_high:+.4%}]")
        print(f"Z-Score               : {z_stat:.4f}")
        print(f"P-Value               : {p_val:.6e}")
        print(f"Statistically Sig.    : {'YES (p < ' + str(alpha) + ')' if p_val < alpha else 'NO'}")
        print("-" * 70)
        print(f"Sample Ratio Check (50/50 test):")
        print(f"  Observed: A = {n_a:,} ({n_a / (n_a + n_b):.2%}), B = {n_b:,} ({n_b / (n_a + n_b):.2%})")
        print(f"  Chi-Square = {srm_result['chi2']:.2f}, p-value = {srm_result['p_value']:.4e}")
        print(f"  SRM Flag   = {srm_result['srm_detected']} ({srm_result['message']})")
        print("-" * 70)
        print("PLAIN ENGLISH INTERPRETATION:")
        print(f"  Decision    : {decision}")
        print(f"  Explanation : {explanation}")
        print("-" * 70)
        print(disclaimer)
        print(sep + "\n")

    return {
        "n_a": n_a,
        "x_a": x_a,
        "n_b": n_b,
        "x_b": x_b,
        "test_result": test_result,
        "srm_result": srm_result,
        "decision_result": decision_result,
        "disclaimer": disclaimer,
    }


def load_clean_and_test(
    filepath_or_buffer: Any,
    group_col: str,
    converted_col: str,
    user_id_col: Optional[str] = None,
    group_mapping: Optional[Dict[str, str]] = None,
    alpha: float = 0.05,
    log_path: Optional[Union[str, Path]] = "reports/data_quality_log.md",
    print_output: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """End-to-end convenience loader: reads CSV, inspects, cleans, maps, and runs test.

    Args:
        filepath_or_buffer: CSV file path, URL, or file-like buffer.
        group_col: Name of column containing group labels.
        converted_col: Name of column containing conversion indicators.
        user_id_col: Optional name of column containing user IDs.
        group_mapping: Mapping dict, e.g. {"psa": "A", "ad": "B"} (control must map to 'A').
                       If None, defaults to mapping unique values alphabetically.
        alpha: Significance level for statistical test (default: 0.05).
        log_path: Path to output data quality Markdown table.
        print_output: Whether to print progress and reports to stdout.

    Returns:
        Tuple of (cleaned_mapped_df, test_results_dict).
    """
    if print_output:
        print(f"Loading dataset from: {filepath_or_buffer}")

    df = pd.read_csv(filepath_or_buffer)

    # 1. Inspect
    inspect_data(df, user_id_col=user_id_col, print_output=print_output)

    # 2. Clean
    cleaned_df, _ = clean_data(
        df,
        group_col=group_col,
        converted_col=converted_col,
        user_id_col=user_id_col,
        log_path=log_path,
    )

    # 3. Determine or apply group mapping
    if group_mapping is None:
        unique_groups = sorted(cleaned_df[group_col].unique())
        if len(unique_groups) != 2:
            raise ValueError(
                f"Cannot auto-map groups: Expected exactly 2 unique group labels in '{group_col}', "
                f"found {len(unique_groups)}: {unique_groups}. Please pass an explicit group_mapping."
            )
        group_mapping = {unique_groups[0]: "A", unique_groups[1]: "B"}
        if print_output:
            print(f"No group_mapping provided; defaulting alphabetically: {group_mapping}")

    mapped_df = map_groups(
        cleaned_df,
        group_col=group_col,
        group_mapping=group_mapping,
        variant_col="variant",
        print_confirmation=print_output,
    )

    # 4. Statistical Analysis
    results = run_ab_analysis(
        mapped_df,
        variant_col="variant",
        converted_col=converted_col,
        alpha=alpha,
        print_output=print_output,
    )

    return mapped_df, results
