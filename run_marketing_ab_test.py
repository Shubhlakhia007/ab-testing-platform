# %% [markdown]
# # Marketing A/B Testing Analysis: Kaggle Dataset
#
# This notebook-style script executes an end-to-end A/B test evaluation pipeline
# on the Kaggle "Marketing A/B Testing" dataset without hardcoded column assumptions.
#
# **Pipeline Steps:**
# 1. Read raw CSV & inspect schema, null counts, and row/user duplicates.
# 2. Configurable parameter assignment (group, converted, and user ID columns).
# 3. Data cleaning & data quality logging (saved to `reports/data_quality_log.md`).
# 4. Group mapping confirmation (mapping control to 'A' and treatment to 'B').
# 5. Statistical hypothesis testing (`sample_ratio_check` and `two_prop_test`).
# 6. Plain-English business interpretation and randomisation mechanism notice.

# %%
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Set up project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    clean_data,
    inspect_data,
    map_groups,
    run_ab_analysis,
)

# Candidate paths for the Kaggle Marketing dataset
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_CANDIDATE_PATHS = [
    DATA_DIR / "marketing_AB.csv",
    DATA_DIR / "marketing_ab.csv",
    DATA_DIR / "marketing_ab_testing.csv",
    PROJECT_ROOT / "marketing_AB.csv",
]

# Column name parameters (customize these to match your exact CSV headers)
GROUP_COL = "test group"
CONVERTED_COL = "converted"
USER_ID_COL = "user id"

# Group mapping: control MUST map to 'A', treatment to 'B'
GROUP_MAPPING: Dict[str, str] = {
    "psa": "A",
    "ad": "B",
}

ALPHA = 0.05
LOG_PATH = REPORTS_DIR / "data_quality_log.md"
SYNTHETIC_NOTICE = "SYNTHETIC DATA, NOT THE REAL KAGGLE FILE"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the marketing A/B test runner."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end A/B testing pipeline on a Marketing A/B CSV dataset."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to the Marketing A/B Testing CSV dataset.",
    )
    return parser.parse_args(argv)


def _resolve_csv_path(csv_arg: Optional[Path]) -> Optional[Path]:
    """Return explicit --csv path if given, else search default candidate paths."""
    if csv_arg is not None:
        if not csv_arg.exists():
            raise FileNotFoundError(f"Specified --csv file not found: {csv_arg}")
        return csv_arg

    for candidate in DEFAULT_CANDIDATE_PATHS:
        if candidate.exists():
            return candidate
    return None


def _generate_synthetic_dataset(output_path: Path) -> pd.DataFrame:
    """Generate a representative synthetic dataset matching Kaggle distributions."""
    np.random.seed(42)
    n_total = 588_829
    n_psa = 23_524   # ~4% control
    n_ad = n_total - n_psa  # ~96% treatment

    # Baseline conversion: PSA ~ 1.785%, Ad ~ 2.554%
    conv_psa = np.random.binomial(1, 0.01785, size=n_psa)
    conv_ad = np.random.binomial(1, 0.02554, size=n_ad)

    raw_df = pd.DataFrame({
        "Unnamed: 0": np.arange(n_total),
        USER_ID_COL: np.random.randint(100_000, 999_999, size=n_total),
        GROUP_COL: ["psa"] * n_psa + ["ad"] * n_ad,
        CONVERTED_COL: np.concatenate([conv_psa, conv_ad]),
        "total ads": np.random.poisson(24, size=n_total),
        "most ads": np.random.choice(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            size=n_total,
        ),
    })
    # Add a few intentional dirty values to demonstrate data quality cleaning
    raw_df[CONVERTED_COL] = raw_df[CONVERTED_COL].astype(object)
    raw_df.loc[0, GROUP_COL] = " psa "
    raw_df.loc[1, CONVERTED_COL] = "True"
    raw_df.loc[2, CONVERTED_COL] = "False"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(output_path, index=False)
    return raw_df


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Execute the full marketing A/B test analysis pipeline."""
    args = parse_args(argv)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_file_path = _resolve_csv_path(args.csv)
    used_synthetic = csv_file_path is None

    print("Environment initialized successfully.")
    print(f"Target CSV Path      : {csv_file_path if csv_file_path else 'None found (using synthetic fallback)'}")
    print(f"Configured Group Col : '{GROUP_COL}'")
    print(f"Configured Conv Col  : '{CONVERTED_COL}'")
    print(f"Configured User ID   : '{USER_ID_COL}'")
    print(f"Group Mapping        : {GROUP_MAPPING}")

    if not used_synthetic and csv_file_path is not None:
        print(f"Reading CSV from: {csv_file_path}")
        raw_df = pd.read_csv(csv_file_path, low_memory=False)
    else:
        banner = "=" * 70
        print(f"\n{banner}\nWARNING: {SYNTHETIC_NOTICE}\n{banner}")
        print("Generating representative synthetic dataset matching Kaggle distributions (N=588,829)...")
        synth_path = DATA_DIR / "marketing_AB_synthetic.csv"
        raw_df = _generate_synthetic_dataset(synth_path)
        print(f"Synthetic dataset saved to: {synth_path}")

    print(f"Loaded DataFrame with shape: {raw_df.shape}")

    inspect_data(raw_df, user_id_col=USER_ID_COL, print_output=True)

    cleaned_df, _ = clean_data(
        raw_df,
        group_col=GROUP_COL,
        converted_col=CONVERTED_COL,
        user_id_col=USER_ID_COL,
        log_path=LOG_PATH,
    )
    print(f"Cleaned dataset rows: {len(cleaned_df):,} (from {len(raw_df):,})")
    print(f"Quality audit log written to: {LOG_PATH}")

    mapped_df = map_groups(
        cleaned_df,
        group_col=GROUP_COL,
        group_mapping=GROUP_MAPPING,
        variant_col="variant",
        print_confirmation=True,
    )

    analysis_results = run_ab_analysis(
        mapped_df,
        variant_col="variant",
        converted_col=CONVERTED_COL,
        alpha=ALPHA,
        print_output=True,
    )

    if used_synthetic:
        banner = "=" * 70
        print(f"{banner}\nREMINDER: {SYNTHETIC_NOTICE}\n{banner}")

    print("Analysis script execution finished successfully.")
    return analysis_results


if __name__ == "__main__":
    main()
