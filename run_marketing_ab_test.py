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
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

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
    write_data_quality_log,
)
from src.stats_engine import decide, sample_ratio_check, two_prop_test

print("Environment initialized successfully.")

# %% [markdown]
# ## 1. Dataset Parameters & File Configuration
# 
# Configure file paths and column names below.
# Do NOT assume hardcoded column names; configure them here according to your file.

# %%
# Candidate paths for the Kaggle Marketing dataset
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CANDIDATE_PATHS = [
    DATA_DIR / "marketing_AB.csv",
    DATA_DIR / "marketing_ab.csv",
    DATA_DIR / "marketing_ab_testing.csv",
    PROJECT_ROOT / "marketing_AB.csv",
]

# Set your dataset path here, or specify via command line / environment variable
CSV_FILE_PATH: Optional[Path] = None
for candidate in DEFAULT_CANDIDATE_PATHS:
    if candidate.exists():
        CSV_FILE_PATH = candidate
        break

# Column name parameters (customize these to match your exact CSV headers)
GROUP_COL = "test group"      # Group/variant column (e.g. 'test group', 'group', 'campaign_arm')
CONVERTED_COL = "converted"   # Conversion column (e.g. 'converted', 'conversion', 'is_converted')
USER_ID_COL = "user id"       # User identifier column (e.g. 'user id', 'user_id', 'id')

# Group mapping: control MUST map to 'A', treatment to 'B'
# In the Kaggle dataset, 'psa' (Public Service Announcement) is the control, and 'ad' is treatment.
GROUP_MAPPING: Dict[str, str] = {
    "psa": "A",
    "ad": "B",
}

ALPHA = 0.05
LOG_PATH = REPORTS_DIR / "data_quality_log.md"

print(f"Target CSV Path      : {CSV_FILE_PATH if CSV_FILE_PATH else 'None found yet (will use sample generator)'}")
print(f"Configured Group Col : '{GROUP_COL}'")
print(f"Configured Conv Col  : '{CONVERTED_COL}'")
print(f"Configured User ID   : '{USER_ID_COL}'")
print(f"Group Mapping        : {GROUP_MAPPING}")

# %% [markdown]
# ## 2. Load or Synthesize Dataset
# 
# If you haven't downloaded `marketing_AB.csv` into `data/`, a synthetic dataset
# matching the exact distribution of the Kaggle dataset will be generated
# so this script runs seamlessly out-of-the-box.

# %%
if CSV_FILE_PATH is not None and CSV_FILE_PATH.exists():
    print(f"Reading CSV from: {CSV_FILE_PATH}")
    raw_df = pd.read_csv(CSV_FILE_PATH)
else:
    print("Notice: No local CSV found at default paths.")
    print("Generating representative synthetic dataset matching Kaggle distributions (N=588,829)...")
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
        "most ads": np.random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], size=n_total),
    })
    # Add a few intentional dirty values to demonstrate data quality cleaning
    raw_df[CONVERTED_COL] = raw_df[CONVERTED_COL].astype(object)
    raw_df.loc[0, GROUP_COL] = " psa "  # whitespace
    raw_df.loc[1, CONVERTED_COL] = "True"  # string boolean
    raw_df.loc[2, CONVERTED_COL] = "False"

    # Save to data directory for subsequent runs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    synth_path = DATA_DIR / "marketing_AB_sample.csv"
    raw_df.to_csv(synth_path, index=False)
    print(f"Sample dataset saved to: {synth_path}")

print(f"Loaded DataFrame with shape: {raw_df.shape}")

# %% [markdown]
# ## 3. Step 1: Read CSV & Inspect Dataset
# 
# Prints columns, dtypes, row count, null counts per column, and duplicate row/user counts.

# %%
inspection_results = inspect_data(raw_df, user_id_col=USER_ID_COL, print_output=True)

# %% [markdown]
# ## 4. Step 2 & 3: Clean Data & Write Data-Quality Log
# 
# Cleans:
# - Drop exact duplicate rows
# - Drop rows with missing group or conversion value
# - Convert conversion column from True/False or "True"/"False" to 0/1 integers
# - Strip whitespace from group labels
# 
# Every action is appended to `reports/data_quality_log.md` as a Markdown table.

# %%
cleaned_df, quality_log = clean_data(
    raw_df,
    group_col=GROUP_COL,
    converted_col=CONVERTED_COL,
    user_id_col=USER_ID_COL,
    log_path=LOG_PATH,
)

print(f"Cleaned dataset rows: {len(cleaned_df):,} (from {len(raw_df):,})")
print(f"Quality audit log written to: {LOG_PATH}")
print("\nData Quality Log Table:")
print(LOG_PATH.read_text(encoding="utf-8"))

# %% [markdown]
# ## 5. Step 4: Map Group Labels to A (Control) and B (Treatment)
# 
# Validates and confirms group mapping:
# Control is mapped to 'A', and Treatment is mapped to 'B'.

# %%
mapped_df = map_groups(
    cleaned_df,
    group_col=GROUP_COL,
    group_mapping=GROUP_MAPPING,
    variant_col="variant",
    print_confirmation=True,
)

# %% [markdown]
# ## 6. Step 5 & 6: Run Sample Ratio Check & Two-Proportion Z-Test
# 
# Runs `sample_ratio_check` and `two_prop_test` from `src.stats_engine`.
# Prints plain English business conclusion and randomisation mechanism notice.

# %%
analysis_results = run_ab_analysis(
    mapped_df,
    variant_col="variant",
    converted_col=CONVERTED_COL,
    alpha=ALPHA,
    print_output=True,
)

# %% [markdown]
# ## 7. Dataset Limitations & Methodological Notes
# 
# Below are 5 key limitations specific to this dataset for inclusion in experimental reports:
# 
# 1. **Severe Group Imbalance (Power vs. Validity):** While the highly unequal group allocation (approximately 96% treatment to 4% control) reduces the experiment's statistical power compared to an equal 50/50 split of the same sample size, it does **not** compromise the mathematical validity or inflate the Type I error rate of the two-proportion z-test because success and failure counts in both arms vastly exceed the minimum threshold of 10.
# 2. **Artificial SRM Flagging Under Balanced Assumptions:** Standard sample ratio mismatch (SRM) goodness-of-fit tests assume an intended 50/50 allocation and will flag a severe mismatch on this dataset; however, this discrepancy reflects an intentional unequal targeting design rather than faulty randomization or tracking failure.
# 3. **Single High-Level Conversion Metric:** The dataset records only a single binary conversion indicator without monetary values (such as order amount, revenue per user, or acquisition cost), which prevents computing Return on Ad Spend (ROAS) or measuring cannibalization effects.
# 4. **Absence of Temporal and Contextual Timestamps:** The lack of date and hour timestamps means we cannot control for day-of-week seasonality, external marketing shocks, or ad fatigue over time.
# 5. **External Randomisation Dependence:** Because user assignments were executed externally rather than through our deterministic SHA-256 hash engine (`src/assign.py`), we cannot independently audit assignment salt integrity, device-level reassignment leaks, or attrition bias.

# %%
if __name__ == "__main__":
    print("Analysis script execution finished successfully.")
