"""A/B Testing & Statistical Simulation Platform.

A multi-page Streamlit application integrating experiment pre-registration,
sample size sizing, stochastic visitor simulation, CSV upload, SQL-based
aggregation, classical hypothesis testing, assumption checks, and Monte Carlo
simulation labs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import repeat
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assign import assign, assign_and_store
from src.data_loader import prepare_csv_for_experiment
from src.db import (
    clear_experiment_data,
    create_experiment,
    get_counts,
    get_experiment,
    get_experiment_by_name,
    init_db,
    list_experiments,
    log_event,
    log_events_batch,
    store_assignments_batch,
    verify_experiment_hash,
)
from src.simulate import aa_test, peeking
from src.stats_engine import (
    decide,
    sample_ratio_check,
    sample_size,
    two_prop_test,
)

# -----------------------------------------------------------------------------
# App Configuration & Page Layout
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="A/B Testing & Statistical Simulation Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Database
conn = init_db()

PAGES = [
    "1. Plan & Sample Size",
    "2. Data & Simulation",
    "3. Statistical Verdict",
    "4. Simulation Labs",
    "5. SHA-256 Demo",
]

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "experiment_id" not in st.session_state:
    st.session_state["experiment_id"] = None
if "is_locked" not in st.session_state:
    st.session_state["is_locked"] = False
if "creating_new" not in st.session_state:
    st.session_state["creating_new"] = False
if "active_page" not in st.session_state or st.session_state["active_page"] not in PAGES:
    st.session_state["active_page"] = PAGES[0]
if "demo_visitor_id" not in st.session_state:
    st.session_state["demo_visitor_id"] = 101
if "demo_logged_views" not in st.session_state:
    st.session_state["demo_logged_views"] = set()
if "loaded_sample_df" not in st.session_state:
    st.session_state["loaded_sample_df"] = None


def navigate_to(page: str) -> None:
    """Switch the active page and rerun."""
    st.session_state["active_page"] = page
    st.rerun()


# -----------------------------------------------------------------------------
# Sidebar: Experiment Context & Selector
# -----------------------------------------------------------------------------
st.sidebar.title("🧪 Experiment Suite")
st.sidebar.caption("Pre-registration, SQLite tracking & statistical verification.")
st.sidebar.markdown("---")

all_exps = list_experiments(conn)

if all_exps:
    exp_options = {f"#{e['id']} - {e['name']}": e["id"] for e in all_exps}
    exp_labels = list(exp_options.keys())
    exp_ids = list(exp_options.values())

    if st.session_state["creating_new"]:
        st.sidebar.info("📝 **Drafting a new experiment** in **1. Plan & Sample Size**.")
        if st.sidebar.button("↩️ Back to Saved Experiments", use_container_width=True):
            st.session_state["creating_new"] = False
            st.session_state["experiment_id"] = exp_ids[0]
            st.session_state["is_locked"] = True
            st.rerun()
    else:
        if st.session_state["experiment_id"] not in exp_ids:
            st.session_state["experiment_id"] = exp_ids[0]
            st.session_state["is_locked"] = True

        current_idx = exp_ids.index(st.session_state["experiment_id"])
        selected_label = st.sidebar.selectbox(
            "Active Experiment",
            options=exp_labels,
            index=current_idx,
        )
        selected_id = exp_options[selected_label]
        if st.session_state["experiment_id"] != selected_id:
            st.session_state["experiment_id"] = selected_id
            st.session_state["is_locked"] = True
            st.rerun()

        if st.sidebar.button("➕ Create New Experiment", use_container_width=True):
            st.session_state["creating_new"] = True
            st.session_state["experiment_id"] = None
            st.session_state["is_locked"] = False
            st.session_state["active_page"] = PAGES[0]
            st.rerun()
else:
    st.session_state["creating_new"] = True
    st.session_state["experiment_id"] = None
    st.session_state["is_locked"] = False
    st.sidebar.info(
        "No saved experiments yet. Fill out **1. Plan & Sample Size** and click "
        "**Lock Hypothesis & Register Experiment** to create your first one."
    )

current_exp: Optional[Dict[str, Any]] = None
if not st.session_state["creating_new"] and st.session_state["experiment_id"] is not None:
    current_exp = get_experiment(conn, st.session_state["experiment_id"])
    if current_exp is not None:
        st.session_state["is_locked"] = True
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Status:** 🔒 Locked")
        st.sidebar.markdown(f"**Name:** `{current_exp['name']}`")
        st.sidebar.markdown(f"**Planned N:** `{current_exp['planned_n_per_group']:,} / group`")
        st.sidebar.markdown(f"**Target MDE:** `{current_exp['mde'] * 100:.2f}%`")
        st.sidebar.markdown(f"**Alpha (α):** `{current_exp['alpha']:.2f}`")
        st.sidebar.markdown(f"**Power (1-β):** `{current_exp['power']:.2f}`")
    else:
        st.session_state["experiment_id"] = None
        st.session_state["is_locked"] = False

# -----------------------------------------------------------------------------
# Header & Top Navigation
# -----------------------------------------------------------------------------
st.title("🧪 A/B Testing & Statistical Simulation Platform")
st.markdown(
    "End-to-end statistical experimentation workflow: **Plan & Pre-Register** → **Simulate or Ingest CSV Data** → "
    "**Evaluate SRM & Hypothesis Verdict**, plus standalone **Monte Carlo Simulation Labs**."
)

active_page = st.segmented_control(
    "Workflow Navigation",
    options=PAGES,
    default=st.session_state["active_page"],
    key="nav_control",
    label_visibility="collapsed",
)
if active_page is not None and active_page != st.session_state["active_page"]:
    st.session_state["active_page"] = active_page
    st.rerun()

page = st.session_state["active_page"]
st.markdown("---")


# =============================================================================
# PAGE 1: PLAN & SAMPLE SIZE
# =============================================================================
if page == PAGES[0]:
    st.header("1. Experiment Planning & Hypothesis Pre-Registration")
    st.markdown(
        "Define your hypothesis and business decision rule before collecting data. "
        "Pre-registering and locking the hypothesis with a SHA-256 hash protects against "
        "p-hacking, HARKing, and early-stopping bias."
    )

    is_locked = bool(st.session_state.get("is_locked", False) and current_exp is not None)

    if is_locked and current_exp:
        st.success(
            f"🔒 **Hypothesis Locked & Pre-Registered:** Experiment `#{current_exp['id']} - {current_exp['name']}` "
            f"was locked at `{current_exp['locked_at']}` UTC."
        )
        st.code(f"Hypothesis SHA-256 Digest:\n{current_exp['hypothesis_hash']}", language="text")
        if verify_experiment_hash(current_exp):
            st.caption("🛡️ **Plan integrity: verified** (stored SHA-256 matches all locked plan parameters)")
        else:
            st.warning("⚠️ **Plan integrity: mismatch** (record was created with an older hash format or modified)")

        col_next1, col_next2 = st.columns([2, 1])
        with col_next1:
            if st.button("➡️ Next: Collect or Simulate Data (Step 2)", type="primary", use_container_width=True):
                navigate_to(PAGES[1])
        with col_next2:
            if st.button("➕ Draft Another Experiment", use_container_width=True):
                st.session_state["creating_new"] = True
                st.session_state["experiment_id"] = None
                st.session_state["is_locked"] = False
                st.rerun()

    col_name, col_rule = st.columns([1, 1])

    default_name = f"ab_experiment_v{len(all_exps) + 1}"
    with col_name:
        exp_name = st.text_input(
            "Experiment Name",
            value=current_exp["name"] if current_exp else default_name,
            disabled=is_locked,
            help="Unique machine-readable identifier for this experiment (used as SHA-256 salt).",
        )

    with col_rule:
        success_rule = st.text_area(
            "Success Rule / Business Decision Criterion",
            value=(
                current_exp["success_rule"]
                if current_exp
                else "Ship Variant B if conversion increases by at least 1 percentage point with p < 0.05 and a positive 95% confidence interval; otherwise retain Variant A."
            ),
            disabled=is_locked,
            help="Example: Ship Variant B if p-value < 0.05 and the 95% confidence interval lower bound > 0.",
            height=100,
        )

    hypothesis_text = st.text_area(
        "Hypothesis Statement",
        value=(
            current_exp["hypothesis"]
            if current_exp
            else "Treatment (Variant B) will produce a statistically significant increase in conversion rate over Control (Variant A)."
        ),
        disabled=is_locked,
        help="State the expected behavioral change and business mechanism.",
        height=90,
    )

    st.subheader("Statistical Power & Sample Size Sizing")
    c_p1, c_lift, c_alpha, c_power = st.columns(4)

    with c_p1:
        baseline_rate = st.number_input(
            "Baseline Rate (p₁)",
            min_value=0.001,
            max_value=0.999,
            value=float(current_exp["baseline_rate"]) if current_exp else 0.10,
            step=0.01,
            format="%.3f",
            disabled=is_locked,
            help="Current baseline conversion rate for control (Variant A).",
        )

    with c_lift:
        lift = st.number_input(
            "Smallest Detectable Lift (MDE)",
            min_value=0.001,
            max_value=0.500,
            value=float(current_exp["mde"]) if current_exp else 0.01,
            step=0.005,
            format="%.3f",
            disabled=is_locked,
            help="Minimum absolute difference |p₂ - p₁| to detect with statistical power.",
        )

    with c_alpha:
        alpha = st.number_input(
            "Significance Level (α)",
            min_value=0.001,
            max_value=0.500,
            value=float(current_exp["alpha"]) if current_exp else 0.05,
            step=0.01,
            format="%.2f",
            disabled=is_locked,
            help="False positive tolerance threshold (Type I error). Typically 0.05.",
        )

    with c_power:
        power = st.number_input(
            "Statistical Power (1 - β)",
            min_value=0.500,
            max_value=0.999,
            value=float(current_exp["power"]) if current_exp else 0.80,
            step=0.05,
            format="%.2f",
            disabled=is_locked,
            help="Probability of detecting the true effect if it exists. Typically 0.80.",
        )

    # Live sample size calculation
    target_rate = min(0.999, max(0.001, baseline_rate + lift))
    try:
        req_sample_size = sample_size(baseline_rate, target_rate, alpha=alpha, power=power)
        total_sample = req_sample_size * 2
        st.info(
            f"📊 **Required Sample Size (Live):** You need about **{req_sample_size:,} users per version** "
            f"({total_sample:,} total across both versions) to detect a shift from {baseline_rate * 100:.1f}% to "
            f"{target_rate * 100:.1f}% at α = {alpha} and Power = {power * 100:.0f}%."
        )
    except Exception as err:
        req_sample_size = 0
        st.error(f"Error sizing sample: {err}")

    st.markdown("---")
    if not is_locked:
        if st.button("🔒 Lock Hypothesis & Register Experiment", type="primary", use_container_width=True):
            clean_name = exp_name.strip()
            if not clean_name:
                st.error("Please enter a valid Experiment Name before locking.")
            elif not hypothesis_text.strip():
                st.error("Please enter a non-empty Hypothesis before locking.")
            elif req_sample_size <= 0:
                st.error("Invalid sample size calculation.")
            elif get_experiment_by_name(conn, clean_name) is not None:
                st.error(
                    f"An experiment named `{clean_name}` already exists. "
                    "Please choose a unique Experiment Name (or select the existing one from the sidebar)."
                )
            else:
                now_str = datetime.now(timezone.utc).isoformat()
                new_id = create_experiment(
                    conn=conn,
                    name=clean_name,
                    hypothesis=hypothesis_text.strip(),
                    success_rule=success_rule.strip(),
                    baseline_rate=float(baseline_rate),
                    mde=float(lift),
                    alpha=float(alpha),
                    power=float(power),
                    planned_n_per_group=int(req_sample_size),
                    locked_at=now_str,
                )
                st.session_state["creating_new"] = False
                st.session_state["experiment_id"] = new_id
                st.session_state["is_locked"] = True
                st.rerun()
    else:
        st.caption("Hypothesis inputs are locked and read-only. Use Step 2 (Data) and Step 3 (Verdict) to run your analysis.")


# =============================================================================
# PAGE 2: DATA & SIMULATION
# =============================================================================
elif page == PAGES[1]:
    st.header("2. Data Collection & Population Ingestion")

    if not st.session_state.get("is_locked") or current_exp is None:
        st.warning(
            "⚠️ **No active locked experiment selected.** "
            "Please create and lock an experiment in **1. Plan & Sample Size** first."
        )
        if st.button("📝 Go to 1. Plan & Sample Size", type="primary"):
            navigate_to(PAGES[0])
    else:
        st.markdown(
            f"Populate visitor observations for locked experiment: "
            f"`#{current_exp['id']} - {current_exp['name']}`."
        )

        data_option = st.radio(
            "Choose Data Source:",
            ["Option A: Simulate Visitors", "Option B: Upload CSV / Sample Dataset"],
            horizontal=True,
        )

        # ---------------------------------------------------------------------
        # OPTION A: SIMULATE VISITORS
        # ---------------------------------------------------------------------
        if data_option == "Option A: Simulate Visitors":
            st.subheader("Option A: Stochastic Visitor Simulation")
            st.markdown(
                "Simulates user traffic through deterministic variant assignment via SHA-256 hashing. "
                "Each visitor logs a `view_checkout` event, and converts with an `order_complete` event "
                "according to their assigned variant's true probability."
            )

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                sim_n_per_version = st.number_input(
                    "Users per Version",
                    min_value=100,
                    max_value=500000,
                    value=int(current_exp["planned_n_per_group"]),
                    step=1000,
                    help="Sample size to target for each variant (A and B).",
                )
            with col_s2:
                sim_true_a = st.number_input(
                    "TRUE Rate for A (Control)",
                    min_value=0.001,
                    max_value=0.999,
                    value=float(current_exp["baseline_rate"]),
                    step=0.01,
                    format="%.3f",
                )
            with col_s3:
                sim_true_b = st.number_input(
                    "TRUE Rate for B (Treatment)",
                    min_value=0.001,
                    max_value=0.999,
                    value=min(0.999, float(current_exp["baseline_rate"]) + float(current_exp["mde"])),
                    step=0.01,
                    format="%.3f",
                )
            with col_s4:
                sim_seed = st.number_input(
                    "Random Seed (RNG)",
                    min_value=0,
                    max_value=9999999,
                    value=42,
                    step=1,
                    help="Seed for reproducible pseudorandom binomial draws.",
                )

            total_sim_users = int(sim_n_per_version * 2)
            st.caption(
                f"Total simulated population: **{total_sim_users:,} visitors** "
                f"({sim_n_per_version:,} per version)."
            )

            col_btn_sim, col_btn_clear = st.columns([3, 1])
            with col_btn_sim:
                run_sim_clicked = st.button("🚀 Run Visitor Simulation", type="primary", use_container_width=True)
            with col_btn_clear:
                if st.button("🗑️ Clear Existing Data", use_container_width=True):
                    clear_experiment_data(conn, current_exp["id"])
                    st.success("Existing visitor data cleared.")
                    st.rerun()

            if run_sim_clicked:
                rng = np.random.default_rng(int(sim_seed))
                exp_id = current_exp["id"]
                exp_name = current_exp["name"]
                now_iso = datetime.now(timezone.utc).isoformat()

                clear_experiment_data(conn, exp_id)

                progress_bar = st.progress(0.0)
                status_placeholder = st.empty()

                chunk_size = 5000 if total_sim_users > 20000 else total_sim_users
                total_processed = 0

                for start_idx in range(1, total_sim_users + 1, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_sim_users + 1)
                    chunk_uids = list(range(start_idx, end_idx))
                    chunk_len = len(chunk_uids)

                    if total_sim_users > 20000:
                        variants = [assign(uid, exp_name) for uid in chunk_uids]
                        assignments = [
                            (chunk_uids[i], exp_id, variants[i], now_iso)
                            for i in range(chunk_len)
                        ]
                        store_assignments_batch(conn, assignments)

                        view_events = [
                            (chunk_uids[i], exp_id, "view_checkout", now_iso)
                            for i in range(chunk_len)
                        ]
                        log_events_batch(conn, view_events)

                        draws = rng.random(chunk_len)
                        order_events = [
                            (chunk_uids[i], exp_id, "order_complete", now_iso)
                            for i in range(chunk_len)
                            if draws[i] < (sim_true_b if variants[i] == "B" else sim_true_a)
                        ]
                        log_events_batch(conn, order_events)
                    else:
                        for uid in chunk_uids:
                            var = assign_and_store(conn, uid, exp_id, exp_name)
                            log_event(conn, uid, exp_id, "view_checkout", now_iso)
                            prob = sim_true_b if var == "B" else sim_true_a
                            if rng.random() < prob:
                                log_event(conn, uid, exp_id, "order_complete", now_iso)

                    total_processed += chunk_len
                    progress_val = min(1.0, total_processed / float(total_sim_users))
                    progress_bar.progress(progress_val)
                    status_placeholder.text(f"Simulating visitors: {total_processed:,} / {total_sim_users:,}...")

                progress_bar.progress(1.0)
                status_placeholder.empty()

                counts = get_counts(conn, exp_id)
                st.success(
                    f"🎉 Successfully simulated **{counts['total_users']:,} visitors** into the database! "
                    f"Variant A: {counts['n_a']:,} visitors, {counts['x_a']:,} orders ({counts['conv_a'] * 100:.2f}%). "
                    f"Variant B: {counts['n_b']:,} visitors, {counts['x_b']:,} orders ({counts['conv_b'] * 100:.2f}%)."
                )

            counts_now = get_counts(conn, current_exp["id"])
            if counts_now["total_users"] > 0:
                if st.button("➡️ Next: View Statistical Verdict (Step 3)", type="primary", use_container_width=True):
                    navigate_to(PAGES[2])

        # ---------------------------------------------------------------------
        # OPTION B: UPLOAD CSV / SAMPLE DATASET
        # ---------------------------------------------------------------------
        else:
            st.subheader("Option B: Upload External Dataset (CSV) or Load Sample")
            st.markdown(
                "Upload a CSV file containing experiment records, or load the project's built-in "
                "sample dataset (`data/marketing_AB_synthetic.csv` or `data/marketing_AB_sample.csv`)."
            )

            synth_csv_path = PROJECT_ROOT / "data" / "marketing_AB_synthetic.csv"
            legacy_sample_path = PROJECT_ROOT / "data" / "marketing_AB_sample.csv"
            sample_csv_path = synth_csv_path if synth_csv_path.exists() else legacy_sample_path
            col_up1, col_up2 = st.columns([2, 1])
            with col_up1:
                uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
            with col_up2:
                if sample_csv_path.exists():
                    st.write("&nbsp;")
                    if st.button("📂 Load Built-in Sample CSV", use_container_width=True):
                        with st.spinner(f"Loading {sample_csv_path.name}..."):
                            st.session_state["loaded_sample_df"] = pd.read_csv(sample_csv_path, low_memory=False)

            df: Optional[pd.DataFrame] = None
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, low_memory=False)
                st.session_state["loaded_sample_df"] = None
            elif st.session_state.get("loaded_sample_df") is not None:
                df = st.session_state["loaded_sample_df"]

            if df is not None:
                try:
                    st.markdown(f"**Loaded Preview:** `{df.shape[0]:,}` rows × `{df.shape[1]}` columns")
                    st.dataframe(df.head(5), use_container_width=True)

                    cols_list = list(df.columns)
                    default_grp_idx = cols_list.index("test group") if "test group" in cols_list else 0
                    default_conv_idx = cols_list.index("converted") if "converted" in cols_list else min(1, len(cols_list) - 1)

                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        group_col = st.selectbox(
                            "Select Group / Variant Column",
                            options=cols_list,
                            index=default_grp_idx,
                        )
                    with col_m2:
                        conv_col = st.selectbox(
                            "Select Converted / Outcome Column",
                            options=cols_list,
                            index=default_conv_idx,
                        )
                    with col_m3:
                        uid_options = ["<Use row number>"] + cols_list
                        selected_uid_opt = st.selectbox(
                            "User ID Column (optional)",
                            options=uid_options,
                            index=0,
                            help="If chosen, duplicate user IDs are dropped (keeping first).",
                        )
                        user_id_col = None if selected_uid_opt == "<Use row number>" else selected_uid_opt

                    # Strip whitespace and ignore blank labels before building group choices.
                    raw_groups = df[group_col].dropna().astype(str).str.strip()
                    cleaned_groups = [g for g in raw_groups.unique().tolist() if g != ""]
                    default_a_idx = cleaned_groups.index("psa") if "psa" in cleaned_groups else 0
                    default_b_idx = cleaned_groups.index("ad") if "ad" in cleaned_groups else (1 if len(cleaned_groups) > 1 else 0)

                    st.markdown("##### Value Mapping")
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        val_a = st.selectbox(
                            "Value representing Variant A (Control)",
                            options=cleaned_groups,
                            index=default_a_idx,
                        )
                    with col_v2:
                        val_b = st.selectbox(
                            "Value representing Variant B (Treatment)",
                            options=cleaned_groups,
                            index=default_b_idx,
                        )

                    clear_before_import = st.checkbox(
                        "Clear existing experiment data before importing",
                        value=True,
                    )

                    if st.button("📥 Ingest CSV Data into Experiment", type="primary", use_container_width=True):
                        exp_id = current_exp["id"]
                        if clear_before_import:
                            clear_experiment_data(conn, exp_id)

                        with st.spinner("Cleaning and ingesting CSV rows..."):
                            now_iso = datetime.now(timezone.utc).isoformat()
                            uids, variant_arr, conv_bool, ingest_stats = prepare_csv_for_experiment(
                                df=df,
                                group_col=group_col,
                                conv_col=conv_col,
                                val_a=val_a,
                                val_b=val_b,
                                user_id_col=user_id_col,
                            )

                            # Vectorized tuple list construction via zip + repeat for 588k+ row speed.
                            uids_list = uids.tolist()
                            variants_list = variant_arr.tolist()
                            assignments = list(zip(uids_list, repeat(exp_id), variants_list, repeat(now_iso)))
                            view_events = list(zip(uids_list, repeat(exp_id), repeat("view_checkout"), repeat(now_iso)))
                            order_uids = uids[conv_bool].tolist()
                            order_events = list(zip(order_uids, repeat(exp_id), repeat("order_complete"), repeat(now_iso)))

                            store_assignments_batch(conn, assignments)
                            log_events_batch(conn, view_events)
                            log_events_batch(conn, order_events)

                        counts = get_counts(conn, exp_id)
                        st.success(
                            f"🎉 Ingested **{counts['total_users']:,} records** from CSV "
                            f"(out of {ingest_stats['total_rows']:,} rows)! "
                            f"Variant A: {counts['n_a']:,} users ({counts['x_a']:,} conversions). "
                            f"Variant B: {counts['n_b']:,} users ({counts['x_b']:,} conversions)."
                        )

                        skipped_parts = []
                        if ingest_stats["skipped_blank"] > 0:
                            skipped_parts.append(f"{ingest_stats['skipped_blank']:,} blank/missing values")
                        if ingest_stats["skipped_other_groups"] > 0:
                            skipped_parts.append(f"{ingest_stats['skipped_other_groups']:,} other group labels")
                        if ingest_stats["skipped_unparseable"] > 0:
                            skipped_parts.append(f"{ingest_stats['skipped_unparseable']:,} unparseable outcomes")
                        if ingest_stats["dropped_duplicate_users"] > 0:
                            skipped_parts.append(f"{ingest_stats['dropped_duplicate_users']:,} duplicate user IDs")

                        if skipped_parts:
                            total_skipped = (
                                ingest_stats["skipped_blank"]
                                + ingest_stats["skipped_other_groups"]
                                + ingest_stats["skipped_unparseable"]
                                + ingest_stats["dropped_duplicate_users"]
                            )
                            st.info(
                                f"ℹ️ **Skipped / Dropped {total_skipped:,} rows:** " + ", ".join(skipped_parts) + "."
                            )

                    counts_now = get_counts(conn, current_exp["id"])
                    if counts_now["total_users"] > 0:
                        if st.button("➡️ Next: View Statistical Verdict (Step 3)", type="primary", use_container_width=True):
                            navigate_to(PAGES[2])
                except Exception as ex:
                    st.error(f"Error parsing dataset: {ex}")


# =============================================================================
# PAGE 3: STATISTICAL VERDICT
# =============================================================================
elif page == PAGES[2]:
    st.header("3. Statistical Verdict & Decision Card")

    if not st.session_state.get("is_locked") or current_exp is None:
        st.warning(
            "⚠️ **No active locked experiment selected.** "
            "Please create and lock an experiment in **1. Plan & Sample Size**."
        )
        if st.button("📝 Go to 1. Plan & Sample Size", type="primary"):
            navigate_to(PAGES[0])
    else:
        exp_id = current_exp["id"]
        counts = get_counts(conn, exp_id)
        x_a = counts["x_a"]
        n_a = counts["n_a"]
        x_b = counts["x_b"]
        n_b = counts["n_b"]
        total_users = counts["total_users"]

        if total_users == 0:
            st.info(
                "ℹ️ **No visitor data logged yet for this experiment.** "
                "Simulate visitor traffic or upload a CSV in **2. Data & Simulation** to compute the verdict."
            )
            if st.button("📊 Go to 2. Data & Simulation", type="primary"):
                navigate_to(PAGES[1])
        else:
            exp_alpha = float(current_exp["alpha"])
            planned_n = int(current_exp["planned_n_per_group"])

            test_res = two_prop_test(x_a, n_a, x_b, n_b, alpha=exp_alpha)
            srm_res = sample_ratio_check(n_a, n_b)
            decide_res = decide(test_res, planned_n_per_group=planned_n, n_a=n_a, n_b=n_b)

            # 1. Summary Metrics
            st.subheader("Traffic & Conversion Summary (SQL Aggregation)")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)

            with col_m1:
                st.metric(
                    label="Variant A (Control)",
                    value=f"{test_res['p_a'] * 100:.2f}%",
                    help=f"{x_a:,} conversions / {n_a:,} visitors",
                )
            with col_m2:
                st.metric(
                    label="Variant B (Treatment)",
                    value=f"{test_res['p_b'] * 100:.2f}%",
                    delta=f"{test_res['diff'] * 100:+.2f}%",
                    help=f"{x_b:,} conversions / {n_b:,} visitors",
                )
            with col_m3:
                st.metric(
                    label="Absolute Difference (Lift)",
                    value=f"{test_res['diff'] * 100:+.2f} pp",
                    help="p_B - p_A in percentage points",
                )
            with col_m4:
                st.metric(
                    label="Relative Lift",
                    value=f"{test_res['relative_lift'] * 100:+.2f}%",
                    help="(p_B - p_A) / p_A",
                )

            st.markdown("---")

            # 2. Key Statistical Tests
            st.subheader("Classical Hypothesis Test Results")
            c_p, c_ci, c_h = st.columns(3)

            with c_p:
                p_display = (
                    f"{test_res['p_value']:.4f}"
                    if test_res["p_value"] >= 0.0001
                    else f"{test_res['p_value']:.2e} (< 0.0001)"
                )
                st.metric(
                    label="p-value (Two-Tailed)",
                    value=p_display,
                    delta="Significant" if test_res["p_value"] < exp_alpha else "Not Significant",
                    delta_color="normal" if test_res["p_value"] < exp_alpha else "off",
                )
                with st.expander("❓ What does this p-value mean?"):
                    st.markdown(
                        "The **p-value** measures the probability of observing a conversion difference "
                        "at least as large as the one seen, assuming the true difference between variants is zero (null hypothesis). "
                        "\n\n**Common Pitfall:** The p-value is **NOT** the probability that Variant B is better than Variant A."
                    )

            with c_ci:
                ci_str = f"[{test_res['ci_low'] * 100:+.2f}%, {test_res['ci_high'] * 100:+.2f}%]"
                st.metric(
                    label=f"{(1.0 - exp_alpha) * 100:.0f}% Confidence Interval",
                    value=ci_str,
                    help="Unpooled Wald Confidence Interval for (p_B - p_A)",
                )
                with st.expander("❓ What does this interval mean?"):
                    st.markdown(
                        f"We are **{(1.0 - exp_alpha) * 100:.0f}% confident** that the true population lift "
                        f"lies between **{test_res['ci_low'] * 100:+.2f}%** and **{test_res['ci_high'] * 100:+.2f}%**."
                    )

            with c_h:
                h_val = test_res["cohens_h"]
                if abs(h_val) < 0.2:
                    h_desc = "Small effect size"
                elif abs(h_val) < 0.5:
                    h_desc = "Medium effect size"
                else:
                    h_desc = "Large effect size"

                st.metric(
                    label="Effect Size (Cohen's h)",
                    value=f"{h_val:.4f}",
                    help=f"Standardized effect magnitude: {h_desc}",
                )
                with st.expander("❓ What does Cohen's h mean?"):
                    st.markdown(
                        "**Cohen's h** measures the standardized magnitude of the difference between two proportions, "
                        "independent of sample size."
                    )

            # 3. Confidence Interval Chart
            st.subheader("Confidence Interval Visualization")
            fig, ax = plt.subplots(figsize=(8.5, 2.2), dpi=200)

            diff_pct = test_res["diff"] * 100.0
            ci_low_pct = test_res["ci_low"] * 100.0
            ci_high_pct = test_res["ci_high"] * 100.0

            ci_color = "#16A34A" if ci_low_pct > 0 else ("#DC2626" if ci_high_pct < 0 else "#4B5563")

            ax.errorbar(
                [diff_pct],
                [0],
                xerr=[[diff_pct - ci_low_pct], [ci_high_pct - diff_pct]],
                fmt="o",
                color=ci_color,
                ecolor=ci_color,
                elinewidth=3,
                capsize=6,
                capthick=2,
                markersize=8,
                label=f"Observed Lift: {diff_pct:+.2f}%",
            )

            ax.axvline(
                0.0,
                color="#EF4444",
                linestyle="--",
                linewidth=1.8,
                label="Zero Effect Line (p_B = p_A)",
            )

            margin = max(abs(ci_high_pct - ci_low_pct) * 0.4, 0.5)
            x_min = min(-0.5, ci_low_pct - margin)
            x_max = max(0.5, ci_high_pct + margin)
            ax.set_xlim(x_min, x_max)
            ax.set_yticks([])
            ax.set_xlabel("Difference in Conversion Rate (Percentage Points %)", fontsize=10)
            ax.set_title(
                f"95% Confidence Interval for Lift: [{ci_low_pct:+.2f}%, {ci_high_pct:+.2f}%]",
                fontsize=11,
                fontweight="bold",
            )
            ax.grid(axis="x", linestyle=":", alpha=0.6)
            ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # 4. Assumption Checks and SRM Badges
            st.subheader("Pre-Flight Assumption Checks")
            col_b1, col_b2 = st.columns(2)

            with col_b1:
                if not srm_res["srm_detected"]:
                    st.success(
                        f"✅ **Sample Ratio Check Passed (50/50 Allocation OK)**\n\n"
                        f"No selection bias or tracking dropouts detected "
                        f"(χ² = {srm_res['chi2']:.2f}, p = {srm_res['p_value']:.4f}). "
                        f"Observed: {n_a:,} vs {n_b:,}."
                    )
                else:
                    st.error(
                        f"🚨 **Sample Ratio Mismatch (SRM) Detected!**\n\n"
                        f"Severe allocation imbalance between variants "
                        f"(χ² = {srm_res['chi2']:.2f}, p = {srm_res['p_value']:.2e} < 0.001). "
                        f"Observed: {n_a:,} vs {n_b:,}. Test results may be compromised!"
                    )

            with col_b2:
                if test_res["assumptions_ok"]:
                    st.success(
                        "✅ **Normality Assumptions Satisfied**\n\n"
                        "Success-failure condition verified: at least 10 conversions and 10 non-conversions "
                        "in each variant. Central Limit Theorem normal approximation is valid."
                    )
                else:
                    st.error(
                        "🚨 **Normality Assumptions Violated**\n\n"
                        + "\n".join(f"- {note}" for note in test_res["assumption_notes"])
                    )

            # 5. Early-Stopping Warning
            min_observed_n = min(n_a, n_b)
            if min_observed_n < planned_n:
                st.warning(
                    f"⚠️ **EARLY-STOPPING WARNING (PEEKING PROBLEM)**\n\n"
                    f"Observed **{min_observed_n:,}** vs. **{planned_n:,} planned per version** "
                    f"({(min_observed_n / float(planned_n)) * 100:.1f}% of target reached). "
                    f"Stopping early without an alpha-spending function inflates false-positive rates."
                )

            # 6. Decision Card
            st.subheader("Business Decision & Recommendation")
            dec_code = decide_res["decision"]

            if dec_code == "SHIP_B":
                st.success(
                    f"### 🚀 Decision: SHIP VARIANT B\n\n"
                    f"**Explanation:** {decide_res['explanation']}"
                )
            elif dec_code == "KEEP_A":
                st.error(
                    f"### 🛑 Decision: KEEP VARIANT A (DISCARD B)\n\n"
                    f"**Explanation:** {decide_res['explanation']}"
                )
            else:
                st.info(
                    f"### ⚖️ Decision: INCONCLUSIVE (NO WINNER)\n\n"
                    f"**Explanation:** {decide_res['explanation']}"
                )


# =============================================================================
# PAGE 4: SIMULATION LABS
# =============================================================================
elif page == PAGES[3]:
    st.header("4. Monte Carlo Simulation Labs")
    st.markdown(
        "Standalone interactive Monte Carlo simulations verifying the mathematical foundations of A/B testing "
        "(no locked experiment required): **A/A p-value calibration** and **optional-stopping α inflation**."
    )

    tab_aa, tab_peek = st.tabs(["A/A Test Calibration", "The Peeking Problem"])

    with tab_aa:
        st.subheader("A/A Test: P-Value Uniformity under H₀")
        st.markdown(
            "When testing two identical variants (p_A = p_B), the null hypothesis is true. "
            "Under H₀, p-values must follow a **Uniform(0, 1)** distribution, with ~5% falling below α = 0.05."
        )

        col_aa1, col_aa2, col_aa3 = st.columns(3)
        with col_aa1:
            aa_tests_count = st.slider("Number of A/A Experiments", min_value=100, max_value=2000, value=300, step=100)
        with col_aa2:
            aa_sample_size = st.slider("Sample Size per Variant (N)", min_value=1000, max_value=20000, value=5000, step=1000)
        with col_aa3:
            aa_base_rate = st.slider("Baseline Rate (p)", min_value=0.01, max_value=0.50, value=0.10, step=0.01)

        if st.button("🧪 Run A/A Simulation", type="primary", key="btn_run_aa"):
            with st.spinner("Running Monte Carlo A/A tests..."):
                aa_res = aa_test(
                    n_tests=aa_tests_count,
                    n=aa_sample_size,
                    p=aa_base_rate,
                    alpha=0.05,
                    seed=42,
                )

            p_vals = aa_res["p_values"]
            frac_sig = aa_res["fraction_significant"]

            st.write(
                f"**Simulation Output:** Observed {aa_res['significant_count']} false positives out of {aa_tests_count} "
                f"({frac_sig * 100:.1f}%), within the expected 95% binomial range of "
                f"[{aa_res['binomial_range_95'][0]}, {aa_res['binomial_range_95'][1]}]."
            )

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
            bins = np.linspace(0.0, 1.0, 21)
            counts_hist, edges, patches = ax.hist(
                p_vals,
                bins=bins,
                color="#3B82F6",
                edgecolor="#1E3A8A",
                linewidth=0.8,
                alpha=0.85,
            )

            for patch, left, right in zip(patches, edges[:-1], edges[1:]):
                if right <= 0.05:
                    patch.set_facecolor("#EF4444")
                    patch.set_edgecolor("#991B1B")

            expected_bin = len(p_vals) / 20.0
            ax.axhline(
                expected_bin,
                color="#111827",
                linestyle="--",
                linewidth=1.5,
                label=f"Expected Uniform ({int(expected_bin)} / bin)",
            )
            ax.axvline(
                0.05,
                color="#DC2626",
                linestyle=":",
                linewidth=1.8,
                label=f"α = 0.05 (Observed: {frac_sig * 100:.1f}%)",
            )

            ax.set_title("A/A Test P-Value Distribution (Uniform Under H₀)", fontsize=11, fontweight="bold")
            ax.set_xlabel("p-value", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_xlim(0.0, 1.0)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab_peek:
        st.subheader("The Peeking Problem: False Positive Inflation")
        st.markdown(
            "Continuously checking p-values as visitors arrive and stopping when p < 0.05 "
            "drastically inflates the Type I error rate beyond the nominal 5% threshold."
        )

        col_pk1, col_pk2, col_pk3 = st.columns(3)
        with col_pk1:
            pk_tests = st.slider("Simulated Experiments", min_value=100, max_value=1000, value=250, step=50)
        with col_pk2:
            pk_days = st.slider("Evaluation Days / Peeks", min_value=5, max_value=21, value=12, step=1)
        with col_pk3:
            pk_daily_n = st.slider("Daily Visitors per Variant", min_value=200, max_value=3000, value=1000, step=200)

        if st.button("📈 Run Peeking Simulation", type="primary", key="btn_run_peeking"):
            with st.spinner("Simulating multi-day optional stopping..."):
                peek_res = peeking(
                    n_tests=pk_tests,
                    per_day=pk_daily_n,
                    days=pk_days,
                    p=0.10,
                    alpha=0.05,
                    seed=42,
                )

            peeks = peek_res["peeks"]
            rates = [r * 100.0 for r in peek_res["cumulative_fpr"]]
            final_fpr = peek_res["false_positive_rate"] * 100.0

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
            ax.plot(
                peeks,
                rates,
                marker="o",
                color="#DC2626",
                linewidth=2.2,
                markersize=6,
                label=f"Cumulative False-Positive Rate (Ends at {final_fpr:.1f}%)",
            )
            ax.axhline(
                5.0,
                color="#1E3A8A",
                linestyle="--",
                linewidth=1.8,
                label="Nominal Significance Level α = 5%",
            )
            ax.fill_between(
                peeks,
                [5.0] * len(peeks),
                rates,
                color="#F87171",
                alpha=0.2,
                label="Type I Error Inflation Region",
            )

            ax.set_title("False-Positive Rate vs. Number of Peeks (Optional Stopping)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Number of Peeks (Daily Statistical Evaluations)", fontsize=10)
            ax.set_ylabel("Cumulative False-Positive Rate (%)", fontsize=10)
            ax.set_xticks(peeks)
            ax.set_ylim(0, max(rates) * 1.25)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# =============================================================================
# PAGE 5: SHA-256 DEMO
# =============================================================================
elif page == PAGES[4]:
    st.header("5. Live SHA-256 Variant Assignment Demo")

    if not st.session_state.get("is_locked") or current_exp is None:
        st.warning(
            "⚠️ **No active locked experiment selected.** "
            "Please lock an experiment in **1. Plan & Sample Size** to test live user assignments."
        )
        if st.button("📝 Go to 1. Plan & Sample Size", type="primary"):
            navigate_to(PAGES[0])
    else:
        exp_id = current_exp["id"]
        exp_name = current_exp["name"]

        st.markdown(
            "Test deterministic user assignment in real-time. Enter a Visitor ID or generate a random one. "
            "The platform hashes `{experiment_name}-{user_id}` via SHA-256 to deterministically assign "
            "Variant A or B."
        )

        # Default OFF so browsing the demo tab does not pollute real experiment data.
        record_demo = st.checkbox(
            "Record demo visitors in this experiment's data",
            value=False,
            help="When off, only previews which variant the visitor would receive without writing to the database.",
        )

        col_id_input, col_id_rand = st.columns([3, 1])
        with col_id_input:
            visitor_id_input = st.number_input(
                "Enter Visitor ID (Integer)",
                min_value=1,
                max_value=99999999,
                value=int(st.session_state["demo_visitor_id"]),
                step=1,
            )
            st.session_state["demo_visitor_id"] = visitor_id_input

        with col_id_rand:
            st.write("&nbsp;")
            if st.button("🎲 Random Visitor ID", use_container_width=True):
                st.session_state["demo_visitor_id"] = random.randint(1000, 99999)
                st.rerun()

        active_uid = int(st.session_state["demo_visitor_id"])
        assigned_variant = assign(active_uid, exp_name)

        st.markdown("---")
        st.subheader(f"Visitor Preview: User #{active_uid} → **Variant {assigned_variant}**")

        # When recording is enabled, log view_checkout once per (experiment, visitor) per session.
        if record_demo:
            visit_key = (exp_id, active_uid)
            if visit_key not in st.session_state["demo_logged_views"]:
                now_iso = datetime.now(timezone.utc).isoformat()
                assign_and_store(conn, active_uid, exp_id, exp_name)
                log_event(conn, active_uid, exp_id, "view_checkout", now_iso)
                st.session_state["demo_logged_views"].add(visit_key)
                st.caption(f"Recorded `view_checkout` for User #{active_uid} in Variant {assigned_variant}.")
            else:
                st.caption(f"Visitor #{active_uid} already recorded in this session (no duplicate `view_checkout` logged).")
        else:
            st.caption("Preview mode only — database recording is OFF.")

        if assigned_variant == "A":
            with st.container(border=True):
                st.markdown("### 🛒 Standard Checkout (Variant A - Control)")
                st.markdown("Traditional two-step checkout with standard credit card fields.")
                st.text_input("Card Number", value="•••• •••• •••• 4242", disabled=True)
                c_exp, c_cvc = st.columns(2)
                with c_exp:
                    st.text_input("Expiration", value="12/28", disabled=True)
                with c_cvc:
                    st.text_input("CVC", value="123", disabled=True)

                if st.button("Complete Purchase ($49.00)", type="secondary", key="btn_order_a", disabled=not record_demo):
                    now_iso = datetime.now(timezone.utc).isoformat()
                    visit_key = (exp_id, active_uid)
                    if visit_key not in st.session_state["demo_logged_views"]:
                        assign_and_store(conn, active_uid, exp_id, exp_name)
                        log_event(conn, active_uid, exp_id, "view_checkout", now_iso)
                        st.session_state["demo_logged_views"].add(visit_key)
                    log_event(conn, active_uid, exp_id, "order_complete", now_iso)
                    st.balloons()
                    st.success(f"🎉 Order Completed! Recorded conversion event for User #{active_uid} in Variant A.")
        else:
            with st.container(border=True):
                st.markdown("### ⚡ Express 1-Click Checkout (Variant B - Treatment)")
                st.markdown("Optimized one-click checkout with buyer guarantees and trusted payment badges.")
                st.markdown(
                    "🔒 **256-Bit SSL Encrypted** | ⭐ **4.9/5 Rating (12k reviews)** | 🚚 **Free Express Delivery**"
                )
                if st.button("⚡ Instant 1-Click Purchase ($49.00)", type="primary", key="btn_order_b", disabled=not record_demo):
                    now_iso = datetime.now(timezone.utc).isoformat()
                    visit_key = (exp_id, active_uid)
                    if visit_key not in st.session_state["demo_logged_views"]:
                        assign_and_store(conn, active_uid, exp_id, exp_name)
                        log_event(conn, active_uid, exp_id, "view_checkout", now_iso)
                        st.session_state["demo_logged_views"].add(visit_key)
                    log_event(conn, active_uid, exp_id, "order_complete", now_iso)
                    st.balloons()
                    st.success(f"🎉 Order Completed! Recorded conversion event for User #{active_uid} in Variant B.")

        st.markdown("---")
        st.subheader("📊 Live Experiment Tracking Table")
        live_counts = get_counts(conn, exp_id)

        live_df = pd.DataFrame([
            {
                "Variant": "A (Control)",
                "Visitors (N)": live_counts["n_a"],
                "Orders (X)": live_counts["x_a"],
                "Conversion Rate": f"{live_counts['conv_a'] * 100:.2f}%",
            },
            {
                "Variant": "B (Treatment)",
                "Visitors (N)": live_counts["n_b"],
                "Orders (X)": live_counts["x_b"],
                "Conversion Rate": f"{live_counts['conv_b'] * 100:.2f}%",
            },
        ])
        st.dataframe(live_df, use_container_width=True)
