"""A/B Testing Interactive Platform.

A multi-tab Streamlit application integrating experiment planning, sample size
sizing, stochastic visitor simulation, CSV upload, SQL-based aggregation,
classical hypothesis testing, assumption checks, and educational simulations.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.assign import assign, assign_and_store
from src.db import (
    clear_experiment_data,
    create_experiment,
    get_connection,
    get_counts,
    get_experiment,
    hash_hypothesis,
    init_db,
    list_experiments,
    log_event,
    log_events_batch,
    store_assignments_batch,
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
    page_title="A/B Testing Decision Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Database
conn = init_db()

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "experiment_id" not in st.session_state:
    st.session_state["experiment_id"] = None
if "is_locked" not in st.session_state:
    st.session_state["is_locked"] = False
if "demo_visitor_id" not in st.session_state:
    st.session_state["demo_visitor_id"] = 101

# Sidebar: Experiment Context & Selector
st.sidebar.title("🧪 Experiment Suite")
st.sidebar.markdown("---")

all_exps = list_experiments(conn)

if all_exps:
    exp_options = {f"#{e['id']} - {e['name']}": e["id"] for e in all_exps}
    selected_label = st.sidebar.selectbox(
        "Active Experiment",
        options=list(exp_options.keys()),
        index=0 if st.session_state["experiment_id"] is None else (
            list(exp_options.values()).index(st.session_state["experiment_id"])
            if st.session_state["experiment_id"] in exp_options.values() else 0
        ),
    )
    selected_id = exp_options[selected_label]
    if st.session_state["experiment_id"] != selected_id:
        st.session_state["experiment_id"] = selected_id
        st.session_state["is_locked"] = True
else:
    st.sidebar.info("No experiments found. Create and lock one in the **PLAN** tab.")

if st.sidebar.button("➕ Create New Experiment", use_container_width=True):
    st.session_state["experiment_id"] = None
    st.session_state["is_locked"] = False
    st.rerun()

current_exp: Optional[Dict[str, Any]] = None
if st.session_state["experiment_id"] is not None:
    current_exp = get_experiment(conn, st.session_state["experiment_id"])
    if current_exp is not None:
        st.session_state["is_locked"] = True
        st.sidebar.markdown(f"**Status:** 🔒 Locked")
        st.sidebar.markdown(f"**Name:** `{current_exp['name']}`")
        st.sidebar.markdown(f"**Planned N:** `{current_exp['planned_n_per_group']:,} / group`")
        st.sidebar.markdown(f"**Target MDE:** `{current_exp['mde'] * 100:.2f}%`")
        st.sidebar.markdown(f"**Alpha (α):** `{current_exp['alpha']:.2f}`")
        st.sidebar.markdown(f"**Power (1-β):** `{current_exp['power']:.2f}`")
    else:
        st.session_state["experiment_id"] = None
        st.session_state["is_locked"] = False

st.title("🧪 Statistical A/B Testing & Decision Engine")
st.markdown(
    "A rigorous, end-to-end framework for experiment pre-registration, user assignment, "
    "stochastic simulation, SRM validation, and frequentist hypothesis testing."
)

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_plan, tab_data, tab_verdict, tab_learn, tab_demo = st.tabs(
    ["1. PLAN", "2. DATA", "3. VERDICT", "4. LEARN", "5. DEMO"]
)

# =============================================================================
# TAB 1: PLAN
# =============================================================================
with tab_plan:
    st.header("1. Experiment Planning & Hypothesis Pre-Registration")
    st.markdown(
        "Define your hypothesis and business decision rule before launching data collection. "
        "Pre-registering and locking the hypothesis protects against p-hacking, HARKing, "
        "and arbitrary early-stopping bias."
    )

    is_locked = bool(st.session_state.get("is_locked", False) and current_exp is not None)

    if is_locked and current_exp:
        st.success(
            f"🔒 **Hypothesis Locked & Pre-Registered:** Experiment `#{current_exp['id']} - {current_exp['name']}` "
            f"was locked at `{current_exp['locked_at']}` UTC."
        )
        st.code(f"Hypothesis SHA-256 Digest:\n{current_exp['hypothesis_hash']}", language="text")

    col_name, col_rule = st.columns([1, 1])

    with col_name:
        exp_name = st.text_input(
            "Experiment Name",
            value=current_exp["name"] if current_exp else "checkout_redesign_v1",
            disabled=is_locked,
            help="Unique machine-readable identifier for this experiment.",
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
            help="Helpful Example: Ship Variant B if p-value < 0.05 and the 95% confidence interval lower bound > 0.",
            height=100,
        )

    hypothesis_text = st.text_area(
        "Hypothesis Statement",
        value=(
            current_exp["hypothesis"]
            if current_exp
            else "Replacing the multi-step checkout flow with a simplified 1-click express checkout will increase completion rate from 10% to 11%."
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
            if not exp_name.strip():
                st.error("Please enter a valid Experiment Name before locking.")
            elif not hypothesis_text.strip():
                st.error("Please enter a non-empty Hypothesis before locking.")
            elif req_sample_size <= 0:
                st.error("Invalid sample size calculation.")
            else:
                now_str = datetime.now(timezone.utc).isoformat()
                new_id = create_experiment(
                    conn=conn,
                    name=exp_name.strip(),
                    hypothesis=hypothesis_text.strip(),
                    success_rule=success_rule.strip(),
                    baseline_rate=float(baseline_rate),
                    mde=float(lift),
                    alpha=float(alpha),
                    power=float(power),
                    planned_n_per_group=int(req_sample_size),
                    locked_at=now_str,
                )
                h_digest = hash_hypothesis(hypothesis_text)
                st.session_state["experiment_id"] = new_id
                st.session_state["is_locked"] = True
                st.success(
                    f"✅ **Experiment Successfully Locked!** Registered as ID #{new_id}. "
                    f"Timestamp: `{now_str}` UTC."
                )
                st.code(f"SHA-256 Digest: {h_digest}", language="text")
                st.rerun()
    else:
        st.caption("Hypothesis inputs are locked and read-only. Data collection and analysis tabs are unlocked.")


# =============================================================================
# TAB 2: DATA
# =============================================================================
with tab_data:
    st.header("2. Data Collection & Population Ingestion")

    if not st.session_state.get("is_locked") or current_exp is None:
        st.warning(
            "⚠️ **Data tabs are disabled until an experiment is locked.** "
            "Please go to the **1. PLAN** tab and click 'Lock Hypothesis & Register Experiment'."
        )
    else:
        st.markdown(
            f"Populate visitor observations for locked experiment: "
            f"`#{current_exp['id']} - {current_exp['name']}`."
        )

        data_option = st.radio(
            "Choose Data Source:",
            ["Option A: Simulate Visitors", "Option B: Upload CSV"],
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
                    value=15000,
                    step=1000,
                    help="Sample size to target for each variant (A and B).",
                )
            with col_s2:
                sim_true_a = st.number_input(
                    "TRUE Rate for A (Control)",
                    min_value=0.001,
                    max_value=0.999,
                    value=0.10,
                    step=0.01,
                    format="%.3f",
                )
            with col_s3:
                sim_true_b = st.number_input(
                    "TRUE Rate for B (Treatment)",
                    min_value=0.001,
                    max_value=0.999,
                    value=0.11,
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

                # Clear previous data to avoid double counting across simulation runs
                clear_experiment_data(conn, exp_id)

                progress_bar = st.progress(0.0)
                status_placeholder = st.empty()

                chunk_size = 5000 if total_sim_users > 20000 else total_sim_users
                total_processed = 0

                for start_idx in range(1, total_sim_users + 1, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_sim_users + 1)
                    chunk_uids = list(range(start_idx, end_idx))
                    chunk_len = len(chunk_uids)

                    # For large n (>20,000) batch operations; otherwise call assign_and_store per user
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
                st.info("👉 Switch to the **3. VERDICT** tab to view hypothesis test results and decision cards.")

        # ---------------------------------------------------------------------
        # OPTION B: UPLOAD CSV
        # ---------------------------------------------------------------------
        else:
            st.subheader("Option B: Upload External Dataset (CSV)")
            st.markdown(
                "Upload a CSV file containing experiment records. Map columns to group variants "
                "and conversion flags without relying on hardcoded column names."
            )

            uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.markdown(f"**Loaded Preview:** `{df.shape[0]:,}` rows × `{df.shape[1]}` columns")
                    st.dataframe(df.head(5), use_container_width=True)

                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        group_col = st.selectbox(
                            "Select Group / Variant Column",
                            options=df.columns,
                            index=0,
                        )
                    with col_m2:
                        conv_col = st.selectbox(
                            "Select Converted / Outcome Column",
                            options=df.columns,
                            index=min(1, len(df.columns) - 1),
                        )

                    unique_groups = df[group_col].dropna().unique().tolist()
                    st.markdown("##### Value Mapping")
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        val_a = st.selectbox(
                            "Value representing Variant A (Control)",
                            options=unique_groups,
                            index=0 if unique_groups else 0,
                        )
                    with col_v2:
                        val_b = st.selectbox(
                            "Value representing Variant B (Treatment)",
                            options=unique_groups,
                            index=1 if len(unique_groups) > 1 else 0,
                        )

                    clear_before_import = st.checkbox(
                        "Clear existing experiment data before importing",
                        value=True,
                    )

                    if st.button("📥 Ingest CSV Data into Experiment", type="primary", use_container_width=True):
                        exp_id = current_exp["id"]
                        if clear_before_import:
                            clear_experiment_data(conn, exp_id)

                        now_iso = datetime.now(timezone.utc).isoformat()
                        sub_df = df[[group_col, conv_col]].dropna()

                        assignments: List[Tuple[int, int, str, str]] = []
                        view_events: List[Tuple[int, int, str, str]] = []
                        order_events: List[Tuple[int, int, str, str]] = []

                        # Identify conversion boolean values
                        positive_conv = {1, "1", True, "True", "true", "yes", "Y", "converted"}

                        for idx, row in sub_df.iterrows():
                            uid = int(idx) + 1
                            raw_grp = row[group_col]
                            if raw_grp == val_a:
                                var = "A"
                            elif raw_grp == val_b:
                                var = "B"
                            else:
                                continue

                            assignments.append((uid, exp_id, var, now_iso))
                            view_events.append((uid, exp_id, "view_checkout", now_iso))

                            raw_conv = row[conv_col]
                            if raw_conv in positive_conv or (
                                isinstance(raw_conv, (int, float)) and raw_conv > 0
                            ):
                                order_events.append((uid, exp_id, "order_complete", now_iso))

                        store_assignments_batch(conn, assignments)
                        log_events_batch(conn, view_events)
                        log_events_batch(conn, order_events)

                        counts = get_counts(conn, exp_id)
                        st.success(
                            f"🎉 Ingested **{counts['total_users']:,} records** from CSV! "
                            f"Variant A: {counts['n_a']:,} users ({counts['x_a']:,} conversions). "
                            f"Variant B: {counts['n_b']:,} users ({counts['x_b']:,} conversions)."
                        )
                        st.info("👉 Check the **3. VERDICT** tab for statistical analysis.")
                except Exception as ex:
                    st.error(f"Error parsing uploaded file: {ex}")


# =============================================================================
# TAB 3: VERDICT
# =============================================================================
with tab_verdict:
    st.header("3. Statistical Verdict & Decision Card")

    if not st.session_state.get("is_locked") or current_exp is None:
        st.warning(
            "⚠️ **Verdict tab is disabled until an experiment is locked.** "
            "Please go to the **1. PLAN** tab to configure and lock your hypothesis."
        )
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
                "ℹ️ **No visitor data logged yet.** "
                "Simulate visitor traffic or upload a CSV in the **2. DATA** tab to compute the verdict."
            )
        else:
            exp_alpha = float(current_exp["alpha"])
            planned_n = int(current_exp["planned_n_per_group"])

            # Run core statistical engine functions
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

            # 2. Key Statistical Tests with mandatory side-by-side presentation
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
                        "\n\n**Common Pitfall:** The p-value is **NOT** the probability that Variant B is better than Variant A. "
                        "A p-value below α (typically 0.05) indicates the observed difference is unlikely due to chance alone."
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
                        f"lies between **{test_res['ci_low'] * 100:+.2f}%** and **{test_res['ci_high'] * 100:+.2f}%**. "
                        "\n\nIf the entire interval sits strictly above zero, Variant B provides a guaranteed positive gain within tolerance."
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
                        "independent of sample size. \n\n"
                        "- **|h| < 0.2**: Small effect\n"
                        "- **|h| ≈ 0.5**: Medium effect\n"
                        "- **|h| > 0.8**: Large effect\n\n"
                        "In high-traffic e-commerce, small effect sizes (e.g. h = 0.03) can still deliver substantial revenue."
                    )

            # 3. Simple Confidence Interval Chart with Line at Zero
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

            # Dynamic x-limits
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

            # 5. Prominent Early-Stopping Warning if sample size < planned
            min_observed_n = min(n_a, n_b)
            if min_observed_n < planned_n:
                st.warning(
                    f"⚠️ **PROMINENT EARLY-STOPPING WARNING (PEEKING PROBLEM)**\n\n"
                    f"The experiment collected fewer users per variant than the planned pre-registered sample size: "
                    f"**{min_observed_n:,} observed** vs. **{planned_n:,} planned per version** "
                    f"({(min_observed_n / float(planned_n)) * 100:.1f}% of target reached). "
                    f"Stopping or deciding early without an alpha-spending function severely inflates false-positive rates. "
                    f"Treat these results as preliminary and continue data collection."
                )

            # 6. Decision Card from decide()
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
# TAB 4: LEARN
# =============================================================================
with tab_learn:
    st.header("4. Learn: Statistical Verification & Simulation")
    st.markdown(
        "Interactive Monte Carlo simulations demonstrating the foundational concepts of classical A/B testing: "
        "A/A calibration and the destructive effects of optional stopping (the peeking problem)."
    )

    tab_aa, tab_peek = st.tabs(["A/A Test Calibration", "The Peeking Problem"])

    # -------------------------------------------------------------------------
    # SUBTAB: A/A TEST
    # -------------------------------------------------------------------------
    with tab_aa:
        st.subheader("A/A Test: P-Value Uniformity under H₀")
        st.markdown(
            "When testing two identical variants (Variant A vs. Variant A), the null hypothesis is completely true. "
            "Under H₀, p-values must follow a **Uniform(0, 1)** distribution, resulting in a flat histogram. "
            "Exactly ~5% of p-values should fall below α = 0.05 purely by random sampling noise."
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
            counts, edges, patches = ax.hist(
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

            st.caption(
                f"Caption: Under the null hypothesis (A/A test), p-values are uniformly distributed between 0 and 1, "
                f"with {frac_sig * 100:.1f}% falling below α = 0.05 (expected 5.0% ± binomial sampling noise)."
            )

    # -------------------------------------------------------------------------
    # SUBTAB: PEEKING PROBLEM
    # -------------------------------------------------------------------------
    with tab_peek:
        st.subheader("The Peeking Problem: False Positive Inflation")
        st.markdown(
            "Continuously checking p-values as visitors arrive and stopping when p < 0.05 "
            "drastically increases the probability of declaring a false winner. "
            "Without adjusting the significance threshold (e.g. via O'Brien-Fleming or Pocock boundaries), "
            "Type I error rates quickly exceed 20% to 30%."
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

            st.caption(
                f"Caption: Repeatedly evaluating accumulating data daily without correction inflates the false-positive rate "
                f"from {rates[0]:.1f}% on Day 1 to {final_fpr:.1f}% by Day {len(peeks)}, far exceeding the intended 5% risk."
            )


# =============================================================================
# TAB 5: DEMO
# =============================================================================
with tab_demo:
    st.header("5. Live Variant Assignment & Interactive Checkout Demo")

    if not st.session_state.get("is_locked") or current_exp is None:
        st.warning(
            "⚠️ **Demo tab is disabled until an experiment is locked.** "
            "Please lock an experiment in the **1. PLAN** tab to test live user assignments."
        )
    else:
        exp_id = current_exp["id"]
        exp_name = current_exp["name"]

        st.markdown(
            "Test deterministic user assignment in real-time. Enter a Visitor ID or generate a random one. "
            "The app deterministically hashes `{experiment_name}-{user_id}` via SHA-256 to assign the visitor "
            "to Variant A or B, renders the corresponding page, and logs conversion events upon clicking."
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
                rand_id = random.randint(1000, 99999)
                st.session_state["demo_visitor_id"] = rand_id
                st.rerun()

        active_uid = int(st.session_state["demo_visitor_id"])
        # Deterministically assign visitor
        assigned_variant = assign(active_uid, exp_name)

        # Log checkout view event
        now_iso = datetime.now(timezone.utc).isoformat()
        assign_and_store(conn, active_uid, exp_id, exp_name)
        log_event(conn, active_uid, exp_id, "view_checkout", now_iso)

        st.markdown("---")
        st.subheader(f"Visitor View: User #{active_uid} assigned to **Variant {assigned_variant}**")

        # Render corresponding mock UI
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

                if st.button("Complete Purchase ($49.00)", type="secondary", key="btn_order_a"):
                    log_event(conn, active_uid, exp_id, "order_complete", datetime.now(timezone.utc).isoformat())
                    st.balloons()
                    st.success(f"🎉 Order Completed! Recorded conversion event for User #{active_uid} in Variant A.")
        else:
            with st.container(border=True):
                st.markdown("### ⚡ Express 1-Click Checkout (Variant B - Treatment)")
                st.markdown("Optimized one-click checkout with buyer guarantees and trusted payment badges.")
                st.markdown(
                    "🔒 **256-Bit SSL Encrypted** | ⭐ **4.9/5 Rating (12k reviews)** | 🚚 **Free Express Delivery**"
                )
                if st.button("⚡ Instant 1-Click Purchase ($49.00)", type="primary", key="btn_order_b"):
                    log_event(conn, active_uid, exp_id, "order_complete", datetime.now(timezone.utc).isoformat())
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
