"""Unit tests for simulation routines and plotting in src/simulate.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulate import (
    aa_test,
    format_theory_vs_experiment,
    generate_all_plots,
    peeking,
    plot_aa_p_values,
    plot_peeking_curve,
    plot_power_curve_comparison,
    power_check,
    power_curve,
    simulate_experiment,
)
from src.stats_engine import power_for


def test_simulate_experiment_determinism_and_types() -> None:
    """simulate_experiment must be deterministic when given a fixed seed."""
    x_a1, x_b1 = simulate_experiment(1000, 0.10, 0.12, seed=42)
    x_a2, x_b2 = simulate_experiment(1000, 0.10, 0.12, seed=42)

    assert isinstance(x_a1, int)
    assert isinstance(x_b1, int)
    assert x_a1 == x_a2
    assert x_b1 == x_b2
    assert 0 <= x_a1 <= 1000
    assert 0 <= x_b1 <= 1000


def test_simulate_experiment_generator_reuse() -> None:
    """simulate_experiment should accept an existing np.random.Generator without re-seeding."""
    rng = np.random.default_rng(123)
    x_a, x_b = simulate_experiment(500, 0.20, 0.20, seed=rng)
    assert 0 <= x_a <= 500
    assert 0 <= x_b <= 500


def test_simulate_experiment_validation() -> None:
    """Invalid parameters should raise ValueError."""
    with pytest.raises(ValueError):
        simulate_experiment(-10, 0.1, 0.1)
    with pytest.raises(ValueError):
        simulate_experiment(100, -0.1, 0.5)
    with pytest.raises(ValueError):
        simulate_experiment(100, 0.1, 1.5)


def test_aa_test_ground_truth_seed_42() -> None:
    """Verify ground truth output of aa_test with seed=42."""
    res = aa_test(n_tests=1000, n=15000, p=0.10, alpha=0.05, seed=42)

    assert res["n_tests"] == 1000
    assert res["significant_count"] == 56
    assert abs(res["fraction_significant"] - 0.056) < 1e-4
    assert len(res["p_values"]) == 1000

    # 95% binomial range for 1,000 runs at 5% is ~36 to 64
    low, high = res["binomial_range_95"]
    assert low == 36
    assert high == 64
    assert low <= res["significant_count"] <= high


def test_peeking_ground_truth_seed_42() -> None:
    """Verify peeking simulation matches expected cumulative rates and monotonicity."""
    res = peeking(n_tests=1000, per_day=1000, days=14, p=0.10, alpha=0.05, seed=42)

    assert res["days"] == 14
    assert len(res["cumulative_fpr"]) == 14
    assert abs(res["false_positive_rate"] - 0.229) < 1e-4

    # Day 1 should be close to nominal alpha ~ 5.8%
    assert abs(res["cumulative_fpr"][0] - 0.058) < 1e-4

    # Monotonically non-decreasing cumulative rates
    for i in range(len(res["cumulative_fpr"]) - 1):
        assert res["cumulative_fpr"][i] <= res["cumulative_fpr"][i + 1]


def test_power_check_and_power_curve() -> None:
    """Verify empirical power computation and agreement with theoretical power."""
    # Empirical power at planned n=14,749
    emp_pow = power_check(n_tests=1000, n=14749, p_a=0.10, p_b=0.11, seed=42)
    theo_pow = power_for(14749, 0.10, 0.11)

    assert 0.75 <= emp_pow <= 0.85
    assert abs(emp_pow - theo_pow) < 0.03  # Within sampling noise

    # Test power_curve with small sample size list
    sizes = [2000, 14749]
    powers = power_curve(sizes, n_tests=200, seed=42)
    assert len(powers) == 2
    assert powers[0] < powers[1]

    # Test as_dict option
    powers_dict = power_curve(sizes, n_tests=200, seed=42, as_dict=True)
    assert isinstance(powers_dict, dict)
    assert 14749 in powers_dict


def test_identical_runs_with_same_seed_give_identical_results() -> None:
    """Verify that all simulation routines produce identical results with identical seed."""
    # 1. simulate_experiment
    res1_sim = simulate_experiment(500, 0.10, 0.12, seed=123)
    res2_sim = simulate_experiment(500, 0.10, 0.12, seed=123)
    assert res1_sim == res2_sim

    # 2. aa_test
    res1_aa = aa_test(n_tests=50, n=1000, p=0.10, seed=123)
    res2_aa = aa_test(n_tests=50, n=1000, p=0.10, seed=123)
    assert res1_aa["significant_count"] == res2_aa["significant_count"]
    assert res1_aa["p_values"] == res2_aa["p_values"]

    # 3. peeking
    res1_peek = peeking(n_tests=50, per_day=200, days=5, p=0.10, seed=123)
    res2_peek = peeking(n_tests=50, per_day=200, days=5, p=0.10, seed=123)
    assert res1_peek["cumulative_fpr"] == res2_peek["cumulative_fpr"]
    assert res1_peek["first_sig_days"] == res2_peek["first_sig_days"]

    # 4. power_check
    res1_pow = power_check(n_tests=50, n=1000, p_a=0.10, p_b=0.15, seed=123)
    res2_pow = power_check(n_tests=50, n=1000, p_a=0.10, p_b=0.15, seed=123)
    assert res1_pow == res2_pow


def test_plot_generation(tmp_path: Path) -> None:
    """Ensure plotting functions write valid image files with captions."""
    aa_fig = plot_aa_p_values(
        p_values=[0.01, 0.04, 0.15, 0.42, 0.77, 0.95],
        output_path=tmp_path / "test_aa.png",
        dpi=50,
    )
    assert aa_fig.exists()
    assert aa_fig.stat().st_size > 500

    peek_fig = plot_peeking_curve(
        peeking_data={
            "days": 3,
            "peeks": [1, 2, 3],
            "cumulative_fpr": [0.05, 0.08, 0.12],
            "false_positive_rate": 0.12,
        },
        output_path=tmp_path / "test_peek.png",
        dpi=50,
    )
    assert peek_fig.exists()
    assert peek_fig.stat().st_size > 500

    power_fig = plot_power_curve_comparison(
        sample_sizes=[2000, 5000],
        empirical_powers=[0.35, 0.65],
        output_path=tmp_path / "test_power.png",
        dpi=50,
    )
    assert power_fig.exists()
    assert power_fig.stat().st_size > 500


def test_theory_vs_experiment_formatting() -> None:
    """Verify that templates format correctly without errors."""
    res = format_theory_vs_experiment(seed=42)
    assert "aa_test" in res
    assert "peeking" in res
    assert "power" in res
    assert "discrepancy_reasons" in res
    assert "5.60%" in res["aa_test"]
    assert "22.90%" in res["peeking"]
    assert "80.10%" in res["power"]
