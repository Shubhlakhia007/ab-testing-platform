"""Unit tests for the statistical analysis engine (src/stats_engine.py).

Verifies ground truth precision, edge-case safety, business decision mapping,
and validates calculations against statsmodels.stats.proportion.proportions_ztest.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
from statsmodels.stats.proportion import proportions_ztest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.stats_engine import (
    decide,
    power_for,
    sample_ratio_check,
    sample_size,
    two_prop_test,
)


def test_sample_size_ground_truth() -> None:
    """Verify that sample_size(0.10, 0.11) returns exactly 14749."""
    n = sample_size(0.10, 0.11, alpha=0.05, power=0.80)
    assert n == 14749, f"Expected 14749, got {n}"
    assert isinstance(n, int)


def test_two_prop_test_ground_truth() -> None:
    """Verify ground truth precision for two_prop_test(1500, 15000, 1650, 15000)."""
    res = two_prop_test(1500, 15000, 1650, 15000, alpha=0.05)

    assert round(res["z"], 3) == 2.825, f"Expected z=2.825, got {round(res['z'], 3)}"
    assert round(res["p_value"], 4) == 0.0047, f"Expected p=0.0047, got {round(res['p_value'], 4)}"
    assert round(res["ci_low"], 4) == 0.0031, f"Expected ci_low=0.0031, got {round(res['ci_low'], 4)}"
    assert round(res["ci_high"], 4) == 0.0169, f"Expected ci_high=0.0169, got {round(res['ci_high'], 4)}"
    assert round(res["cohens_h"], 4) == 0.0326, f"Expected cohens_h=0.0326, got {round(res['cohens_h'], 4)}"
    assert res["assumptions_ok"] is True
    assert isinstance(res["assumption_notes"], list)
    assert len(res["assumption_notes"]) > 0


def test_cross_check_statsmodels_agreement() -> None:
    """Cross-check z and p_value against statsmodels proportions_ztest to 4+ decimals."""
    test_cases = [
        # (x_a, n_a, x_b, n_b)
        (1500, 15000, 1650, 15000),  # Baseline ground truth
        (1200, 10000, 1100, 10000),  # B worse than A (negative z)
        (500, 5000, 505, 5000),      # Small difference / inconclusive
        (2500, 20000, 2700, 20000),  # Large sample size
    ]

    for x_a, n_a, x_b, n_b in test_cases:
        res = two_prop_test(x_a, n_a, x_b, n_b, alpha=0.05)
        # Note: proportions_ztest accepts count=[count_b, count_a] to evaluate (p_b - p_a)
        sm_z, sm_p = proportions_ztest(count=[x_b, x_a], nobs=[n_b, n_a])

        assert math.isclose(res["z"], float(sm_z), abs_tol=1e-4), (
            f"z-score mismatch with statsmodels for case ({x_a}, {n_a}, {x_b}, {n_b}): "
            f"ours={res['z']}, statsmodels={sm_z}"
        )
        assert math.isclose(res["p_value"], float(sm_p), abs_tol=1e-4), (
            f"p-value mismatch with statsmodels for case ({x_a}, {n_a}, {x_b}, {n_b}): "
            f"ours={res['p_value']}, statsmodels={sm_p}"
        )


def test_edge_case_zero_users() -> None:
    """Zero users in one or both groups must not crash with ZeroDivisionError."""
    res1 = two_prop_test(0, 0, 10, 100)
    assert res1["assumptions_ok"] is False
    assert res1["z"] == 0.0
    assert res1["p_value"] == 1.0
    assert any("zero" in note.lower() for note in res1["assumption_notes"])

    res2 = two_prop_test(10, 100, 0, 0)
    assert res2["assumptions_ok"] is False
    assert res2["z"] == 0.0

    res3 = two_prop_test(0, 0, 0, 0)
    assert res3["assumptions_ok"] is False


def test_edge_case_zero_conversions() -> None:
    """Zero conversions across both groups (p_pool = 0) must return flagged result without crashing."""
    res = two_prop_test(0, 1000, 0, 1000)
    assert res["assumptions_ok"] is False
    assert res["z"] == 0.0
    assert res["p_value"] == 1.0
    assert res["diff"] == 0.0
    assert res["relative_lift"] == 0.0
    assert any("p_pool = 0" in note or "zero" in note.lower() for note in res["assumption_notes"])


def test_edge_case_all_conversions_pool_one() -> None:
    """100% conversion rate in both groups (p_pool = 1) must return flagged result without crashing."""
    res = two_prop_test(500, 500, 500, 500)
    assert res["assumptions_ok"] is False
    assert res["z"] == 0.0
    assert res["p_value"] == 1.0
    assert res["diff"] == 0.0
    assert any("p_pool = 1" in note or "100%" in note for note in res["assumption_notes"])


def test_edge_case_one_group_zero_conversions() -> None:
    """When group A has 0 conversions and group B has conversions, relative_lift must not raise ZeroDivisionError."""
    res = two_prop_test(0, 1000, 50, 1000)
    assert res["p_a"] == 0.0
    assert res["p_b"] == 0.05
    assert math.isinf(res["relative_lift"])
    assert res["assumptions_ok"] is False  # x_a < 10 violates success-failure condition
    assert res["z"] > 0
    assert res["p_value"] < 0.05


def test_assumptions_diagnostic_flagging() -> None:
    """Sample sizes with counts < 10 must flag assumptions_ok=False."""
    # Only 5 conversions in group A
    res = two_prop_test(5, 1000, 50, 1000)
    assert res["assumptions_ok"] is False
    assert any("group A (5) is less than 10" in note for note in res["assumption_notes"])


def test_decide_ship_b() -> None:
    """Significant positive lift should result in SHIP_B."""
    res = two_prop_test(1500, 15000, 1650, 15000, alpha=0.05)
    decision = decide(res, planned_n_per_group=15000, n_a=15000, n_b=15000)
    assert decision["decision"] == "SHIP_B"
    assert "roll out" in decision["explanation"].lower() or "recommend" in decision["explanation"].lower()
    # Ensure explanation is 2-3 plain sentences and does not misinterpret p-value
    sentences = [s for s in decision["explanation"].split(".") if s.strip()]
    assert 2 <= len(sentences) <= 3
    assert "chance b is better" not in decision["explanation"].lower()
    assert "probability b is better" not in decision["explanation"].lower()


def test_decide_keep_a() -> None:
    """Significant negative difference should result in KEEP_A."""
    res = two_prop_test(1650, 15000, 1500, 15000, alpha=0.05)
    decision = decide(res, planned_n_per_group=15000, n_a=15000, n_b=15000)
    assert decision["decision"] == "KEEP_A"
    assert "worse" in decision["explanation"].lower() or "keep variant a" in decision["explanation"].lower()
    sentences = [s for s in decision["explanation"].split(".") if s.strip()]
    assert 2 <= len(sentences) <= 3
    assert "chance b is better" not in decision["explanation"].lower()


def test_decide_inconclusive() -> None:
    """Non-significant results should be INCONCLUSIVE."""
    res = two_prop_test(1500, 15000, 1510, 15000, alpha=0.05)
    decision = decide(res, planned_n_per_group=15000, n_a=15000, n_b=15000)
    assert decision["decision"] == "INCONCLUSIVE"
    sentences = [s for s in decision["explanation"].split(".") if s.strip()]
    assert 2 <= len(sentences) <= 3


def test_decide_peeking_early_stopping_warning() -> None:
    """If observed sample size is below planned_n_per_group, include early stopping warning."""
    res = two_prop_test(1500, 15000, 1650, 15000, alpha=0.05)
    decision = decide(res, planned_n_per_group=20000, n_a=15000, n_b=15000)
    assert decision["decision"] == "SHIP_B"
    assert "peeking" in decision["explanation"].lower()
    assert "stopped early" in decision["explanation"].lower()
    sentences = [s for s in decision["explanation"].split(".") if s.strip()]
    assert 2 <= len(sentences) <= 3


def test_sample_ratio_check() -> None:
    """Verify sample ratio mismatch (SRM) detection via chi-square goodness of fit."""
    # Balanced: 50,000 vs 50,000 -> p ~ 1.0
    res_balanced = sample_ratio_check(50000, 50000)
    assert res_balanced["flag_problem"] is False
    assert res_balanced["srm_detected"] is False
    assert res_balanced["p_value"] > 0.05

    # Severely skewed: 48,000 vs 52,000 (huge sample size -> p < 0.001)
    res_skewed = sample_ratio_check(48000, 52000)
    assert res_skewed["flag_problem"] is True
    assert res_skewed["srm_detected"] is True
    assert res_skewed["p_value"] < 0.001

    # Edge case: zero counts
    res_zero = sample_ratio_check(0, 0)
    assert res_zero["flag_problem"] is True


def test_power_for_theoretical_curve() -> None:
    """Verify power_for returns expected power and matches sample_size calculation."""
    # Power for n=14749 at 0.10 and 0.11 should be approximately 0.80
    pw = power_for(14749, 0.10, 0.11, alpha=0.05)
    assert math.isclose(pw, 0.80, abs_tol=1e-3)

    # When p1 == p2, power equals type I error rate alpha (0.05)
    pw_null = power_for(10000, 0.10, 0.10, alpha=0.05)
    assert math.isclose(pw_null, 0.05, abs_tol=1e-5)

    # Edge cases: n <= 0
    assert power_for(0, 0.10, 0.11) == 0.0
    assert power_for(-100, 0.10, 0.11) == 0.0


def test_sample_size_validation_errors() -> None:
    """Verify input validation errors for sample_size."""
    with pytest.raises(ValueError, match="distinct"):
        sample_size(0.10, 0.10)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        sample_size(0.0, 0.10)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        sample_size(0.10, 1.0)
