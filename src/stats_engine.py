"""Statistical analysis engine for A/B testing.

Implements rigorous hypothesis testing, sample size calculation, sample ratio mismatch
(SRM) checks, and business decision mapping according to classical frequentist statistics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from scipy.stats import chisquare, norm


def sample_size(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Calculate required sample size per group for a two-proportion z-test.

    Uses the unpooled variance formula under the alternative hypothesis to determine
    the minimum number of observations required in each variant to detect an absolute
    difference of |p2 - p1| at significance level alpha and statistical power (1 - beta).

    Args:
        p1: Baseline conversion rate (0 < p1 < 1).
        p2: Target conversion rate (0 < p2 < 1, p2 != p1).
        alpha: Type I error rate / significance level (default: 0.05).
        power: Statistical power / (1 - Type II error rate) (default: 0.80).

    Returns:
        Integer minimum sample size required per group (rounded up to nearest int).

    Raises:
        ValueError: If proportions are identical or outside (0, 1), or if alpha/power
            are outside (0, 1).
    """
    if p1 == p2:
        raise ValueError("Proportions p1 and p2 must be distinct to compute sample size.")
    if not (0.0 < p1 < 1.0 and 0.0 < p2 < 1.0):
        raise ValueError("Proportions p1 and p2 must be strictly between 0 and 1.")
    if not (0.0 < alpha < 1.0 and 0.0 < power < 1.0):
        raise ValueError("alpha and power must be strictly between 0 and 1.")

    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_beta = norm.ppf(power)
    delta = abs(p2 - p1)

    variance = p1 * (1.0 - p1) + p2 * (1.0 - p2)
    n = ((z_alpha + z_beta) ** 2 * variance) / (delta ** 2)

    return int(math.ceil(n))


def two_prop_test(
    x_a: int,
    n_a: int,
    x_b: int,
    n_b: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Perform a two-tailed two-proportion z-test comparing variant B to variant A.

    Computes conversion proportions, absolute and relative lift, pooled z-statistic,
    p-value, unpooled Wald confidence interval for the difference (p_b - p_a), Cohen's h
    effect size, and verifies the normal approximation assumptions.

    Edge cases (such as zero users, zero conversions, or 100% conversion rates) are handled
    gracefully to prevent division by zero, returning flagged results.

    Args:
        x_a: Number of conversions in group A.
        n_a: Total sample size in group A.
        x_b: Number of conversions in group B.
        n_b: Total sample size in group B.
        alpha: Significance level for the confidence interval (default: 0.05).

    Returns:
        Dict with keys:
            p_a (float): Conversion rate in group A.
            p_b (float): Conversion rate in group B.
            diff (float): Absolute difference (p_b - p_a).
            relative_lift (float): Relative lift ((p_b - p_a) / p_a).
            z (float): Test statistic z.
            p_value (float): Two-tailed p-value.
            ci_low (float): Lower bound of the (1 - alpha) confidence interval.
            ci_high (float): Upper bound of the (1 - alpha) confidence interval.
            cohens_h (float): Cohen's h effect size measure.
            assumptions_ok (bool): True if success-failure conditions (>= 10) hold.
            assumption_notes (list of str): Detailed diagnostics regarding assumptions.
    """
    notes: List[str] = []

    # Edge case 1: Zero or invalid user sample sizes
    if n_a <= 0 or n_b <= 0:
        notes.append("Zero or non-positive users in one or both groups; cannot compute test.")
        return {
            "p_a": 0.0,
            "p_b": 0.0,
            "diff": 0.0,
            "relative_lift": 0.0,
            "z": 0.0,
            "p_value": 1.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "ci": (0.0, 0.0),
            "cohens_h": 0.0,
            "assumptions_ok": False,
            "assumption_notes": notes,
            "alpha": float(alpha),
        }

    # Clamp conversions to valid bounds [0, n] for numerical safety
    x_a_clamped = max(0, min(int(x_a), int(n_a)))
    x_b_clamped = max(0, min(int(x_b), int(n_b)))

    p_a = x_a_clamped / n_a
    p_b = x_b_clamped / n_b
    diff = p_b - p_a

    # Relative lift computation
    if p_a > 0.0:
        relative_lift = diff / p_a
    elif p_b > 0.0:
        relative_lift = float("inf")
    else:
        relative_lift = 0.0

    p_pool = (x_a_clamped + x_b_clamped) / (n_a + n_b)

    # Edge cases 2 & 3: p_pool is 0 or 1 (zero conversions or 100% conversions)
    if p_pool <= 0.0 or p_pool >= 1.0:
        if p_pool <= 0.0:
            notes.append("Zero conversions observed across all groups (p_pool = 0); pooled variance is 0.")
        else:
            notes.append("100% conversion rate observed across all groups (p_pool = 1); pooled variance is 0.")
        return {
            "p_a": float(p_a),
            "p_b": float(p_b),
            "diff": float(diff),
            "relative_lift": float(relative_lift),
            "z": 0.0,
            "p_value": 1.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "ci": (0.0, 0.0),
            "cohens_h": 0.0,
            "assumptions_ok": False,
            "assumption_notes": notes,
            "alpha": float(alpha),
        }

    # Test statistic z using pooled standard error under H0
    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    z = diff / se_pool
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z))))

    # Confidence interval using unpooled standard error under H1
    se_diff = math.sqrt(p_a * (1.0 - p_a) / n_a + p_b * (1.0 - p_b) / n_b)
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    ci_low = float(diff - z_alpha * se_diff)
    ci_high = float(diff + z_alpha * se_diff)

    # Cohen's h effect size: 2 * (arcsin(sqrt(p_b)) - arcsin(sqrt(p_a)))
    p_b_safe = min(1.0, max(0.0, p_b))
    p_a_safe = min(1.0, max(0.0, p_a))
    cohens_h = float(2.0 * (math.asin(math.sqrt(p_b_safe)) - math.asin(math.sqrt(p_a_safe))))

    # Check normal approximation assumptions (success-failure condition: counts >= 10)
    if x_a < 10:
        notes.append(f"Success count in group A ({x_a}) is less than 10.")
    if (n_a - x_a) < 10:
        notes.append(f"Failure count in group A ({n_a - x_a}) is less than 10.")
    if x_b < 10:
        notes.append(f"Success count in group B ({x_b}) is less than 10.")
    if (n_b - x_b) < 10:
        notes.append(f"Failure count in group B ({n_b - x_b}) is less than 10.")

    assumptions_ok = len(notes) == 0
    if assumptions_ok:
        notes.append("Success-failure condition satisfied (at least 10 successes and failures in each group).")

    return {
        "p_a": float(p_a),
        "p_b": float(p_b),
        "diff": float(diff),
        "relative_lift": float(relative_lift),
        "z": float(z),
        "p_value": float(p_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci": (float(ci_low), float(ci_high)),
        "cohens_h": float(cohens_h),
        "assumptions_ok": assumptions_ok,
        "assumption_notes": notes,
        "alpha": float(alpha),
    }


def decide(
    result: Dict[str, Any],
    planned_n_per_group: Optional[int],
    n_a: int,
    n_b: int,
) -> Dict[str, str]:
    """Map hypothesis test outcomes to an actionable business decision and explanation.

    Decision rules:
        - "SHIP_B" if p_value < alpha AND ci_low > 0
        - "KEEP_A" if p_value < alpha AND ci_high < 0 (B is significantly worse)
        - "INCONCLUSIVE" otherwise

    If planned_n_per_group is provided and either group is smaller, a warning is included
    noting that the experiment was stopped early and results are unreliable due to the
    peeking problem.

    The explanation provides 2 to 3 plain, non-technical sentences and strictly avoids
    stating or implying that the p-value represents the probability that variant B is better.

    Args:
        result: Dictionary returned by two_prop_test.
        planned_n_per_group: Planned sample size per variant, or None if unspecified.
        n_a: Observed sample size in group A.
        n_b: Observed sample size in group B.

    Returns:
        Dict with keys "decision" (str) and "explanation" (str).
    """
    alpha = float(result.get("alpha", 0.05))
    p_val = float(result.get("p_value", 1.0))
    ci_low = float(result.get("ci_low", 0.0))
    ci_high = float(result.get("ci_high", 0.0))

    stopped_early = bool(
        planned_n_per_group is not None
        and (n_a < planned_n_per_group or n_b < planned_n_per_group)
    )

    if p_val < alpha and ci_low > 0.0:
        decision = "SHIP_B"
        if stopped_early:
            explanation = (
                "Variant B shows a statistically significant improvement over Variant A with a positive confidence interval. "
                "However, this test was stopped early before reaching the planned sample size per group, making the findings unreliable due to the peeking problem. "
                "We advise against shipping prematurely because stopping early inflates false discovery rates."
            )
        else:
            explanation = (
                "Variant B demonstrates a statistically significant improvement over Variant A, with a confidence interval confirming positive gains. "
                "The observed lift is unlikely to have occurred by random variation under the assumption that both variants perform the same. "
                "We recommend rolling out Variant B to all users."
            )
    elif p_val < alpha and ci_high < 0.0:
        decision = "KEEP_A"
        if stopped_early:
            explanation = (
                "Variant B currently exhibits a statistically significant decline compared to Variant A. "
                "However, the experiment was stopped before achieving the planned sample size per group, making the conclusion vulnerable to the peeking problem. "
                "We recommend investigating potential issues and completing the planned sample collection before making a final business decision."
            )
        else:
            explanation = (
                "Variant B performs significantly worse than Variant A, resulting in a statistically detectable drop in conversion. "
                "The observed decline is unlikely to be explained by random fluctuation under the assumption of no effect. "
                "We recommend keeping Variant A and discarding Variant B."
            )
    else:
        decision = "INCONCLUSIVE"
        if stopped_early:
            explanation = (
                "The current results show no conclusive difference between Variant A and Variant B beyond expected random variation. "
                "Furthermore, the test was stopped early before reaching the planned sample size per group, introducing the peeking problem and reducing reliability. "
                "We recommend continuing the test until the planned sample size is reached before making any decisions."
            )
        else:
            explanation = (
                "The observed difference in conversion between Variant A and Variant B is not statistically distinguishable from random chance. "
                "There is insufficient evidence to conclude that either variant provides a measurable advantage over the other. "
                "We recommend keeping the baseline Variant A or continuing data collection if a larger sample size is desired."
            )

    return {
        "decision": decision,
        "explanation": explanation,
    }


def sample_ratio_check(n_a: int, n_b: int) -> Dict[str, Any]:
    """Perform a Chi-Square Goodness-of-Fit test against an expected 50/50 allocation ratio.

    Flags a Sample Ratio Mismatch (SRM) if the p-value is strictly less than 0.001.

    Args:
        n_a: Observed user count in group A.
        n_b: Observed user count in group B.

    Returns:
        Dict containing chi2, p_value, flag_problem (bool), srm_detected (bool),
        observed counts, expected counts, and a summary message.
    """
    total = n_a + n_b
    if total <= 0 or n_a < 0 or n_b < 0:
        return {
            "chi2": float("nan"),
            "p_value": 0.0,
            "flag_problem": True,
            "srm_detected": True,
            "expected": {"A": 0.0, "B": 0.0},
            "observed": {"A": int(n_a), "B": int(n_b)},
            "message": "Invalid or zero sample size; sample ratio check cannot be computed.",
        }

    expected = [total / 2.0, total / 2.0]
    observed = [float(n_a), float(n_b)]

    res = chisquare(f_obs=observed, f_exp=expected)
    chi2_val = float(res.statistic)
    p_val = float(res.pvalue)
    problem = bool(p_val < 0.001)

    return {
        "chi2": chi2_val,
        "p_value": p_val,
        "flag_problem": problem,
        "srm_detected": problem,
        "expected": {"A": expected[0], "B": expected[1]},
        "observed": {"A": int(n_a), "B": int(n_b)},
        "message": (
            "Sample Ratio Mismatch (SRM) detected (p < 0.001). Assignment mechanism or tracking may be biased."
            if problem
            else "Sample ratio is consistent with 50/50 allocation (p >= 0.001)."
        ),
    }


def power_for(
    n_per_group: int | float,
    p1: float,
    p2: float,
    alpha: float = 0.05,
) -> float:
    """Calculate the theoretical statistical power for given sample size and proportions.

    Used to construct theoretical power curves across varying sample sizes or effect sizes.

    Args:
        n_per_group: Number of observations per variant.
        p1: Baseline conversion rate (0 <= p1 <= 1).
        p2: Target conversion rate (0 <= p2 <= 1).
        alpha: Significance level (default: 0.05).

    Returns:
        Theoretical power as a float in [0.0, 1.0]. Returns alpha when p1 == p2.

    Raises:
        ValueError: If proportions or alpha are outside valid probability ranges.
    """
    if n_per_group <= 0:
        return 0.0
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("Proportions p1 and p2 must be between 0 and 1.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1.")

    variance = p1 * (1.0 - p1) + p2 * (1.0 - p2)
    if variance <= 0.0:
        return 1.0 if p1 != p2 else float(alpha)

    se = math.sqrt(variance / float(n_per_group))
    delta = abs(p2 - p1)
    z_alpha = norm.ppf(1.0 - alpha / 2.0)

    # Standard two-sided power: Phi(delta / se - z_alpha) + Phi(-delta / se - z_alpha)
    term1 = norm.cdf(delta / se - z_alpha)
    term2 = norm.cdf(-delta / se - z_alpha)

    return float(term1 + term2)
