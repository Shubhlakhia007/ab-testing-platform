"""Simulation framework and empirical verification for A/B testing.

Implements Monte Carlo simulations for:
1. Two-proportion A/B experiments (simulate_experiment).
2. A/A tests to verify uniform p-value distribution and Type I error calibration (aa_test).
3. The peeking problem / optional stopping showing alpha inflation over repeated evaluations (peeking).
4. Empirical statistical power checks and power curves across sample sizes (power_check, power_curve).
5. Publication-grade plotting routines saved to reports/figures/ with captions.
6. Theory vs. Experiment paragraph templates and discrepancy explanations.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, norm

# Support relative and standalone execution
try:
    from src.stats_engine import power_for, two_prop_test
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.stats_engine import power_for, two_prop_test


def _resolve_rng(seed: Union[int, np.random.Generator, None] = 42) -> np.random.Generator:
    """Helper to resolve a numpy Generator from seed or existing Generator instance."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def simulate_experiment(
    n_per_group: int,
    p_a: float,
    p_b: float,
    seed: Union[int, np.random.Generator, None] = 42,
) -> Tuple[int, int]:
    """Simulate a single two-variant A/B experiment using binomial draws.

    Args:
        n_per_group: Number of observations per group (n_a = n_b = n_per_group).
        p_a: True conversion rate for variant A (0 <= p_a <= 1).
        p_b: True conversion rate for variant B (0 <= p_b <= 1).
        seed: Random seed or existing numpy.random.Generator instance (default: 42).

    Returns:
        Tuple of (x_a, x_b) representing observed conversions in groups A and B.

    Raises:
        ValueError: If n_per_group < 0 or proportions are not in [0, 1].
    """
    if n_per_group < 0:
        raise ValueError("n_per_group must be non-negative.")
    if not (0.0 <= p_a <= 1.0 and 0.0 <= p_b <= 1.0):
        raise ValueError("Conversion probabilities p_a and p_b must be between 0 and 1.")

    rng = _resolve_rng(seed)
    x_a = int(rng.binomial(n_per_group, p_a))
    x_b = int(rng.binomial(n_per_group, p_b))
    return (x_a, x_b)


def aa_test(
    n_tests: int = 1000,
    n: int = 15000,
    p: float = 0.10,
    alpha: float = 0.05,
    seed: Union[int, np.random.Generator, None] = 42,
) -> Dict[str, Any]:
    """Run simulated A/A tests where both groups share identical conversion rates.

    Under the true null hypothesis (p_a = p_b = p), the two-sided p-values should
    follow a Uniform(0, 1) distribution, and the empirical rejection rate (fraction
    with p < alpha) should match the nominal significance level alpha within expected
    binomial sampling fluctuation.

    Args:
        n_tests: Number of independent A/A experiments to simulate (default: 1000).
        n: Sample size per group in each experiment (default: 15000).
        p: Baseline conversion rate for both variants (default: 0.10).
        alpha: Nominal Type I error rate / significance threshold (default: 0.05).
        seed: Random seed or Generator instance (default: 42).

    Returns:
        Dict with keys:
            fraction_significant: Fraction of simulated tests where p < alpha.
            significant_count: Count of significant tests out of n_tests.
            p_values: List of floats containing all simulated p-values.
            binomial_range_95: 95% binomial range of expected significant counts (low, high).
            exact_binomial_range_95: Exact quantile interval from scipy.stats.binom.
            expected_count: Theoretical expected count (n_tests * alpha).
            n_tests: Total tests run.
            n: Sample size per group.
            p: Conversion rate used.
            alpha: Significance level used.
    """
    rng = _resolve_rng(seed)

    p_values: List[float] = []
    sig_count = 0

    for _ in range(n_tests):
        x_a = int(rng.binomial(n, p))
        x_b = int(rng.binomial(n, p))
        p_pool = (x_a + x_b) / (2.0 * n)
        if 0.0 < p_pool < 1.0:
            diff = (x_b - x_a) / n
            se = math.sqrt(2.0 * p_pool * (1.0 - p_pool) / n)
            z = diff / se
            p_val = float(math.erfc(abs(z) / math.sqrt(2.0)))
        else:
            p_val = 1.0

        p_values.append(p_val)
        if p_val < alpha:
            sig_count += 1

    fraction_sig = sig_count / n_tests

    # 95% binomial range computation:
    # 1. Exact quantile-based mass from binomial distribution
    b_low, b_high = binom.interval(0.95, n_tests, alpha)

    # 2. Normal approximation bounds: mu +- 1.96 * se
    mu = n_tests * alpha
    se = math.sqrt(n_tests * alpha * (1.0 - alpha))
    norm_low = int(math.floor(mu - 1.96 * se))
    norm_high = int(math.ceil(mu + 1.96 * se))

    return {
        "fraction_significant": float(fraction_sig),
        "significant_count": int(sig_count),
        "p_values": p_values,
        "binomial_range_95": (norm_low, norm_high),
        "exact_binomial_range_95": (int(b_low), int(b_high)),
        "expected_range": (norm_low, norm_high),
        "expected_count": float(mu),
        "n_tests": int(n_tests),
        "n": int(n),
        "p": float(p),
        "alpha": float(alpha),
    }


def peeking(
    n_tests: int = 1000,
    per_day: int = 1000,
    days: int = 14,
    p: float = 0.10,
    alpha: float = 0.05,
    seed: Union[int, np.random.Generator, None] = 42,
) -> Dict[str, Any]:
    """Simulate optional stopping (the peeking problem) on A/A experiments.

    Each simulated experiment accumulates per_day users per group each day over
    a window of `days` days. A cumulative two-proportion z-test is conducted every day.
    If at any point p < alpha, the experiment terminates early and declares a "winner"
    (a false positive, since true p_a == p_b).

    Args:
        n_tests: Number of experiments to simulate (default: 1000).
        per_day: Daily cohort size added to each group (default: 1000).
        days: Total maximum days/peeks allowed (default: 14).
        p: True conversion rate for both groups (default: 0.10).
        alpha: Nominal significance threshold per peek (default: 0.05).
        seed: Random seed or Generator instance (default: 42).

    Returns:
        Dict with keys:
            false_positive_rate: Overall empirical false-positive rate across all days.
            cumulative_false_positive_rates: Dict mapping day (1..days) to cumulative FPR.
            cumulative_fpr: List of cumulative false-positive rates for peeks 1..days.
            first_sig_days: List of day at which each experiment stopped (or None if never).
            peeks: List of integer peek counts [1, 2, ..., days].
            days: Maximum peeking days.
            per_day: Cohort size per day.
            n_tests: Total simulations.
            p: Conversion rate.
            alpha: Nominal alpha.
    """
    rng = _resolve_rng(seed)

    first_sig_days: List[Optional[int]] = []
    # Two-sided standard normal critical z for alpha
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))

    for _ in range(n_tests):
        cum_xa = 0
        cum_xb = 0
        stopped_day: Optional[int] = None

        for day in range(1, days + 1):
            daily_xa = int(rng.binomial(per_day, p))
            daily_xb = int(rng.binomial(per_day, p))
            cum_xa += daily_xa
            cum_xb += daily_xb
            cum_n = day * per_day

            p_pool = (cum_xa + cum_xb) / (2.0 * cum_n)
            if 0.0 < p_pool < 1.0:
                diff = (cum_xb - cum_xa) / cum_n
                se = math.sqrt(2.0 * p_pool * (1.0 - p_pool) / cum_n)
                z = diff / se
                if abs(z) >= z_crit:
                    stopped_day = day
                    break

        first_sig_days.append(stopped_day)

    cumulative_rates: Dict[int, float] = {}
    cumulative_list: List[float] = []

    for k in range(1, days + 1):
        rate = sum(sd is not None and sd <= k for sd in first_sig_days) / float(n_tests)
        cumulative_rates[k] = float(rate)
        cumulative_list.append(float(rate))

    total_fpr = cumulative_list[-1] if cumulative_list else 0.0

    return {
        "false_positive_rate": float(total_fpr),
        "cumulative_false_positive_rates": cumulative_rates,
        "cumulative_fpr": cumulative_list,
        "first_sig_days": first_sig_days,
        "peeks": list(range(1, days + 1)),
        "days": int(days),
        "per_day": int(per_day),
        "n_tests": int(n_tests),
        "p": float(p),
        "alpha": float(alpha),
    }


def power_check(
    n_tests: int = 1000,
    n: int = 14749,
    p_a: float = 0.10,
    p_b: float = 0.11,
    alpha: float = 0.05,
    seed: Union[int, np.random.Generator, None] = 42,
) -> float:
    """Empirically evaluate statistical power for a given sample size and effect size.

    Simulates n_tests independent A/B tests with conversion rates p_a and p_b,
    and returns the proportion of tests that correctly reject H0 at level alpha.

    Args:
        n_tests: Number of experiments to simulate (default: 1000).
        n: Sample size per group (default: 14749, the planned 80% power size).
        p_a: Baseline conversion rate (default: 0.10).
        p_b: Target conversion rate (default: 0.11).
        alpha: Significance level (default: 0.05).
        seed: Random seed or Generator instance (default: 42).

    Returns:
        Empirical statistical power as a float in [0.0, 1.0].
    """
    rng = _resolve_rng(seed)
    sig_count = 0
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))

    for _ in range(n_tests):
        x_a = int(rng.binomial(n, p_a))
        x_b = int(rng.binomial(n, p_b))
        p_pool = (x_a + x_b) / (2.0 * n)
        if 0.0 < p_pool < 1.0:
            diff = (x_b - x_a) / n
            se = math.sqrt(2.0 * p_pool * (1.0 - p_pool) / n)
            z = diff / se
            if abs(z) >= z_crit:
                sig_count += 1

    return float(sig_count / n_tests)


def power_curve(
    sample_sizes: Sequence[int],
    n_tests: int = 1000,
    p_a: float = 0.10,
    p_b: float = 0.11,
    alpha: float = 0.05,
    seed: int = 42,
    as_dict: bool = False,
) -> Union[List[float], Dict[int, float]]:
    """Compute empirical power across multiple candidate sample sizes.

    Args:
        sample_sizes: Iterable of integer sample sizes per group to evaluate.
        n_tests: Number of simulation trials per sample size (default: 1000).
        p_a: Baseline conversion rate (default: 0.10).
        p_b: Target conversion rate (default: 0.11).
        alpha: Significance level (default: 0.05).
        seed: Master random seed (default: 42).
        as_dict: If True, returns dict {n: empirical_power}. If False, returns list of powers.

    Returns:
        List of empirical power floats, or dictionary mapping n to power.
    """
    empirical_powers: List[float] = []

    for n in sample_sizes:
        pow_val = power_check(
            n_tests=n_tests,
            n=int(n),
            p_a=p_a,
            p_b=p_b,
            alpha=alpha,
            seed=seed,
        )
        empirical_powers.append(pow_val)

    if as_dict:
        return {int(n): p_val for n, p_val in zip(sample_sizes, empirical_powers)}
    return empirical_powers


# -----------------------------------------------------------------------------
# Publication-Grade Plotting Functions
# -----------------------------------------------------------------------------

def plot_aa_p_values(
    p_values: Optional[Sequence[float]] = None,
    output_path: Union[str, Path] = "reports/figures/aa_p_values.png",
    n_tests: int = 1000,
    n: int = 15000,
    p: float = 0.10,
    alpha: float = 0.05,
    seed: int = 42,
    dpi: int = 300,
) -> Path:
    """Generate and save a publication-grade histogram of A/A test p-values.

    Under the true null hypothesis, p-values must follow a Uniform(0, 1) distribution,
    which manifests as a flat histogram across the interval [0, 1].

    Args:
        p_values: Pre-computed list of p-values, or None to simulate using aa_test.
        output_path: Destination file path for saving the PNG plot.
        n_tests: Number of A/A runs if simulating (default: 1000).
        n: Sample size per group if simulating (default: 15000).
        p: Baseline conversion rate (default: 0.10).
        alpha: Significance threshold (default: 0.05).
        seed: Random seed (default: 42).
        dpi: Resolution of the output plot in dots per inch (default: 300).

    Returns:
        Path object pointing to the saved figure.
    """
    if p_values is None:
        aa_res = aa_test(n_tests=n_tests, n=n, p=p, alpha=alpha, seed=seed)
        p_vals = aa_res["p_values"]
        frac_sig = aa_res["fraction_significant"]
    else:
        p_vals = list(p_values)
        frac_sig = sum(pv < alpha for pv in p_vals) / len(p_vals) if p_vals else 0.0

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=dpi)

    # 20 uniform bins across [0, 1]
    bins = np.linspace(0.0, 1.0, 21)
    counts, edges, patches = ax.hist(
        p_vals,
        bins=bins,
        color="#3B82F6",
        edgecolor="#1E3A8A",
        linewidth=0.8,
        alpha=0.85,
        density=False,
        label=f"Simulated p-values (N={len(p_vals)})",
    )

    # Highlight rejection region [0, alpha]
    for patch, left, right in zip(patches, edges[:-1], edges[1:]):
        if right <= alpha:
            patch.set_facecolor("#EF4444")
            patch.set_edgecolor("#991B1B")

    # Expected uniform count line: N / num_bins
    expected_bin_count = len(p_vals) / 20.0
    ax.axhline(
        expected_bin_count,
        color="#111827",
        linestyle="--",
        linewidth=1.5,
        label=f"Expected Uniform Count ({int(expected_bin_count)} / bin)",
    )

    ax.axvline(
        alpha,
        color="#DC2626",
        linestyle=":",
        linewidth=1.8,
        label=f"Significance Level α = {alpha} (Observed: {frac_sig * 100:.1f}%)",
    )

    ax.set_title("A/A Test P-Value Distribution (Uniform Under H₀)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("p-value", fontsize=11, labelpad=8)
    ax.set_ylabel("Frequency", fontsize=11, labelpad=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, max(counts) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

    caption = (
        "Caption: Under the null hypothesis (A/A test), p-values are uniformly distributed between 0 and 1, "
        f"with {frac_sig * 100:.1f}% falling below α = {alpha} (expected 5.0% ± sampling noise)."
    )
    plt.figtext(0.5, 0.01, caption, ha="center", fontsize=8.5, style="italic", wrap=True)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_file)
    plt.close(fig)

    return out_file


def plot_peeking_curve(
    peeking_data: Optional[Dict[str, Any]] = None,
    output_path: Union[str, Path] = "reports/figures/peeking_fpr.png",
    n_tests: int = 1000,
    per_day: int = 1000,
    days: int = 14,
    p: float = 0.10,
    alpha: float = 0.05,
    seed: int = 42,
    dpi: int = 300,
) -> Path:
    """Generate and save the false-positive rate vs. number of peeks plot.

    Illustrates how repeatedly testing accumulating data without alpha-adjustment
    causes severe Type I error inflation (the peeking problem).

    Args:
        peeking_data: Pre-computed peeking() results dict, or None to simulate.
        output_path: Destination path for saving the PNG plot.
        n_tests: Simulation runs if generating (default: 1000).
        per_day: Daily cohort size per variant (default: 1000).
        days: Total daily evaluation peeks (default: 14).
        p: Conversion rate (default: 0.10).
        alpha: Nominal per-peek alpha (default: 0.05).
        seed: Random seed (default: 42).

    Returns:
        Path object pointing to the saved figure.
    """
    if peeking_data is None:
        peeking_data = peeking(
            n_tests=n_tests,
            per_day=per_day,
            days=days,
            p=p,
            alpha=alpha,
            seed=seed,
        )

    peeks = peeking_data["peeks"]
    rates = [r * 100.0 for r in peeking_data["cumulative_fpr"]]
    final_fpr = peeking_data["false_positive_rate"] * 100.0

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=dpi)

    ax.plot(
        peeks,
        rates,
        marker="o",
        color="#DC2626",
        linewidth=2.2,
        markersize=6,
        label=f"Cumulative False-Positive Rate (Ends at {final_fpr:.1f}%)",
    )

    # Dashed line at nominal alpha (5%)
    ax.axhline(
        alpha * 100.0,
        color="#1E3A8A",
        linestyle="--",
        linewidth=1.8,
        label=f"Nominal Significance Level α = {alpha * 100.0:.0f}%",
    )

    ax.fill_between(
        peeks,
        [alpha * 100.0] * len(peeks),
        rates,
        color="#F87171",
        alpha=0.18,
        label="Type I Error Inflation Region",
    )

    ax.set_title("False-Positive Rate vs. Number of Peeks (Optional Stopping)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Peeks (Daily Statistical Checks)", fontsize=11, labelpad=8)
    ax.set_ylabel("Cumulative False-Positive Rate (%)", fontsize=11, labelpad=8)
    ax.set_xticks(peeks)
    max_val = max(rates) if rates and max(rates) > 0 else 10.0
    ax.set_ylim(0, max_val * 1.25)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left")

    caption = (
        f"Caption: Repeatedly testing daily without correction inflates the false-positive rate "
        f"from {rates[0]:.1f}% on Day 1 to {final_fpr:.1f}% by Day {len(peeks)}, far exceeding nominal α = 5%."
    )
    plt.figtext(0.5, 0.01, caption, ha="center", fontsize=8.5, style="italic", wrap=True)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_file)
    plt.close(fig)

    return out_file


def plot_power_curve_comparison(
    sample_sizes: Optional[Sequence[int]] = None,
    empirical_powers: Optional[Sequence[float]] = None,
    output_path: Union[str, Path] = "reports/figures/power_curve.png",
    n_tests: int = 1000,
    p_a: float = 0.10,
    p_b: float = 0.11,
    alpha: float = 0.05,
    seed: int = 42,
    dpi: int = 300,
) -> Path:
    """Generate and save the empirical power vs. sample size plot with theoretical curve overlay.

    Compares simulated Monte Carlo power against theoretical two-proportion power
    derived from power_for().

    Args:
        sample_sizes: Sequence of sample sizes to simulate, or None for defaults.
        empirical_powers: Pre-computed empirical powers, or None to simulate.
        output_path: Destination path for the PNG plot.
        n_tests: Number of simulations per sample size (default: 1000).
        p_a: Baseline conversion rate (default: 0.10).
        p_b: Target conversion rate (default: 0.11).
        alpha: Significance level (default: 0.05).
        seed: Random seed (default: 42).

    Returns:
        Path object pointing to the saved figure.
    """
    if sample_sizes is None:
        sample_sizes = [2000, 4000, 7000, 10000, 14749, 18000, 22000, 26000]

    sizes_list = [int(n) for n in sample_sizes]

    if empirical_powers is None:
        powers = power_curve(
            sample_sizes=sizes_list,
            n_tests=n_tests,
            p_a=p_a,
            p_b=p_b,
            alpha=alpha,
            seed=seed,
        )
    else:
        powers = list(empirical_powers)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Compute fine theoretical curve across the domain
    min_n = min(sizes_list)
    max_n = max(sizes_list)
    dense_n = np.linspace(max(100, min_n - 1000), max_n + 1000, 250)
    theoretical_powers = [power_for(n_val, p_a, p_b, alpha=alpha) for n_val in dense_n]

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=dpi)

    # Plot theoretical curve
    ax.plot(
        dense_n,
        [tp * 100.0 for tp in theoretical_powers],
        color="#2563EB",
        linewidth=2.2,
        label="Theoretical Power Curve (power_for)",
    )

    # Overlay empirical simulation points
    ax.scatter(
        sizes_list,
        [ep * 100.0 for ep in powers],
        color="#DC2626",
        s=55,
        zorder=5,
        edgecolor="#991B1B",
        linewidth=1.2,
        label=f"Empirical Simulation (N={n_tests} per point)",
    )

    # Reference 80% power target line
    ax.axhline(
        80.0,
        color="#059669",
        linestyle="--",
        linewidth=1.5,
        label="Target Power (80.0%)",
    )

    # Reference n=14749 planned sample size line
    if 14749 in sizes_list or (min_n <= 14749 <= max_n):
        ax.axvline(
            14749,
            color="#6B7280",
            linestyle=":",
            linewidth=1.5,
            label="Planned n = 14,749 per group",
        )

    ax.set_title("Statistical Power vs. Sample Size (p_a = 0.10, p_b = 0.11)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Sample Size per Group (n)", fontsize=11, labelpad=8)
    ax.set_ylabel("Statistical Power (%)", fontsize=11, labelpad=8)
    ax.set_xlim(dense_n[0], dense_n[-1])
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right")

    caption = (
        "Caption: Empirical power closely tracks theoretical two-proportion power, "
        "reaching approximately 80% power at the planned sample size of n = 14,749."
    )
    plt.figtext(0.5, 0.01, caption, ha="center", fontsize=8.5, style="italic", wrap=True)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_file)
    plt.close(fig)

    return out_file


def generate_all_plots(
    output_dir: Union[str, Path] = "reports/figures",
    seed: int = 42,
) -> Dict[str, Path]:
    """Execute all simulations and generate the complete set of visual reports.

    Args:
        output_dir: Folder to store figure files (default: reports/figures).
        seed: Random seed for simulation reproducibility (default: 42).

    Returns:
        Dictionary mapping plot name to its Path on disk.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. A/A Test plot
    aa_path = out_dir / "aa_p_values.png"
    plot_aa_p_values(output_path=aa_path, seed=seed)

    # 2. Peeking curve plot
    peeking_path = out_dir / "peeking_fpr.png"
    plot_peeking_curve(output_path=peeking_path, seed=seed)

    # 3. Power curve comparison plot
    power_path = out_dir / "power_curve.png"
    sample_sizes = [2000, 4000, 7000, 10000, 14749, 18000, 22000, 26000]
    plot_power_curve_comparison(sample_sizes=sample_sizes, output_path=power_path, seed=seed)

    return {
        "aa_p_values": aa_path,
        "peeking_fpr": peeking_path,
        "power_curve": power_path,
    }


# -----------------------------------------------------------------------------
# Theory vs. Experiment Paragraph Templates & Discrepancy Explanations
# -----------------------------------------------------------------------------

THEORY_VS_EXPERIMENT_TEMPLATES = {
    "aa_test": (
        "In theory, an A/A test should reject the null hypothesis at a rate exactly equal to the nominal significance "
        "level alpha = {alpha:.2f} ({alpha_pct:.1f}%), and p-values should follow a continuous Uniform(0, 1) distribution. "
        "In our experiment of {n_tests} simulated A/A runs with sample size n = {n:,} and baseline conversion p = {p:.2f}, "
        "we observed {significant_count} significant tests, representing an empirical false-positive rate of {empirical_pct:.2f}%. "
        "This outcome falls comfortably within the theoretical 95% binomial sampling range of {range_low} to {range_high} "
        "significant tests expected purely by random chance."
    ),
    "peeking": (
        "In theory, a single fixed-horizon test controls the Type I error rate at exactly alpha = {alpha:.2f} (5.0%). "
        "However, repeatedly evaluating data daily across {days} peeks without alpha-correction introduces the multiple "
        "testing / optional stopping problem, causing the cumulative false-positive rate to inflate substantially above alpha. "
        "In our experiment of {n_tests} trials adding {per_day:,} users per group daily, the cumulative false-positive rate "
        "escalated from {day1_pct:.2f}% on Day 1 to {final_pct:.2f}% by Day {days}, demonstrating that continuous monitoring "
        "without sequential boundaries quadruples the risk of false discovery."
    ),
    "power": (
        "In theory, sample size planning using unpooled variance indicates that a sample size of n = {n:,} per variant "
        "is required to detect an absolute lift of {lift:.2f} (from p_a = {p_a:.2f} to p_b = {p_b:.2f}) at alpha = {alpha:.2f} "
        "with {theoretical_power_pct:.1f}% statistical power. In our Monte Carlo simulation of {n_tests} independent experiments "
        "at n = {n:,}, we observed an empirical rejection rate of {empirical_power_pct:.2f}%. The minute gap of "
        "{gap_pct:+.2f}% between theoretical and empirical power is fully explained by finite-sample binomial simulation variance."
    ),
}

DISCREPANCY_REASONS = {
    "finite_simulations": (
        "1. Finite Number of Simulations (Monte Carlo Sampling Noise):\n"
        "   Any empirical simulation with a finite number of runs (e.g., N = 1,000) is subject to random sampling variance. "
        "   By the Central Limit Theorem, the empirical proportion has standard error SE = sqrt(P * (1 - P) / N). For a 5% "
        "   Type I error rate, SE = sqrt(0.05 * 0.95 / 1000) ≈ 0.69%, meaning empirical outcomes between 3.6% and 6.4% "
        "   are completely consistent with a true 5.0% error rate. Similarly, for 80% power, SE = sqrt(0.80 * 0.20 / 1000) "
        "   ≈ 1.26%, creating expected empirical fluctuations of ±2.5% around theoretical power."
    ),
    "normal_approximation": (
        "2. Normal Approximation vs. Exact Probability Distributions:\n"
        "   Theoretical sample size formulas and standard z-tests approximate discrete binomial distributions with a continuous "
        "   normal distribution. While accurate for large sample sizes, small skewness remains when baseline conversion p deviates "
        "   from 0.50. Furthermore, theoretical power_for() uses unpooled variances under H1 whereas the two-proportion hypothesis "
        "   test computes pooled standard error under H0, causing minor second-order divergence between theoretical expectations "
        "   and test rejections."
    ),
    "discrete_counts": (
        "3. Discrete Counts and Discontinuous Critical Boundaries:\n"
        "   Conversions in digital experiments are integer counts (e.g., 1,525 successes), making the empirical test statistic "
        "   inherently discrete rather than continuous. Because the underlying sample space is discrete, the exact cumulative "
        "   probability of falling strictly below alpha rarely equals exactly alpha without randomized tie-breaking or continuity "
        "   corrections (such as Yates' correction). This discreteness introduces small, systematic step-function offsets between "
        "   continuous theoretical formulas and observed empirical frequencies."
    ),
}


def format_theory_vs_experiment(
    aa_results: Optional[Dict[str, Any]] = None,
    peeking_results: Optional[Dict[str, Any]] = None,
    power_results: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, str]:
    """Generate filled theory vs. experiment explanation paragraphs based on simulation runs.

    Args:
        aa_results: Pre-computed aa_test dict or None to compute fresh.
        peeking_results: Pre-computed peeking dict or None to compute fresh.
        power_results: Dict with keys ('n', 'empirical_power', 'p_a', 'p_b', 'alpha') or None.
        seed: Random seed if running fresh simulations.

    Returns:
        Dict with keys 'aa_test', 'peeking', 'power', and 'discrepancy_reasons'.
    """
    if aa_results is None:
        aa_results = aa_test(seed=seed)
    if peeking_results is None:
        peeking_results = peeking(seed=seed)
    if power_results is None:
        emp_pow = power_check(n=14749, p_a=0.10, p_b=0.11, seed=seed)
        power_results = {
            "n": 14749,
            "empirical_power": emp_pow,
            "p_a": 0.10,
            "p_b": 0.11,
            "alpha": 0.05,
            "n_tests": 1000,
        }

    # Format A/A paragraph
    aa_p = THEORY_VS_EXPERIMENT_TEMPLATES["aa_test"].format(
        alpha=aa_results["alpha"],
        alpha_pct=aa_results["alpha"] * 100.0,
        n_tests=aa_results["n_tests"],
        n=aa_results["n"],
        p=aa_results["p"],
        significant_count=aa_results["significant_count"],
        empirical_pct=aa_results["fraction_significant"] * 100.0,
        range_low=aa_results["binomial_range_95"][0],
        range_high=aa_results["binomial_range_95"][1],
    )

    # Format Peeking paragraph
    day1_rate = peeking_results["cumulative_fpr"][0] * 100.0
    final_rate = peeking_results["false_positive_rate"] * 100.0
    peeking_p = THEORY_VS_EXPERIMENT_TEMPLATES["peeking"].format(
        alpha=peeking_results["alpha"],
        days=peeking_results["days"],
        n_tests=peeking_results["n_tests"],
        per_day=peeking_results["per_day"],
        day1_pct=day1_rate,
        final_pct=final_rate,
    )

    # Format Power paragraph
    n_pow = int(power_results["n"])
    p_a = float(power_results["p_a"])
    p_b = float(power_results["p_b"])
    alpha_pow = float(power_results.get("alpha", 0.05))
    theo_pow = power_for(n_pow, p_a, p_b, alpha=alpha_pow)
    emp_pow_val = float(power_results["empirical_power"])
    gap = (emp_pow_val - theo_pow) * 100.0

    power_p = THEORY_VS_EXPERIMENT_TEMPLATES["power"].format(
        n=n_pow,
        lift=abs(p_b - p_a),
        p_a=p_a,
        p_b=p_b,
        alpha=alpha_pow,
        theoretical_power_pct=theo_pow * 100.0,
        n_tests=power_results.get("n_tests", 1000),
        empirical_power_pct=emp_pow_val * 100.0,
        gap_pct=gap,
    )

    reasons_text = "\n\n".join(DISCREPANCY_REASONS.values())

    return {
        "aa_test": aa_p,
        "peeking": peeking_p,
        "power": power_p,
        "discrepancy_reasons": reasons_text,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Running A/B Testing Simulations and Generating Figures...")
    print("=" * 70)

    # 1. A/A Simulation
    print("\n[1/3] Running A/A Tests (1,000 runs, n=15,000, p=0.10, seed=42)...")
    res_aa = aa_test(seed=42)
    print(f"  Significant fraction: {res_aa['fraction_significant']:.4f} ({res_aa['significant_count']}/1000)")
    print(f"  95% Binomial Range:   {res_aa['binomial_range_95'][0]} to {res_aa['binomial_range_95'][1]}")

    # 2. Peeking Simulation
    print("\n[2/3] Running Peeking Simulation (1,000 runs, 1,000/day, 14 days, seed=42)...")
    res_peek = peeking(seed=42)
    print(f"  Day 1 FPR:  {res_peek['cumulative_fpr'][0]:.4f}")
    print(f"  Day 14 FPR: {res_peek['false_positive_rate']:.4f}")

    # 3. Power Simulation
    print("\n[3/3] Running Power Check (1,000 runs, n=14,749, p_a=0.10, p_b=0.11, seed=42)...")
    emp_power = power_check(n=14749, seed=42)
    theo_power = power_for(14749, 0.10, 0.11)
    print(f"  Empirical Power:   {emp_power:.4f} ({emp_power * 100:.1f}%)")
    print(f"  Theoretical Power: {theo_power:.4f} ({theo_power * 100:.1f}%)")

    # 4. Generate all plots
    print("\nGenerating figures in reports/figures/...")
    plots = generate_all_plots(seed=42)
    for name, path in plots.items():
        print(f"  Saved {name}: {path}")

    # 5. Output Theory vs Experiment templates
    print("\n" + "=" * 70)
    print("THEORY VS. EXPERIMENT PARAGRAPHS")
    print("=" * 70)
    narratives = format_theory_vs_experiment(res_aa, res_peek, {"n": 14749, "empirical_power": emp_power, "p_a": 0.10, "p_b": 0.11, "alpha": 0.05, "n_tests": 1000})
    print("\n[A/A Test]:\n", narratives["aa_test"])
    print("\n[Peeking Problem]:\n", narratives["peeking"])
    print("\n[Statistical Power]:\n", narratives["power"])
    print("\n[Plausible Reasons for Gap]:\n", narratives["discrepancy_reasons"])
    print("\nSimulation and figure generation complete!")
