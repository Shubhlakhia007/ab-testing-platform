# Comprehensive A/B Testing & Statistical Simulation Platform

A production-grade Python framework and interactive Streamlit application designed for rigorous hypothesis testing and experiment pre-registration.
Integrates SHA-256 deterministic variant assignment, robust SQLite event tracking, and two-proportion z-tests validated against Statsmodels.
Features Monte Carlo simulation engines demonstrating A/A uniform p-value calibration, optional stopping alpha inflation, and empirical power curves.

See [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for a detailed module-by-module guide, database schema, data flows, and glossary.

---

## 1. Problem Statement

Modern digital product teams frequently make critical deployment decisions based on online experiments. However, standard experimentation pipelines often suffer from four prevalent methodological pitfalls:

1. **Pre-Registration Failure & p-Hacking**: Hypotheses, planned sample sizes, and decision rules are modified post-hoc after inspecting early data.
2. **Assignment Drift & Non-Determinism**: Flawed randomisation mechanisms or non-salted user bucketing lead to user re-assignment leaks across sessions and inter-experiment interference.
3. **The Peeking Problem (Optional Stopping)**: Repeatedly evaluating $p$-values on daily cumulative cohorts without statistical correction dramatically inflates the empirical False Positive Rate (Type I error) far beyond the nominal $\alpha = 0.05$.
4. **Sample Ratio Mismatch (SRM) Ignorance**: Teams fail to detect assignment integrity breaches caused by tracking failures, bot filtering, or redirect latency, leading to severely biased conclusions.

This repository provides an end-to-end mathematical, educational, and computational solution to these challenges through pre-registration locking, deterministic salted SHA-256 assignment, automated SRM detection, robust hypothesis testing, and interactive simulation labs.

---

## 2. Repository Folder Map

```text
maths/
├── .gitignore                      # Git exclusion rules for caches, venvs, and databases
├── AI_DISCLOSURE.md                # Granular AI-tool assistance disclosure statement
├── Makefile                        # Build automation targets (setup, test, run-app, run-simulations)
├── README.md                       # Master project overview, reproduction guide, and file dictionary
├── SUBMISSION_CHECKLIST.md         # Pre-submission verification checklist
├── requirements.txt                # Verified Python package dependencies
├── run.md                          # Platform-specific step-by-step execution guide
├── run_marketing_ab_test.py        # End-to-end dataset analysis pipeline (notebook-style # %%)
│
├── app/
│   └── app.py                      # Multi-tab interactive Streamlit web dashboard
│
├── data/
│   └── marketing_AB_synthetic.csv  # Generated synthetic fallback dataset (when real CSV is absent)
│
├── reports/
│   ├── data_quality_log.md         # Generated audit log tracking data hygiene and drops
│   └── figures/
│       ├── aa_p_values.png         # Figure 1: Uniform p-values in A/A simulations
│       ├── peeking_fpr.png         # Figure 2: Peeking problem cumulative FPR inflation curve
│       └── power_curve.png         # Figure 3: Empirical vs. theoretical statistical power curve
│
├── sql/
│   ├── aggregations.sql            # SQL queries for visitor and conversion aggregation
│   └── schema.sql                  # SQLite DDL schema (experiments, assignments, events)
│
├── src/
│   ├── __init__.py                 # Package marker and public module exports
│   ├── assign.py                   # Deterministic SHA-256 assignment and chi-square balance report
│   ├── data_loader.py              # Flexible CSV ingestion, hygiene cleaner, and pipeline runner
│   ├── db.py                       # SQLite persistence layer, hypothesis hashing, and query engine
│   ├── simulate.py                 # Monte Carlo simulations (A/A tests, peeking, power) and plotting
│   └── stats_engine.py             # Sizing calculator, two-prop z-test, SRM test, and decision engine
│
└── tests/
    ├── __init__.py                 # Test package marker
    ├── test_assign.py              # Tests for deterministic assignment and hashing balance
    ├── test_data_loader.py         # Tests for data inspection, cleaning, mapping, and analysis
    ├── test_db.py                  # Tests for SQLite schema lifecycle, event logging, and queries
    ├── test_simulate.py            # Fast tests (< 10s) for simulations, plots, and fixed seed parity
    └── test_stats_engine.py        # Ground truth mathematical verification vs. Statsmodels
```

---

## 3. Installation Steps

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### Step-by-Step Setup
1. **Clone the repository:**
   ```bash
   git clone <REPO_URL>
   cd maths
   ```

2. **Create and activate a virtual environment:**
   - **On Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **On macOS / Linux (Bash):**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *(All dependencies have been verified for clean installation in an isolated virtual environment).*

---

## 4. How to Run the App

Launch the interactive multi-tab Streamlit dashboard:

```bash
streamlit run app/app.py
```

The application opens automatically in your browser at `http://localhost:8501` featuring:
1. **1. Plan & Sample Size**: Compute sample sizes ($N=14,749$ per arm for $\alpha=0.05, 1-\beta=0.80, p_1=0.10, p_2=0.11$) and lock the full experiment plan via canonical JSON SHA-256 hash.
2. **2. Data & Simulation**: Simulate synthetic user cohorts via deterministic SHA-256 assignment (Option A) or upload external CSV datasets with whitespace trimming, case-insensitive outcome parsing, and optional duplicate user ID removal (Option B).
3. **3. Statistical Verdict**: Run two-proportion z-tests, evaluate Sample Ratio Mismatch (SRM), check normality assumptions, and review plain-English recommendations (`SHIP_B`, `KEEP_A`, or `INCONCLUSIVE`).
4. **4. Simulation Labs**: Interactively reproduce the Peeking Problem and explore A/A test $p$-value uniform distributions.
5. **5. SHA-256 Demo**: Preview deterministic SHA-256 variant assignment for any visitor ID, with an opt-in checkbox (off by default) to record demo visitors into the active experiment.

---

## 5. How to Run Tests

Execute the comprehensive test suite using `pytest`:

```bash
# Run all unit and integration tests
python -m pytest -v

# Run with duration profiling to verify fast execution (< 10s)
python -m pytest tests/test_simulate.py --durations=0
```

### Verification Guarantees
- **Total Tests**: 36 passing tests.
- **Fast Runtime**: Simulation inner loops are vector-optimized to execute all tests in under 10 seconds.
- **Ground Truth Numbers**: Matches analytical benchmarks ($N=14,749$, $z=2.825$, $p=0.0047$, Cohen's $h=0.0326$).
- **Seed Parity Test**: Includes `test_identical_runs_with_same_seed_give_identical_results` confirming deterministic bit-level reproducibility when `seed=42`.

---

## 6. How to Reproduce Each Theory-vs-Experiment Result

### (a) A/A Testing: Uniform $p$-Values & Type I Error Calibration ($\alpha = 0.05$)
- **Theory**: When testing two identical variants ($p_A = p_B = 0.10$), the null hypothesis $H_0$ is true. The resulting $p$-values must follow a $\text{Uniform}(0, 1)$ distribution, and exactly $\alpha = 5\%$ of tests should reject $H_0$ within the 95% binomial sampling confidence interval $[36, 64]$ for 1,000 runs.
- **Reproduction Command**:
  ```bash
  python -c "from src.simulate import aa_test; res = aa_test(n_tests=1000, n=15000, p=0.10, alpha=0.05, seed=42); print(f'Significant: {res[\"significant_count\"]}/1000 ({res[\"fraction_significant\"]*100:.2f}%), 95% Binomial Bounds: {res[\"binomial_range_95\"]}')"
  ```
- **Ground Truth Output**: Exactly **56 / 1,000 (5.60%)** significant tests, inside $[36, 64]$.
- **Figure Generated**: Saved to `reports/figures/aa_p_values.png`.

---

### (b) The Peeking Problem: Optional Stopping $\alpha$-Inflation
- **Theory**: Evaluating cumulative cohorts daily over 14 days without adjusting $\alpha$ provides 14 opportunities to falsely reject $H_0$. By Day 14, the false positive rate inflates from nominal $\sim 5\%$ to over $20\%$.
- **Reproduction Command**:
  ```bash
  python -c "from src.simulate import peeking; res = peeking(n_tests=1000, per_day=1000, days=14, p=0.10, alpha=0.05, seed=42); print(f'Day 1 FPR: {res[\"cumulative_fpr\"][0]*100:.2f}%, Day 14 FPR: {res[\"false_positive_rate\"]*100:.2f}%')"
  ```
- **Ground Truth Output**: Day 1 FPR = **5.80%**, Day 14 Cumulative FPR = **22.90%**.
- **Figure Generated**: Saved to `reports/figures/peeking_fpr.png`.

---

### (c) Statistical Power: Empirical vs. Theoretical Curve
- **Theory**: At planned sample size $N = 14,749$ per arm ($p_A = 0.10, p_B = 0.11, \alpha = 0.05$), theoretical power is $80.00\%$. Empirical Monte Carlo simulations with 1,000 runs closely mirror this target within standard binomial sampling error.
- **Reproduction Command**:
  ```bash
  python -c "from src.stats_engine import power_for; from src.simulate import power_check; theo = power_for(14749, 0.10, 0.11); emp = power_check(n_tests=1000, n=14749, p_a=0.10, p_b=0.11, seed=42); print(f'Theoretical Power: {theo*100:.2f}%, Empirical Power: {emp*100:.2f}%')"
  ```
- **Ground Truth Output**: Theoretical Power = **80.00%**, Empirical Power = **80.10%** (difference $< 0.15\%$).
- **Figure Generated**: Saved to `reports/figures/power_curve.png`.

---

### (d) Real-World Kaggle Marketing A/B Dataset Pipeline
- **Theory**: Ingest the 588,829-row marketing dataset, clean dirty formats (string booleans, whitespace), confirm mapping (`psa` = Control A, `ad` = Treatment B), evaluate Sample Ratio Mismatch, and run two-proportion z-tests.
- **Reproduction Command**:
  ```bash
  python run_marketing_ab_test.py
  ```
- **Ground Truth Output**:
  - Sample sizes: Group A (PSA) $N_A = 23,524$, Group B (Ad) $N_B = 565,305$.
  - Conversion rates: PSA $\hat{p}_A \approx 1.785\%$, Ad $\hat{p}_B \approx 2.554\%$.
  - Result: Statistically significant positive lift ($p < 0.001$, `SHIP_B`).
  - Output Log: Written as a Markdown audit table to `reports/data_quality_log.md`.

---

## 7. File Dictionary: What Every File Does

| File Path | Functional Description & Core Responsibilities |
| :--- | :--- |
| `app/app.py` | Full multi-tab Streamlit web application integrating experiment registration, traffic simulation, SQLite storage, CSV upload, hypothesis analysis, and interactive educational labs. |
| `src/assign.py` | Deterministic variant assignment engine using SHA-256 (`{experiment_name}-{user_id}`), database persistence (`assign_and_store`), and chi-square assignment balance auditor (`balance_report`). |
| `src/stats_engine.py` | Classical frequentist statistical engine: analytical sample sizing (`sample_size`), two-proportion pooled z-tests (`two_prop_test`), SRM goodness-of-fit checks (`sample_ratio_check`), and plain-English business recommendations (`decide`). |
| `src/simulate.py` | High-performance Monte Carlo simulation engine: A/A testing (`aa_test`), optional stopping peeking analysis (`peeking`), empirical power curves (`power_curve`), and publication-grade matplotlib plotting. |
| `src/db.py` | SQLite database access layer: schema creation, experiment pre-registration with SHA-256 hypothesis locking, batch event logging, and SQL aggregation of conversion metrics. |
| `src/data_loader.py` | Dataset ingestion module: missing value and duplicate inspection (`inspect_data`), data hygiene and audit logging (`clean_data`), variant mapping (`map_groups`), and analysis orchestration. |
| `src/__init__.py` | Package root exporting the primary API functions and classes. |
| `run_marketing_ab_test.py` | Jupyter-compatible (`# %%`) standalone analysis script executing the complete pipeline on the Kaggle Marketing dataset and generating data quality logs. |
| `sql/schema.sql` | Production DDL SQL script defining relational tables for experiments, deterministic assignments, and interaction events with foreign keys and indexes. |
| `sql/aggregations.sql` | Standalone SQL aggregation queries computing unique visitors, conversions, and conversion rates grouped by experiment variant. |
| `tests/test_assign.py` | Pytest suite validating SHA-256 determinism, cross-experiment salt isolation, database persistence, and 100,000-user chi-square balance ground truth (49,981 vs 50,019). |
| `tests/test_stats_engine.py` | Pytest suite verifying exact analytical ground truth numbers ($N=14,749$, $z=2.825$, $p=0.0047$), Statsmodels agreement, edge cases (0 conversions, 0 users), and SRM detection. |
| `tests/test_simulate.py` | Fast pytest suite (< 10s) testing `aa_test` (56/1,000), `peeking` (22.90% FPR), power calibration (80.10%), figure generation, and identical-seed determinism. |
| `tests/test_db.py` | Pytest suite verifying SQLite database initialization, pre-registration hash generation, assignment idempotency, event logging, and metric aggregations. |
| `tests/test_data_loader.py` | Pytest suite verifying dataset inspection, dirty data cleaning, markdown quality log creation, variant mapping validation, and end-to-end analysis. |
| `tests/__init__.py` | Package initialization file for the pytest test suite. |
| `reports/data_quality_log.md` | Generated Markdown table documenting data cleaning operations, duplicate removals, and type conversions for Kaggle dataset evaluation. |
| `reports/figures/aa_p_values.png` | Publication-grade histogram showing the uniform distribution of p-values under $H_0$ with binomial confidence bounds. |
| `reports/figures/peeking_fpr.png` | Publication-grade line plot showing daily cumulative false positive rate inflation under optional stopping across 14 peeks. |
| `reports/figures/power_curve.png` | Publication-grade chart comparing empirical statistical power against the theoretical normal power curve across sample sizes. |
| `Makefile` | Standard build automation file providing simple targets: `setup`, `test`, `run-app`, `run-simulations`, and `clean`. |
| `run.md` | Comprehensive cross-platform command reference guide for Windows PowerShell, CMD, and Linux/macOS Bash. |
| `requirements.txt` | Verified list of Python package requirements with version constraints. |
| `AI_DISCLOSURE.md` | Detailed disclosure statement delineating AI-generated components, team modifications, and verified mathematical benchmarks. |
| `SUBMISSION_CHECKLIST.md` | Academic pre-submission checklist confirming readiness across code, data, tests, documentation, and permissions. |
| `.gitignore` | Exclusion specifications preventing compilation caches, virtual environments, and temporary databases from polluting git history. |
