# How the A/B Testing Platform Works

This document describes the architecture, data flows, database schema, statistical concepts, and execution commands for the A/B Testing & Statistical Simulation Platform, derived directly from the current codebase.

---

## 1. Module-by-Module Summary

| File | Core Functions & Responsibilities |
| :--- | :--- |
| `src/assign.py` | Deterministic SHA-256 variant assignment: `assign(user_id, experiment_name)`, `assign_and_store(conn, user_id, experiment_id, experiment_name)`, and `balance_report(experiment_name, n_users)`. |
| `src/stats_engine.py` | Frequentist statistics: `sample_size(p1, p2, alpha, power)`, `two_prop_test(x_a, n_a, x_b, n_b, alpha)`, `decide(result, planned_n_per_group, n_a, n_b)`, `sample_ratio_check(n_a, n_b)`, and `power_for(n_per_group, p1, p2, alpha)`. |
| `src/simulate.py` | Monte Carlo simulations: `simulate_experiment(...)`, `aa_test(...)`, `peeking(...)`, `power_check(...)`, `power_curve(...)`, and matplotlib plotting helpers such as `plot_aa_p_values(...)`. |
| `src/db.py` | SQLite persistence & pre-registration: `get_connection`, `init_db`, `hash_hypothesis`, `verify_experiment_hash`, `create_experiment`, `get_experiment`, `get_experiment_by_name`, `list_experiments`, `log_event`, `log_events_batch`, `store_assignments_batch`, `clear_experiment_data`, and `get_counts`. |
| `src/data_loader.py` | External CSV pipeline: `inspect_data`, `_parse_conversion_value`, `parse_conversion_series`, `prepare_csv_for_experiment`, `write_data_quality_log`, `clean_data`, `map_groups`, `run_ab_analysis`, and `load_clean_and_test`. |
| `app/app.py` | Five-tab interactive Streamlit dashboard (`PAGES` = `1. Plan & Sample Size`, `2. Data & Simulation`, `3. Statistical Verdict`, `4. Simulation Labs`, `5. SHA-256 Demo`). |
| `run_marketing_ab_test.py` | CLI script (`parse_args`, `main`) that runs the `data_loader` + `stats_engine` pipeline on a Marketing A/B CSV (`--csv` or default paths, falling back to `data/marketing_AB_synthetic.csv`). |

---

## 2. Data Flow Across the Five Tabs

1. **PLAN (`1. Plan & Sample Size`)**
   - Enter experiment `name`, `hypothesis`, `success_rule`, `baseline_rate` ($p_1$), `mde`, `alpha`, and `power`.
   - `sample_size()` computes the required sample size per variant live.
   - Clicking **Lock Hypothesis & Register Experiment** calls `create_experiment()`, which computes a SHA-256 digest via `hash_hypothesis()` over a canonical JSON (sorted keys) of all 8 plan fields (`name`, `hypothesis`, `success_rule`, `baseline_rate`, `mde`, `alpha`, `power`, `planned_n_per_group`) and stores the record in `experiments`.
   - Locked experiments display the stored `hypothesis_hash` and a **Plan integrity: verified** indicator checked via `verify_experiment_hash()`.

2. **DATA (`2. Data & Simulation`)**
   - **Option A (Simulate Visitors):** Generates user IDs `1..2N`, assigns variants via `assign()` / `assign_and_store()`, and logs `view_checkout` plus simulated `order_complete` events into SQLite using `store_assignments_batch()` and `log_events_batch()`.
   - **Option B (Upload CSV / Sample Dataset):** Reads a CSV with `pd.read_csv(..., low_memory=False)`. Group labels are stripped of leading/trailing whitespace, outcomes are parsed case-insensitively using `_parse_conversion_value` via `parse_conversion_series()` / `prepare_csv_for_experiment()`, and an optional **User ID Column** selector drops duplicate user IDs (defaulting to row numbers when omitted). Assignments and events are built with vectorized `zip` + `itertools.repeat` and batch-inserted; any skipped rows (blank values, other groups, unparseable outcomes, duplicate user IDs) are counted and reported.

3. **VERDICT (`3. Statistical Verdict`)**
   - Calls `get_counts(conn, experiment_id)` to aggregate `n_a`, `x_a`, `n_b`, `x_b` via SQL.
   - Runs `two_prop_test()`, `sample_ratio_check()`, and `decide()`.
   - Displays conversion rates, lift, two-tailed p-value, unpooled Wald confidence interval chart, Cohen's h, SRM and normality assumption checks, an early-stopping peeking warning if `min(n_a, n_b) < planned_n_per_group`, and the `SHIP_B` / `KEEP_A` / `INCONCLUSIVE` decision card.

4. **LEARN (`4. Simulation Labs`)**
   - Standalone Monte Carlo labs (no locked experiment required):
     - **A/A Test Calibration** (`aa_test`): plots the p-value histogram and checks that the false-positive rate falls within the 95% binomial range around $\alpha = 0.05$.
     - **The Peeking Problem** (`peeking`): plots cumulative false-positive rate inflation across daily peeks.

5. **DEMO (`5. SHA-256 Demo`)**
   - Previews `assign(visitor_id, experiment_name)` in real time for any integer visitor ID.
   - A checkbox **Record demo visitors in this experiment's data** defaults to **OFF**, keeping the tab in preview-only mode with zero database writes.
   - When turned **ON**, `view_checkout` is logged once per `(experiment_id, visitor_id)` per session (tracked in `st.session_state["demo_logged_views"]`) rather than on every Streamlit rerun, and clicking the purchase button logs `order_complete`.

---

## 3. How Deterministic Assignment Works

`assign(user_id, experiment_name)` in `src/assign.py` guarantees that the same user always sees the same variant within an experiment while remaining independent across experiments:

1. Construct the salted key `f"{experiment_name}-{user_id}"`.
2. Compute its SHA-256 hex digest.
3. Convert the hex digest to an integer (`int(digest, 16)`).
4. Return `"A"` if `hash_int % 2 == 0` (even), or `"B"` if odd.

---

## 4. Database Schema (`sql/schema.sql` & `src/db.py`)

SQLite is configured with `PRAGMA foreign_keys = ON`, `journal_mode = WAL`, and `synchronous = NORMAL`.

- **`experiments`**: Stores pre-registered experiment metadata (`id`, `name` UNIQUE, `hypothesis`, `hypothesis_hash`, `success_rule`, `baseline_rate`, `mde`, `alpha`, `power`, `planned_n_per_group`, `locked_at`, `created_at`).
- **`assignments`**: Stores user-to-variant mappings (`user_id`, `experiment_id`, `variant` in `('A', 'B')`, `assigned_at`) with `PRIMARY KEY (user_id, experiment_id)`, foreign key to `experiments(id) ON DELETE CASCADE`, and index `idx_assignments_exp_var (experiment_id, variant)`.
- **`events`**: Stores interaction and conversion events (`id`, `user_id`, `experiment_id`, `event_name`, `created_at`) with foreign key to `experiments(id) ON DELETE CASCADE` and index `idx_events_exp_user_name (experiment_id, user_id, event_name)`.
  - `get_counts()` joins `assignments` with `events` and counts distinct `user_id`s with `event_name = 'order_complete'` as conversions.

---

## 5. Glossary

- **Baseline rate ($p_1$)**: Expected conversion rate of the control group (Variant A) before applying any treatment.
- **MDE (Minimum Detectable Effect)**: Smallest absolute difference $|p_2 - p_1|$ the experiment is powered to detect.
- **Alpha ($\alpha$)**: Significance level / Type I error rate — maximum acceptable probability of a false positive when $H_0$ is true (commonly `0.05`).
- **Power ($1 - \beta$)**: Probability of correctly rejecting $H_0$ when the true difference equals the MDE (commonly `0.80`).
- **Null hypothesis ($H_0$)**: The assumption that Variant A and Variant B have equal true conversion rates ($p_A = p_B$).
- **p-value**: Probability of observing a test statistic at least as extreme as the sample result assuming $H_0$ is true (not the probability that B is better than A).
- **Confidence interval**: Range of plausible values for the true difference $p_B - p_A$ (computed here as an unpooled Wald interval).
- **Cohen's h**: Standardized effect size between two proportions: $2(\arcsin\sqrt{p_B} - \arcsin\sqrt{p_A})$.
- **SRM (Sample Ratio Mismatch)**: Significant deviation from the expected 50/50 split tested via Chi-Square Goodness-of-Fit (`sample_ratio_check`, flagged when $p < 0.001$).
- **Peeking**: Repeatedly testing accumulating data and stopping early when $p < \alpha$, which inflates the Type I error rate above nominal $\alpha$.
- **A/A test**: An experiment comparing two identical variants ($p_A = p_B$) to verify uniform p-values and calibrated false-positive rates.

---

## 6. How to Run

Run all commands from the project root (`maths/`):

```bash
# Launch the interactive Streamlit app
streamlit run app/app.py

# Run the test suite
pytest

# Run the Marketing A/B pipeline (optional --csv path)
python run_marketing_ab_test.py
python run_marketing_ab_test.py --csv path/to/marketing_AB.csv
```
