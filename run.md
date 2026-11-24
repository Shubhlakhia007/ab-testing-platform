# Project Execution Guide (`run.md`)

This guide provides step-by-step commands to set up the environment, run unit and integration tests, launch the interactive Streamlit application, and execute all Monte Carlo simulations across Windows, macOS, and Linux.

---

## 1. Setup (`setup`)

Create a dedicated Python 3.10+ virtual environment and install dependencies:

### On Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip and install all required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### On macOS / Linux (Bash)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Run Tests (`test`)

Execute the complete pytest test suite covering assignment hashing, statistical testing, fast simulations (< 10 seconds), SQLite persistence, and dataset cleaning:

```bash
# Run pytest with verbose output
python -m pytest -v

# Run with duration profiling to confirm fast execution (< 10s)
python -m pytest tests/test_simulate.py --durations=0
```

All 36 unit tests verify analytical ground truth, mathematical invariants, deterministic seed reproducibility, and edge-case resilience.

---

## 3. Run the Interactive Dashboard (`run-app`)

Launch the multi-tab interactive Streamlit web application:

```bash
streamlit run app/app.py
```

- Accessible via browser at: **`http://localhost:8501`**
- Includes interactive tabs for:
  1. **Pre-Registration & Sizing Calculator**: Lock hypotheses with SHA-256 and compute sample sizes.
  2. **Live Traffic Simulator**: Generate synthetic visitor cohorts with deterministic assignments.
  3. **Data Quality & Ingestion**: Upload custom CSVs, clean nulls/duplicates, and audit mapping.
  4. **Analysis & Decision Engine**: Two-proportion z-tests, SRM goodness-of-fit, and automated SHIP/KEEP/INCONCLUSIVE calls.
  5. **Educational Labs**: Live interactive demonstrations of the Peeking Problem and A/A Uniformity.

---

## 4. Run Simulations & Generate Figures (`run-simulations`)

Regenerate all publication-grade figures (`reports/figures/`) and run the end-to-end dataset evaluation:

```bash
# Generate simulation figures (aa_p_values.png, peeking_fpr.png, power_curve.png)
python -c "from src.simulate import generate_all_plots; generate_all_plots(); print('All simulation figures updated in reports/figures/')"

# Execute Kaggle dataset analysis pipeline and generate reports/data_quality_log.md
python run_marketing_ab_test.py
```

Output figures generated in `reports/figures/`:
1. `aa_p_values.png`: Empirical uniform p-value histogram under $H_0$ with binomial confidence envelope.
2. `peeking_fpr.png`: Cumulative False Positive Rate inflation across 14 daily evaluations (peeking problem).
3. `power_curve.png`: Theoretical vs. empirical power comparison across sample sizes.
