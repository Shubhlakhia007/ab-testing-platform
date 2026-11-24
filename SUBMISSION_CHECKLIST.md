# Pre-Submission Verification Checklist

This checklist confirms that the repository satisfies all academic, technical, and reproducibility requirements prior to final submission.

---

| Item | Requirement / Component | Verification Status | Location / Reference |
| :--- | :--- | :---: | :--- |
| **1. README** | Project summary (3 lines), problem statement, folder map, install steps, how to run app, how to run tests, reproduction guide, and file dictionary. | **COMPLETE** | `README.md` |
| **2. requirements.txt** | Minimal, version-compatible dependencies verified in a clean virtual environment without conflicts. | **COMPLETE** | `requirements.txt` |
| **3. SQL Scripts** | Production-grade DDL table creation and aggregation queries for assignments, events, and metrics. | **COMPLETE** | `sql/schema.sql`, `sql/aggregations.sql` |
| **4. Notebooks / Analysis Pipeline** | Executable notebook-style analysis script with `# %%` cell support evaluating the Kaggle marketing dataset end-to-end. | **COMPLETE** | `run_marketing_ab_test.py` |
| **5. Reports & Artifacts** | Data quality logs, sample ratio mismatch checks, and 5 key methodology limitations. | **COMPLETE** | `reports/data_quality_log.md`, `reports/figures/` |
| **6. Figures & Presentation Slides** | High-resolution publication-grade figures (`aa_p_values.png`, `peeking_fpr.png`, `power_curve.png`) with academic captions. | **COMPLETE** | `reports/figures/` |
| **7. Fixed Seeds & Reproducibility** | Explicit deterministic seed parameters (`seed=42`) used across all generators with automated reproducibility unit tests. | **COMPLETE** | `tests/test_simulate.py`, `src/assign.py` |
| **8. Test Suite** | 36 unit/integration tests running in < 10 seconds verifying exact ground truth numbers across modules. | **COMPLETE** | `tests/` |
| **9. AI Tool Disclosure** | Granular statement disclosing AI generation, team modifications, and verified results. | **COMPLETE** | `AI_DISCLOSURE.md` |
| **10. Instructor Repo Access** | Clean repository free of unwanted build artifacts, gitignore configured, permissions verified for instructor access. | **COMPLETE** | `.gitignore`, GitHub Settings |

---

## Final Verification Instructions for the Team
Before submitting your repository link to the portal:
1. Push all latest changes:
   ```bash
   git add .
   git commit -m "chore: finalize submission repository with tests, documentation, and checklist"
   git push origin main
   ```
2. Verify GitHub / GitLab repository visibility:
   - Ensure the repository is set to **Public** OR that the instructor's username has been added as a **Collaborator** under `Settings > Collaborators`.
3. Confirm that the cloning command works from an external terminal:
   ```bash
   git clone <REPO_URL>
   cd maths
   pip install -r requirements.txt
   pytest -v
   ```
