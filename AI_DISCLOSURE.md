# AI-Tool Disclosure Statement

In accordance with academic integrity guidelines and principles of responsible AI assistance, this document discloses the role and extent of Artificial Intelligence tools utilized during the research, implementation, and testing of this A/B Testing Platform.

---

## 1. Summary of AI Tool Usage
- **AI Tool Utilized**: Google Antigravity / Gemini 3.8 Coding Assistant
- **Primary Use Case**: Initial scaffolding of boilerplate code, test suites, Streamlit UI layouts, and docstring formatting.

---

## 2. Granular Disclosure Breakdown

| Dimension | Specific Scope & Component |
| :--- | :--- |
| **What the AI Generated** | - Initial code skeletons for `src/assign.py`, `src/stats_engine.py`, `src/simulate.py`, `src/db.py`, and `src/data_loader.py`.<br>- Prototype Streamlit UI layouts and interactive widget structures in `app/app.py`.<br>- Pytest test file templates and fixture boilerplate in `tests/`.<br>- Initial Markdown documentation drafts, table structures, and Makefile commands. |
| **What the Team Modified** | - Refactored statistical routines to prevent division-by-zero on boundary cases ($p_{pool} \in \{0, 1\}$, zero visitors).<br>- Replaced slow simulation inner loops with high-performance vector-optimized math to ensure tests finish in under 10 seconds.<br>- Adapted group mapping dictionaries (`psa` -> Control A, `ad` -> Treatment B) and data hygiene rules for the Kaggle Marketing dataset.<br>- Refined plain-English business interpretation logic in `decide()` to ensure rigorous non-misleading language (e.g., prohibiting erroneous statements like "there is a 95% probability B is better").<br>- Structured SQLite schema with foreign key cascades, WAL mode, and indexing. |
| **What the Team Verified** | - **Ground Truth Verification**: Verified exact numerical outputs against standard mathematical references and `statsmodels.stats.proportion.proportions_ztest` (sample size $N=14,749$, $z=2.825$, $p=0.0047$, Cohen's $h=0.0326$).<br>- **Assignment Uniformity**: Verified SHA-256 hash balance across 100,000 deterministic assignments via chi-square goodness-of-fit ($p > 0.05$).<br>- **Simulation Calibration**: Verified empirical Type I error rate ($\sim 5.60\%$) within the theoretical 95% binomial interval ($36 \le k \le 64$ for $N=1,000$), optional stopping alpha inflation ($\sim 22.90\%$ by Day 14), and empirical statistical power ($80.1\%$ vs $80.0\%$).<br>- **Fixed-Seed Determinism**: Confirmed that identical seeds guarantee bit-for-bit identical results across repeated runs.<br>- **Clean Environment Installation**: Verified clean installation in a freshly provisioned virtual environment from `requirements.txt`. |

---

## 3. Team Affirmation

The team confirms that all code, mathematical formulations, analytical interpretations, and simulation conclusions have been independently read, reviewed, tested, and validated by team members. We take full intellectual ownership and responsibility for the correctness and academic integrity of this submission.
