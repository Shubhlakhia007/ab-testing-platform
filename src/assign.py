"""Deterministic user assignment module for A/B testing.

Implements hashing-based variant assignment using SHA-256 to ensure
statistically balanced and consistent user experiences across sessions.
"""

from datetime import datetime, timezone
import hashlib
import sqlite3
from typing import Any, Dict

from scipy.stats import chisquare


def assign(user_id: int, experiment_name: str) -> str:
    """Deterministically assign a user to variant 'A' or 'B'.

    Constructs a composite key of '{experiment_name}-{user_id}', hashes it
    with SHA-256, converts the digest to an integer, and returns 'A' if even
    or 'B' if odd.
    """
    salt_key = f"{experiment_name}-{user_id}"
    digest = hashlib.sha256(salt_key.encode("utf-8")).hexdigest()
    hash_int = int(digest, 16)
    return "A" if hash_int % 2 == 0 else "B"


def assign_and_store(
    conn: sqlite3.Connection,
    user_id: int,
    experiment_id: int,
    experiment_name: str,
) -> str:
    """Assign user to a variant and store record in the SQLite database.

    If an assignment already exists for the given (user_id, experiment_id),
    returns the stored variant without modifying the database. Otherwise,
    computes the variant deterministically, inserts the record, and commits.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT variant FROM assignments WHERE user_id = ? AND experiment_id = ?",
        (user_id, experiment_id),
    )
    existing = cursor.fetchone()
    if existing is not None:
        return str(existing[0])

    variant = assign(user_id, experiment_name)
    now_iso = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        "INSERT INTO assignments (user_id, experiment_id, variant, assigned_at) VALUES (?, ?, ?, ?)",
        (user_id, experiment_id, variant, now_iso),
    )
    conn.commit()
    return variant


def balance_report(experiment_name: str, n_users: int = 100000) -> Dict[str, Any]:
    """Evaluate assignment balance for user IDs 1..n_users using a Chi-Square test.

    Computes counts and percentages for variants A and B, followed by a
    goodness-of-fit test against a theoretical 50/50 uniform distribution.
    """
    counts = {"A": 0, "B": 0}
    for uid in range(1, n_users + 1):
        variant = assign(uid, experiment_name)
        counts[variant] += 1

    pct_a = (counts["A"] / n_users) * 100.0
    pct_b = (counts["B"] / n_users) * 100.0

    expected = [n_users / 2.0, n_users / 2.0]
    observed = [counts["A"], counts["B"]]
    chi2_result = chisquare(f_obs=observed, f_exp=expected)

    return {
        "counts": counts,
        "percentages": {"A": pct_a, "B": pct_b},
        "chi2": float(chi2_result.statistic),
        "p_value": float(chi2_result.pvalue),
    }
