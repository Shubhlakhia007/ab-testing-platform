"""Unit tests for deterministic assignment and hashing balance.

Can be run via pytest (pytest tests/test_assign.py) or directly (python tests/test_assign.py).
"""

import math
import os
import sqlite3
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import pytest
except ImportError:
    pytest = None

from src.assign import assign, assign_and_store, balance_report


def test_deterministic_assignment() -> None:
    """Verify that assign(user_id, experiment_name) is pure and deterministic."""
    val_first = assign(42, "x")
    assert val_first in ("A", "B"), f"Variant must be 'A' or 'B', got {val_first}"
    for _ in range(10):
        assert assign(42, "x") == val_first, "assign() returned inconsistent variants for identical inputs"


def test_ground_truth_balance_fee_top() -> None:
    """Verify the exact ground-truth balance for 100,000 users in 'fee_top'."""
    report = balance_report("fee_top", n_users=100000)
    assert report["counts"]["A"] == 49981, f"Expected 49981 for variant A, got {report['counts']['A']}"
    assert report["counts"]["B"] == 50019, f"Expected 50019 for variant B, got {report['counts']['B']}"
    assert math.isclose(report["percentages"]["A"], 49.981, abs_tol=1e-4)
    assert math.isclose(report["percentages"]["B"], 50.019, abs_tol=1e-4)
    assert report["p_value"] > 0.05, f"P-value {report['p_value']} indicates significant imbalance"


def test_experiment_name_salting_isolation() -> None:
    """Verify that including experiment name decouples assignments for the same user."""
    # User 42 gets 'A' in 'x' and 'B' in 'fee_top'
    var_x = assign(42, "x")
    var_fee_top = assign(42, "fee_top")
    assert var_x == "A", f"Expected 'A' for user 42 in experiment 'x', got {var_x}"
    assert var_fee_top == "B", f"Expected 'B' for user 42 in experiment 'fee_top', got {var_fee_top}"
    assert var_x != var_fee_top, "Experiment salting failed to isolate assignments"


def test_assign_and_store_persistence() -> None:
    """Verify DB persistence and idempotency of assign_and_store."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE assignments (
            user_id INTEGER,
            experiment_id INTEGER,
            variant TEXT CHECK (variant IN ('A', 'B')),
            assigned_at TIMESTAMP,
            PRIMARY KEY (user_id, experiment_id)
        )
        """
    )
    conn.commit()

    # Initial assignment
    assigned = assign_and_store(conn, user_id=101, experiment_id=1, experiment_name="fee_top")
    assert assigned in ("A", "B")

    # Second call must retrieve identical assignment without overwriting
    second = assign_and_store(conn, user_id=101, experiment_id=1, experiment_name="fee_top")
    assert second == assigned

    # Verify directly from DB
    cursor.execute("SELECT variant FROM assignments WHERE user_id = ? AND experiment_id = ?", (101, 1))
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == assigned

    conn.close()


if __name__ == "__main__":
    test_deterministic_assignment()
    print("test_deterministic_assignment: PASSED")
    test_ground_truth_balance_fee_top()
    print("test_ground_truth_balance_fee_top: PASSED")
    test_experiment_name_salting_isolation()
    print("test_experiment_name_salting_isolation: PASSED")
    test_assign_and_store_persistence()
    print("test_assign_and_store_persistence: PASSED")
    print("All tests passed successfully!")
