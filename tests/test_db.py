"""Unit tests for SQLite database layer (src/db.py)."""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.assign import assign_and_store
from src.db import (
    clear_experiment_data,
    create_experiment,
    get_counts,
    get_experiment,
    hash_hypothesis,
    init_db,
    log_event,
    log_events_batch,
    store_assignments_batch,
    verify_experiment_hash,
)


def test_db_lifecycle_and_counts(tmp_path):
    db_file = str(tmp_path / "test.db")
    conn = init_db(db_file)

    exp_id = create_experiment(
        conn=conn,
        name="test_checkout_button",
        hypothesis="Changing button color increases conversions",
        success_rule="B beats A with p < 0.05",
        baseline_rate=0.10,
        mde=0.01,
        alpha=0.05,
        power=0.80,
        planned_n_per_group=14749,
    )
    assert exp_id > 0

    exp = get_experiment(conn, exp_id)
    assert exp is not None
    assert exp["name"] == "test_checkout_button"
    assert len(exp["hypothesis_hash"]) == 64

    # Assign users and log events
    v1 = assign_and_store(conn, user_id=1, experiment_id=exp_id, experiment_name="test_checkout_button")
    v2 = assign_and_store(conn, user_id=2, experiment_id=exp_id, experiment_name="test_checkout_button")

    log_event(conn, user_id=1, experiment_id=exp_id, event_name="view_checkout")
    log_event(conn, user_id=1, experiment_id=exp_id, event_name="order_complete")

    counts = get_counts(conn, exp_id)
    assert counts["total_users"] == 2
    assert counts["total_conversions"] == 1

    # Test clear
    clear_experiment_data(conn, exp_id)
    cleared_counts = get_counts(conn, exp_id)
    assert cleared_counts["total_users"] == 0
    assert cleared_counts["total_conversions"] == 0

    conn.close()


def test_hypothesis_hash_sensitivity_and_verification(tmp_path):
    """Verify hash_hypothesis covers all plan fields and verify_experiment_hash detects tampering."""
    base_kwargs = dict(
        hypothesis_text="Button color lifts checkout",
        name="exp_v1",
        success_rule="Ship B if p < 0.05",
        baseline_rate=0.10,
        mde=0.01,
        alpha=0.05,
        power=0.80,
        planned_n_per_group=14749,
    )
    h1 = hash_hypothesis(**base_kwargs)
    h1_repeat = hash_hypothesis(**base_kwargs)
    assert h1 == h1_repeat

    # Changing alpha or success_rule must change the fingerprint
    h_alpha = hash_hypothesis(**{**base_kwargs, "alpha": 0.01})
    h_rule = hash_hypothesis(**{**base_kwargs, "success_rule": "Ship B if p < 0.01"})
    assert h_alpha != h1
    assert h_rule != h1

    # Stored experiment verifies cleanly, and fails if modified
    conn = init_db(str(tmp_path / "verify.db"))
    exp_id = create_experiment(
        conn=conn,
        name="exp_v1",
        hypothesis="Button color lifts checkout",
        success_rule="Ship B if p < 0.05",
        baseline_rate=0.10,
        mde=0.01,
        alpha=0.05,
        power=0.80,
        planned_n_per_group=14749,
    )
    exp = get_experiment(conn, exp_id)
    assert exp is not None
    assert verify_experiment_hash(exp) is True

    tampered = dict(exp, alpha=0.10)
    assert verify_experiment_hash(tampered) is False
    conn.close()
