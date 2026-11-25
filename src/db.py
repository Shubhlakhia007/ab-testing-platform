"""Database access layer for A/B testing platform.

Provides SQLite initialization, experiment lifecycle management, deterministic
assignment logging, event tracking, and SQL-based aggregation for A/B test results.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DB_PATH = os.path.join("data", "ab_test.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open or create a SQLite database connection with optimal PRAGMAs.

    Args:
        db_path: Filesystem path to the SQLite database file.

    Returns:
        sqlite3.Connection configured with foreign keys and WAL mode.
    """
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Initialize database schema including experiments, assignments, and events tables.

    Args:
        db_path: Filesystem path to the SQLite database file.

    Returns:
        Configured and initialized sqlite3.Connection.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            hypothesis TEXT NOT NULL,
            hypothesis_hash TEXT NOT NULL,
            success_rule TEXT,
            baseline_rate REAL NOT NULL,
            mde REAL NOT NULL,
            alpha REAL NOT NULL,
            power REAL NOT NULL,
            planned_n_per_group INTEGER NOT NULL,
            locked_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS assignments (
            user_id INTEGER NOT NULL,
            experiment_id INTEGER NOT NULL,
            variant TEXT CHECK (variant IN ('A', 'B')) NOT NULL,
            assigned_at TIMESTAMP NOT NULL,
            PRIMARY KEY (user_id, experiment_id),
            FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_assignments_exp_var
            ON assignments (experiment_id, variant);

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            experiment_id INTEGER NOT NULL,
            event_name TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_events_exp_user_name
            ON events (experiment_id, user_id, event_name);
        """
    )
    conn.commit()
    return conn


def hash_hypothesis(
    hypothesis_text: str,
    name: str = "",
    success_rule: str = "",
    baseline_rate: Optional[float] = None,
    mde: Optional[float] = None,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    planned_n_per_group: Optional[int] = None,
) -> str:
    """Compute SHA-256 hex digest of a canonical JSON (sorted keys) of the experiment plan."""
    # Hash all locked plan parameters so any post-hoc change to thresholds or rules alters the digest.
    payload = {
        "alpha": float(alpha) if alpha is not None else None,
        "baseline_rate": float(baseline_rate) if baseline_rate is not None else None,
        "hypothesis": hypothesis_text.strip(),
        "mde": float(mde) if mde is not None else None,
        "name": name.strip(),
        "planned_n_per_group": int(planned_n_per_group) if planned_n_per_group is not None else None,
        "power": float(power) if power is not None else None,
        "success_rule": (success_rule or "").strip(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_experiment_hash(experiment: Dict[str, Any]) -> bool:
    """Verify a stored experiment record against its SHA-256 hypothesis_hash."""
    expected = hash_hypothesis(
        hypothesis_text=str(experiment.get("hypothesis", "")),
        name=str(experiment.get("name", "")),
        success_rule=str(experiment.get("success_rule") or ""),
        baseline_rate=experiment.get("baseline_rate"),
        mde=experiment.get("mde"),
        alpha=experiment.get("alpha"),
        power=experiment.get("power"),
        planned_n_per_group=experiment.get("planned_n_per_group"),
    )
    stored = str(experiment.get("hypothesis_hash", ""))
    return hmac.compare_digest(expected, stored)


def create_experiment(
    conn: sqlite3.Connection,
    name: str,
    hypothesis: str,
    success_rule: str,
    baseline_rate: float,
    mde: float,
    alpha: float,
    power: float,
    planned_n_per_group: int,
    locked_at: Optional[str] = None,
) -> int:
    """Persist a locked experiment and return its experiment ID.

    If an experiment with the same name already exists, returns its ID.

    Args:
        conn: SQLite connection.
        name: Unique name of the experiment.
        hypothesis: Text description of the hypothesis.
        success_rule: Plain text decision / success criterion.
        baseline_rate: Expected baseline conversion rate (e.g. 0.10).
        mde: Minimum detectable effect / smallest improvement to detect (e.g. 0.01).
        alpha: Type I error rate (e.g. 0.05).
        power: Statistical power (e.g. 0.80).
        planned_n_per_group: Required sample size per variant.
        locked_at: ISO timestamp of lock; defaults to current UTC time.

    Returns:
        Integer primary key ID of the experiment.
    """
    if locked_at is None:
        locked_at = datetime.now(timezone.utc).isoformat()

    h_hash = hash_hypothesis(
        hypothesis_text=hypothesis,
        name=name,
        success_rule=success_rule,
        baseline_rate=baseline_rate,
        mde=mde,
        alpha=alpha,
        power=power,
        planned_n_per_group=planned_n_per_group,
    )
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM experiments WHERE name = ?", (name.strip(),))
    existing = cursor.fetchone()
    if existing is not None:
        return int(existing[0])

    cursor.execute(
        """
        INSERT INTO experiments (
            name, hypothesis, hypothesis_hash, success_rule,
            baseline_rate, mde, alpha, power, planned_n_per_group, locked_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name.strip(),
            hypothesis.strip(),
            h_hash,
            success_rule.strip(),
            float(baseline_rate),
            float(mde),
            float(alpha),
            float(power),
            int(planned_n_per_group),
            locked_at,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def get_experiment(conn: sqlite3.Connection, experiment_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve an experiment record by ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
    row = cursor.fetchone()
    if row is None:
        return None
    return dict(row)


def get_experiment_by_name(conn: sqlite3.Connection, name: str) -> Optional[Dict[str, Any]]:
    """Retrieve an experiment record by name."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments WHERE name = ?", (name.strip(),))
    row = cursor.fetchone()
    if row is None:
        return None
    return dict(row)


def list_experiments(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """List all experiments ordered by newest first."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments ORDER BY id DESC")
    return [dict(r) for r in cursor.fetchall()]


def log_event(
    conn: sqlite3.Connection,
    user_id: int,
    experiment_id: int,
    event_name: str,
    created_at: Optional[str] = None,
) -> None:
    """Log an interaction or conversion event."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO events (user_id, experiment_id, event_name, created_at) VALUES (?, ?, ?, ?)",
        (user_id, experiment_id, event_name, created_at),
    )
    conn.commit()


def log_events_batch(
    conn: sqlite3.Connection,
    events: List[Tuple[int, int, str, str]],
) -> None:
    """Batch insert multiple events in a single transaction.

    Args:
        conn: SQLite connection.
        events: List of tuples (user_id, experiment_id, event_name, created_at).
    """
    if not events:
        return
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO events (user_id, experiment_id, event_name, created_at) VALUES (?, ?, ?, ?)",
        events,
    )
    conn.commit()


def store_assignments_batch(
    conn: sqlite3.Connection,
    assignments: List[Tuple[int, int, str, str]],
) -> None:
    """Batch insert multiple assignments in a single transaction.

    Args:
        conn: SQLite connection.
        assignments: List of tuples (user_id, experiment_id, variant, assigned_at).
    """
    if not assignments:
        return
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT OR IGNORE INTO assignments (user_id, experiment_id, variant, assigned_at)
        VALUES (?, ?, ?, ?)
        """,
        assignments,
    )
    conn.commit()


def clear_experiment_data(conn: sqlite3.Connection, experiment_id: int) -> None:
    """Clear all assignments and events for an experiment."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM events WHERE experiment_id = ?", (experiment_id,))
    cursor.execute("DELETE FROM assignments WHERE experiment_id = ?", (experiment_id,))
    conn.commit()


def get_counts(conn: sqlite3.Connection, experiment_id: int) -> Dict[str, Any]:
    """Calculate visitor counts and conversions for variants A and B using SQL.

    Returns:
        Dict with keys:
            x_a: Conversions in group A.
            n_a: Total unique visitors assigned to group A.
            x_b: Conversions in group B.
            n_b: Total unique visitors assigned to group B.
            conv_a: Conversion rate in group A (x_a / n_a).
            conv_b: Conversion rate in group B (x_b / n_b).
            total_users: Total assigned users (n_a + n_b).
            total_conversions: Total conversions (x_a + x_b).
    """
    cursor = conn.cursor()
    query = """
    SELECT
        a.variant,
        COUNT(DISTINCT a.user_id) AS total_users,
        COUNT(DISTINCT CASE WHEN e.event_name = 'order_complete' THEN e.user_id END) AS conversions
    FROM assignments a
    LEFT JOIN events e
        ON a.user_id = e.user_id AND a.experiment_id = e.experiment_id
    WHERE a.experiment_id = ?
    GROUP BY a.variant
    """
    cursor.execute(query, (experiment_id,))
    rows = cursor.fetchall()

    counts = {
        "A": {"n": 0, "x": 0},
        "B": {"n": 0, "x": 0},
    }

    for row in rows:
        var = str(row["variant"])
        if var in counts:
            counts[var]["n"] = int(row["total_users"])
            counts[var]["x"] = int(row["conversions"])

    n_a = counts["A"]["n"]
    x_a = counts["A"]["x"]
    n_b = counts["B"]["n"]
    x_b = counts["B"]["x"]

    conv_a = (x_a / n_a) if n_a > 0 else 0.0
    conv_b = (x_b / n_b) if n_b > 0 else 0.0

    return {
        "x_a": x_a,
        "n_a": n_a,
        "x_b": x_b,
        "n_b": n_b,
        "conv_a": float(conv_a),
        "conv_b": float(conv_b),
        "total_users": n_a + n_b,
        "total_conversions": x_a + x_b,
    }
