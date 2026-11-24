-- =============================================================================
-- A/B Testing Platform Database Schema (SQLite)
-- Tables: experiments, assignments, events
-- =============================================================================

-- PRAGMA Configurations
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- 1. Experiments Registry & Pre-Registration Metadata
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

-- 2. Deterministic Variant Assignments
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

-- 3. Behavioral and Conversion Event Stream
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
