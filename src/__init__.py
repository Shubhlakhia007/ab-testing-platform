"""Source package for A/B testing platform."""

from src.assign import assign, assign_and_store, balance_report
from src.db import (
    clear_experiment_data,
    create_experiment,
    get_connection,
    get_counts,
    get_experiment,
    get_experiment_by_name,
    hash_hypothesis,
    init_db,
    list_experiments,
    log_event,
    log_events_batch,
    store_assignments_batch,
    verify_experiment_hash,
)
from src.simulate import (
    aa_test,
    peeking,
    power_check,
    power_curve,
    simulate_experiment,
)
from src.stats_engine import (
    decide,
    power_for,
    sample_ratio_check,
    sample_size,
    two_prop_test,
)

__all__ = [
    "assign",
    "assign_and_store",
    "balance_report",
    "sample_size",
    "two_prop_test",
    "decide",
    "sample_ratio_check",
    "power_for",
    "simulate_experiment",
    "aa_test",
    "peeking",
    "power_check",
    "power_curve",
    "init_db",
    "get_connection",
    "create_experiment",
    "get_experiment",
    "get_experiment_by_name",
    "list_experiments",
    "log_event",
    "log_events_batch",
    "store_assignments_batch",
    "clear_experiment_data",
    "get_counts",
    "hash_hypothesis",
    "verify_experiment_hash",
]
