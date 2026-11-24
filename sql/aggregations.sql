-- =============================================================================
-- A/B Testing Aggregation Queries
-- Computes unique visitor counts (n_a, n_b) and conversions (x_a, x_b)
-- =============================================================================

-- Variant-level aggregation for an experiment (:experiment_id)
SELECT
    a.variant,
    COUNT(DISTINCT a.user_id) AS total_users,
    COUNT(DISTINCT CASE WHEN e.event_name = 'order_complete' THEN e.user_id END) AS conversions,
    ROUND(
        CAST(COUNT(DISTINCT CASE WHEN e.event_name = 'order_complete' THEN e.user_id END) AS REAL) /
        NULLIF(COUNT(DISTINCT a.user_id), 0),
        4
    ) AS conversion_rate
FROM assignments a
LEFT JOIN events e
    ON a.user_id = e.user_id AND a.experiment_id = e.experiment_id
WHERE a.experiment_id = :experiment_id
GROUP BY a.variant;

-- Overall experiment funnel summary
SELECT
    e.name AS experiment_name,
    COUNT(DISTINCT a.user_id) AS total_assigned_users,
    COUNT(DISTINCT e.id) AS total_events_logged,
    COUNT(DISTINCT CASE WHEN e.event_name = 'view_checkout' THEN e.user_id END) AS checkout_views,
    COUNT(DISTINCT CASE WHEN e.event_name = 'order_complete' THEN e.user_id END) AS total_conversions
FROM experiments exp
JOIN assignments a ON exp.id = a.experiment_id
LEFT JOIN events e ON a.user_id = e.user_id AND a.experiment_id = e.experiment_id
WHERE exp.id = :experiment_id
GROUP BY exp.id;
