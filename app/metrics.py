from prometheus_client import Counter, Histogram, Gauge

PREDICTION_COUNTER = Counter(
    "ml_predictions_total", 
    "Total number of predictions made by model",
    ["model_version"]
)

PREDICTION_LATENCY = Histogram(
    "ml_predictions_latency_seconds", 
    "Model inference latency in seconds",
    ["model_version"],
    buckets = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

PREDICTION_PROBABILITY = Histogram(
    "ml_prediction_probability",
    "Distribution of max prediction probability",
    ["model_version"],
    buckets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

PSI_SCORE = Gauge(
    "ml_psi_score",
    "PSI drift score per feature",
    ["feature_name"]
)

SHADOW_DIVERGENCE = Gauge(
    "ml_shadow_divergence",
    "Divergence rate between v1 and v2 predictions"
)

ROLLBACK_COUNTER = Counter(
    "ml_rollback_total",
    "Number of automatic rollbacks triggered"
)