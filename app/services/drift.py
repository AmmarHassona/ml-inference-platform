import threading
import numpy as np
import pandas as pd
from collections import deque
from app.config import REFERENCE_FEATURES_PATH, PSI_WINDOW_SIZE, PSI_MIN_SAMPLES, PSI_N_BINS
from app.metrics import PSI_SCORE
from app.logger import get_logger

logger = get_logger("drift")

reference_features = None
bin_edges_per_feature = []
ref_pcts = []

feature_window = deque(maxlen=PSI_WINDOW_SIZE)
_lock = threading.Lock()


def initialize_drift():
    global reference_features, bin_edges_per_feature, ref_pcts
    reference_features = np.load(str(REFERENCE_FEATURES_PATH))
    n_bins = PSI_N_BINS
    for i in range(reference_features.shape[1]):
        _, edges = pd.qcut(reference_features[:, i], n_bins, retbins=True, duplicates="drop")
        edges[0] = -np.inf
        edges[-1] = np.inf
        bin_edges_per_feature.append(edges)
        ref_counts = np.histogram(reference_features[:, i], bins=edges)[0]
        ref_pct = ref_counts / len(reference_features)
        ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
        ref_pcts.append(ref_pct)


def record_features(features: list[float]):
    with _lock:
        feature_window.append(features)


def calculate_psi(current_features: np.ndarray) -> dict:
    psi_scores = {}
    for i, edges in enumerate(bin_edges_per_feature):
        cur_counts = np.histogram(current_features[:, i], bins=edges)[0]
        cur_pct = cur_counts / len(current_features)
        cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
        psi_scores[f"feature_{i}"] = float(np.sum((cur_pct - ref_pcts[i]) * np.log(cur_pct / ref_pcts[i])))
    return psi_scores


def run_drift_check():
    if len(feature_window) < PSI_MIN_SAMPLES:
        return
    with _lock:
        current = np.array(list(feature_window))
    scores = calculate_psi(current)
    for feature_name, score in scores.items():
        PSI_SCORE.labels(feature_name=feature_name).set(score)
        if score > 0.2:
            logger.warning("psi_drift_detected", feature=feature_name, psi_score=round(score, 4))
        else:
            logger.info("psi_check", feature=feature_name, psi_score=round(score, 4))
    return scores