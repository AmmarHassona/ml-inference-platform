import numpy as np
import app.services.embedding_drift as embedding_drift
from app.config import EMBEDDING_REFERENCE_SIZE, EMBEDDING_MIN_SAMPLES
from app.metrics import EMBEDDING_DRIFT_SCORE
import pytest


@pytest.fixture(autouse=True)
def reset_embedding_state():
    embedding_drift._reference_locked = False
    embedding_drift._reference_embeddings.clear()
    embedding_drift._embedding_window.clear()


def _normalized(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def test_drift_score_rises_after_topic_shift():
    rng = np.random.default_rng(42)

    # fill reference with in-distribution embeddings 
    for _ in range(EMBEDDING_REFERENCE_SIZE):
        # Cluster around [1, 0, 0, ...] with small noise
        vec = _normalized(np.ones(384) + rng.normal(0, 0.05, 384))
        embedding_drift.record_embedding(vec.tolist())

    assert embedding_drift._reference_locked is True

    # baseline check — window is empty, no drift yet
    embedding_drift.compute_embedding_drift()
    baseline_score = EMBEDDING_DRIFT_SCORE._value.get()
    assert baseline_score == 0.0  # gauge reads 0.0 before first successful drift check (empty window)

    # inject OOD embeddings (topic shift — opposite cluster)
    for _ in range(EMBEDDING_MIN_SAMPLES * 3):
        vec = _normalized(np.full(384, -1.0) + rng.normal(0, 0.05, 384))
        embedding_drift.record_embedding(vec.tolist())

    # drift check — score should cross the 0.3 warning threshold
    embedding_drift.compute_embedding_drift()
    drift_score = EMBEDDING_DRIFT_SCORE._value.get()

    assert drift_score > 0.3, f"Expected drift > 0.3 after topic shift, got {drift_score:.4f}"
