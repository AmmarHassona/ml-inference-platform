import pytest
import numpy as np
import app.services.embedding_drift as embedding_drift
from app.config import EMBEDDING_REFERENCE_SIZE, EMBEDDING_MIN_SAMPLES
from app.metrics import EMBEDDING_DRIFT_SCORE

@pytest.fixture(autouse=True)
def reset_drift_state():
    embedding_drift._reference_locked = False
    embedding_drift._reference_embeddings.clear()
    embedding_drift._embedding_window.clear()

def test_reference_locks_after_enough_embeddings():
    for _ in range(EMBEDDING_REFERENCE_SIZE):
        embedding_drift.record_embedding(np.ones(384).tolist())

    assert embedding_drift._reference_locked == True

def test_drift_score_low_on_identical_embeddings():
    emb = np.ones(384).tolist()
    
    for _ in range(EMBEDDING_REFERENCE_SIZE):
        embedding_drift.record_embedding(emb)
    
    for _ in range(EMBEDDING_MIN_SAMPLES):
        embedding_drift.record_embedding(emb)
    
    embedding_drift.compute_embedding_drift()
    
    score = EMBEDDING_DRIFT_SCORE._value.get()
    assert score < 0.1

def test_drift_score_high_on_opposite_embeddings():
    ref_emb = np.ones(384).tolist()
    window_emb = np.full(384, -1.0).tolist()
    
    for _ in range(EMBEDDING_REFERENCE_SIZE):
        embedding_drift.record_embedding(ref_emb)
    
    for _ in range(EMBEDDING_MIN_SAMPLES):
        embedding_drift.record_embedding(window_emb)
    
    embedding_drift.compute_embedding_drift()
    
    score = EMBEDDING_DRIFT_SCORE._value.get()
    assert score > 0.9