import numpy as np
from collections import deque
from app.metrics import EMBEDDING_DRIFT_SCORE
from app.config import EMBEDDING_REFERENCE_SIZE, EMBEDDING_BUFFER_SIZE, EMBEDDING_MIN_SAMPLES
from app.logger import get_logger
import threading

logger = get_logger("embedding_drift")

_reference_embeddings = []
_reference_locked = False
_embedding_window = deque(maxlen=EMBEDDING_BUFFER_SIZE)

_lock = threading.Lock()

def record_embedding(embedding: list[float]):
    global _reference_locked
    emb = np.array(embedding)
    with _lock:
        if not _reference_locked:
            _reference_embeddings.append(emb)
            if len(_reference_embeddings) >= EMBEDDING_REFERENCE_SIZE:
                _reference_locked = True
                logger.info("embedding_reference_locked", reference_size=len(_reference_embeddings))
            return
        _embedding_window.append(emb)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def compute_embedding_drift():
    with _lock:
        if not _reference_locked or len(_embedding_window) < EMBEDDING_MIN_SAMPLES:
            return
        window_snapshot = list(_embedding_window)
        reference_snapshot = list(_reference_embeddings)
    
    similarities = []
    for emb in window_snapshot:
        sims = [cosine_similarity(emb, ref) for ref in reference_snapshot]
        similarities.append(np.mean(sims))
    drift_score = 1.0 - float(np.mean(similarities))
    EMBEDDING_DRIFT_SCORE.set(drift_score)
    if drift_score > 0.3:
        logger.warning("embedding_drift_detected", drift_score=round(drift_score, 4), window_size=len(window_snapshot))
    else:
        logger.info("embedding_drift_check", drift_score=round(drift_score, 4), window_size=len(window_snapshot))