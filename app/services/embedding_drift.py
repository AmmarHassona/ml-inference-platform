import numpy as np
from collections import deque
from app.metrics import EMBEDDING_DRIFT_SCORE

REFERENCE_SIZE = 50
_reference_embeddings = []
_reference_locked = False
_embedding_window = deque(maxlen=200)

def record_embedding(embedding: list[float]):
    global _reference_locked
    emb = np.array(embedding)
    if not _reference_locked:
        _reference_embeddings.append(emb)
        if len(_reference_embeddings) >= REFERENCE_SIZE:
            _reference_locked = True
        return
    _embedding_window.append(emb)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def compute_embedding_drift():
    if not _reference_locked or len(_embedding_window) < 10:
        return
    similarities = []
    for emb in _embedding_window:
        sims = [cosine_similarity(emb, ref) for ref in _reference_embeddings]
        similarities.append(np.mean(sims))
    drift_score = 1.0 - float(np.mean(similarities))
    EMBEDDING_DRIFT_SCORE.set(drift_score)
    print(f"Embedding drift score: {drift_score:.4f}", flush=True)