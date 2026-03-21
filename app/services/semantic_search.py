import numpy as np
from app.config import CORPUS
from app.services.embedding_drift import cosine_similarity

_corpus = CORPUS

# populated at startup by load_corpus()
_corpus_embeddings: list[np.ndarray] = []
_corpus_labels: list[str] = []
_corpus_texts: list[str] = []

def embed_text(text: str, session, tokenizer) -> np.ndarray:    
    inputs = tokenizer(
        text,
        padding = True,
        truncation = True,
        return_tensors = "np",
    )

    outputs = session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))
    })

    # mean pooling with numpy
    token_embeddings = outputs[0]  # shape (1, seq_len, 384)
    attention_mask = inputs["attention_mask"]  # shape (1, seq_len)
    mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
    embedding = (token_embeddings * mask_expanded).sum(axis = 1) / mask_expanded.sum(axis = 1).clip(min = 1e-9)

    # normalize
    embedding = embedding / np.linalg.norm(embedding, axis = 1, keepdims = True)
    embedding = embedding[0]

    return embedding

def load_corpus(session, tokenizer):
    for text, label in _corpus:
        embedding = embed_text(text, session, tokenizer)
        _corpus_labels.append(label)
        _corpus_texts.append(text)
        _corpus_embeddings.append(embedding)

def find_nearest(query_embedding: np.ndarray) -> tuple[str, float, str]:
    scores = [cosine_similarity(query_embedding, emb) for emb in _corpus_embeddings]
    best_idx = int(np.argmax(scores))
    return _corpus_labels[best_idx], round(scores[best_idx], 4), _corpus_texts[best_idx]