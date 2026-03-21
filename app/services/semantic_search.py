import numpy as np
from app.services.embedding_drift import cosine_similarity

_corpus = [
    # machine_learning
    ("Machine learning is training models to learn patterns from data", "machine_learning"),
    ("Neural networks are inspired by the structure of the human brain", "machine_learning"),
    ("Gradient descent is an optimization algorithm used to train models", "machine_learning"),
    ("Overfitting occurs when a model learns noise instead of signal", "machine_learning"),
    ("Random forests combine many decision trees to improve prediction accuracy", "machine_learning"),
    ("Transformers use self-attention to process sequences in parallel", "machine_learning"),
    # sports
    ("The team scored in the final minutes to win the championship", "sports"),
    ("Athletes train for years to compete at the Olympic Games", "sports"),
    ("The referee awarded a penalty kick after the foul in the box", "sports"),
    ("Basketball players must master dribbling, passing, and shooting", "sports"),
    ("The marathon runner collapsed at the finish line after 26 miles", "sports"),
    ("Tennis requires both physical endurance and precise technique", "sports"),
    # finance
    ("Interest rates set by central banks influence borrowing costs", "finance"),
    ("Diversifying a portfolio reduces exposure to individual asset risk", "finance"),
    ("Inflation erodes the purchasing power of money over time", "finance"),
    ("Stock prices reflect investor expectations about future earnings", "finance"),
    ("A bond is a fixed income instrument representing a loan to a borrower", "finance"),
    ("Hedge funds use leverage and derivatives to amplify returns", "finance"),
    # health
    ("Regular exercise reduces the risk of cardiovascular disease", "health"),
    ("A balanced diet rich in vegetables supports immune function", "health"),
    ("Sleep deprivation impairs cognitive performance and memory consolidation", "health"),
    ("Vaccination programs have eliminated several previously deadly diseases", "health"),
    ("Chronic stress elevates cortisol levels and damages long-term health", "health"),
    ("Early cancer detection significantly improves survival outcomes", "health"),
]

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