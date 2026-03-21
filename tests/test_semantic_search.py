import pytest
import numpy as np
import app.services.semantic_search as semantic_search

@pytest.fixture(autouse=True)
def reset_corpus():
    semantic_search._corpus_embeddings = []
    semantic_search._corpus_labels = []
    semantic_search._corpus_texts = []

def test_ml_query_returns_ml_label():
    semantic_search._corpus_embeddings = [np.ones(384), np.full(384, -1.0)]
    semantic_search._corpus_labels = ["machine_learning", "finance"]
    semantic_search._corpus_texts = ["ML sentence", "Finance sentence"]

    query = np.ones(384)
    label, _, _ = semantic_search.find_nearest(query)
    
    assert label == "machine_learning"

def test_finance_query_returns_finance_label():
    semantic_search._corpus_embeddings = [np.ones(384), np.full(384, -1.0)]
    semantic_search._corpus_labels = ["machine_learning", "finance"]
    semantic_search._corpus_texts = ["ML sentence", "Finance sentence"]

    query = np.full(384, -1.0)
    label, _, _ = semantic_search.find_nearest(query)
    
    assert label == "finance"