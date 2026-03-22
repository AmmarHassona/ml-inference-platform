import pytest
import numpy as np
import app.services.topic_classification as topic_classification

@pytest.fixture(autouse=True)
def reset_corpus():
    topic_classification._corpus_embeddings = []
    topic_classification._corpus_labels = []
    topic_classification._corpus_texts = []

def test_ml_query_returns_ml_label():
    topic_classification._corpus_embeddings = [np.ones(384), np.full(384, -1.0)]
    topic_classification._corpus_labels = ["machine_learning", "finance"]
    topic_classification._corpus_texts = ["ML sentence", "Finance sentence"]

    query = np.ones(384)
    label, _, _ = topic_classification.find_nearest(query)
    
    assert label == "machine_learning"

def test_finance_query_returns_finance_label():
    topic_classification._corpus_embeddings = [np.ones(384), np.full(384, -1.0)]
    topic_classification._corpus_labels = ["machine_learning", "finance"]
    topic_classification._corpus_texts = ["ML sentence", "Finance sentence"]

    query = np.full(384, -1.0)
    label, _, _ = topic_classification.find_nearest(query)
    
    assert label == "finance"