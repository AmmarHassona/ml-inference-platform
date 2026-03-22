import pytest
import numpy as np
import app.services.topic_classification as topic_classification

@pytest.fixture(autouse=True)
def reset_corpus():
    topic_classification._corpus_embeddings = []
    topic_classification._corpus_labels = []
    topic_classification._corpus_texts = []

def test_sci_tech_query_returns_sci_tech_label():
    topic_classification._corpus_embeddings = [np.ones(384), np.full(384, -1.0)]
    topic_classification._corpus_labels = ["sci_tech", "business"]
    topic_classification._corpus_texts = ["Tech sentence", "Business sentence"]

    query = np.ones(384)
    label, _, _ = topic_classification.find_nearest(query)

    assert label == "sci_tech"

def test_business_query_returns_business_label():
    topic_classification._corpus_embeddings = [np.ones(384), np.full(384, -1.0)]
    topic_classification._corpus_labels = ["sci_tech", "business"]
    topic_classification._corpus_texts = ["Tech sentence", "Business sentence"]

    query = np.full(384, -1.0)
    label, _, _ = topic_classification.find_nearest(query)

    assert label == "business"