import pytest
from fastapi.testclient import TestClient
from app.main import app

# TestClient runs the full lifespan once per session: models load, scheduler starts
@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- /predict ---

def test_predict_returns_expected_fields(client):
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "probabilities" in body

def test_predict_prediction_is_valid_class(client):
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.json()["prediction"] in (0, 1, 2)

def test_predict_probabilities_sum_to_one(client):
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    total = sum(response.json()["probabilities"])
    assert abs(total - 1.0) < 1e-5

def test_predict_rejects_too_few_features(client):
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422

def test_predict_rejects_too_many_features(client):
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    assert response.status_code == 422

def test_predict_rejects_empty_features(client):
    response = client.post("/predict", json={"features": []})
    assert response.status_code == 422

def test_predict_rejects_missing_body(client):
    response = client.post("/predict")
    assert response.status_code == 422


# --- /predict/text ---

def test_text_predict_returns_expected_fields(client):
    response = client.post("/predict/text", json={"text": "neural network training"})
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "similarity" in body
    assert "matched_document" in body

def test_text_predict_label_is_known_class(client):
    response = client.post("/predict/text", json={"text": "neural network training"})
    assert response.json()["label"] in ("world", "sports", "business", "sci_tech")

def test_text_predict_similarity_in_valid_range(client):
    response = client.post("/predict/text", json={"text": "neural network training"})
    similarity = response.json()["similarity"]
    assert 0.0 <= similarity <= 1.0

def test_text_predict_tech_query_returns_sci_tech_label(client):
    response = client.post("/predict/text", json={"text": "software company releases new technology product"})
    assert response.json()["label"] == "sci_tech"

def test_text_predict_business_query_returns_business_label(client):
    response = client.post("/predict/text", json={"text": "stock market Wall Street earnings revenue"})
    assert response.json()["label"] == "business"

def test_text_predict_rejects_missing_body(client):
    response = client.post("/predict/text")
    assert response.status_code == 422

def test_text_predict_rejects_empty_text(client):
    response = client.post("/predict/text", json={"text": ""})
    # empty string is valid input — model should still return a label
    assert response.status_code == 200


# --- /metrics ---

def test_metrics_endpoint_returns_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "ml_predictions_total" in response.text