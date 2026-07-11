from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input():
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_class" in body
    assert "probabilities" in body


def test_predict_invalid_input_length():
    payload = {"features": [1.0, 2.0]}  # wrong number of features
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "metrics" in body
