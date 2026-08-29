import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "expected_features" in data

def test_manual_prediction():
    payload = {
        "male": 1,
        "age": 55.0,
        "education": 2.0,
        "currentSmoker": 1,
        "BPMeds": 0,
        "prevalentStroke": 0,
        "prevalentHyp": 1,
        "diabetes": 0,
        "BMI": 28.5
    }
    response = client.post("/api/predict/manual", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_probability" in data
    assert "risk_category" in data
    assert "disclaimer" in data
    assert 0.0 <= data["predicted_probability"] <= 1.0
