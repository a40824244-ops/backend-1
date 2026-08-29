import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_endpoint():
    payload = {
        "Gender": "Female",
        "Age": 18,
        "Study_Hours": 6.0,
        "Attendance": 85.0,
        "Assignment_Score": 80.0,
        "Quiz_Score": 75.0,
        "Midterm_Score": 78.0,
        "Internet_Access": "Yes",
        "Parent_Education": "Bachelor",
        "Family_Income": "Medium",
        "Extra_Activities": "Yes",
        "Previous_GPA": 3.4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "pass_probability" in data
    assert "predicted_target" in data
    assert "risk_level" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "feature_importance_ranking" in data
