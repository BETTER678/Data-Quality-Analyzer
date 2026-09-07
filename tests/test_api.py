import pytest
from fastapi.testclient import TestClient
from api.main import app
import io

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_checks_endpoint():
    response = client.get("/checks")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) >= 5

def test_analyze_endpoint(clean_df):
    csv_content = clean_df.to_csv(index=False)
    files = {"file": ("data.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    
    response = client.post("/analyze", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "readiness_score" in data
    assert "score" in data["readiness_score"]
    assert "quality_gate" in data
    assert "profiling" in data
