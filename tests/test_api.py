import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "phishing-detector" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "Phishing Detector API is running"


def test_predict_url_endpoint_returns_result():
    response = client.post(
        "/predict",
        json={"text": "https://www.paypal.com/signin"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "input" in payload
    assert "is_phishing" in payload
    assert "confidence" in payload
    assert "reasons" in payload


def test_predict_rejects_too_long_url():
    response = client.post(
        "/predict",
        json={"text": "https://example.com/" + ("a" * 3000)},
    )
    assert response.status_code == 422


def test_email_predict_rejects_too_large_payload():
    long_text = "a" * 60000
    response = client.post(
        "/predict/email",
        json={"raw_email": long_text},
    )
    assert response.status_code == 422
