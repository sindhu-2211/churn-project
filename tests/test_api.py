import os
import tempfile

os.environ["PREDICTIONS_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.schemas import Customer  # noqa: E402

client = TestClient(app)
HIGH_RISK = Customer.model_config["json_schema_extra"]["example"]
LOW_RISK = {**HIGH_RISK, "customerID": "low-1", "tenure": 70, "Contract": "Two year",
            "InternetService": "DSL", "TechSupport": "Yes", "OnlineSecurity": "Yes",
            "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 60.0,
            "TotalCharges": 4200.0}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_high_risk_customer():
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert body["risk_tier"] == "high"
    assert body["churn_probability"] > 0.8
    assert body["top_reasons"][0]["feature"] == "Contract"


def test_low_risk_customer():
    body = client.post("/predict", json=LOW_RISK).json()
    assert body["risk_tier"] == "low"
    assert body["churn_probability"] < 0.1
    assert body["top_reasons"] == []


def test_invalid_input_rejected():
    r = client.post("/predict", json={**HIGH_RISK, "Contract": "month to month"})
    assert r.status_code == 422


def test_predictions_are_logged():
    s = client.get("/monitoring/summary").json()
    assert s["n_predictions"] >= 2
    assert set(s["tier_counts"]) <= {"high", "medium", "low"}
