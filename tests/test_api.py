from fastapi.testclient import TestClient
from app.main import app

EXAMPLE = app.openapi()["components"]["schemas"]["Customer"]["example"]


def test_health_and_predict():
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/predict", json=EXAMPLE)
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["churn_probability"] <= 1
        assert body["risk_level"] in {"low", "medium", "high"}


def test_risk_ordering():
    safe = {**EXAMPLE, "contract_type": "Two year", "tenure_months": 60, "total_charges": 3000,
            "support_tickets_90d": 0, "late_payments_12m": 0, "days_since_last_login": 2,
            "satisfaction_score": 9, "payment_method": "Credit card", "monthly_charges": 50}
    with TestClient(app) as c:
        hi = c.post("/predict", json=EXAMPLE).json()["churn_probability"]
        lo = c.post("/predict", json=safe).json()["churn_probability"]
        assert hi > lo


def test_validation_and_batch_and_missing_values():
    with TestClient(app) as c:
        assert c.post("/predict", json={**EXAMPLE, "contract_type": "Weekly"}).status_code == 422
        partial = {**EXAMPLE, "age": None, "satisfaction_score": None, "avg_usage_gb": None}
        r = c.post("/predict-batch", json=[EXAMPLE, partial])
        assert r.status_code == 200 and len(r.json()) == 2
