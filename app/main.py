"""FastAPI service for churn predictions.

Run:  uvicorn app.main:app --reload      (docs at http://localhost:8000/docs)
"""
from contextlib import asynccontextmanager
from typing import List, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_PATH

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Run `python -m src.train` first.")
    state["bundle"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0", lifespan=lifespan)


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    age: float | None = Field(None, ge=18, le=100)
    region: Literal["North", "South", "East", "West", "Central"]
    contract_type: Literal["Month-to-month", "One year", "Two year"]
    payment_method: Literal["Credit card", "Bank transfer", "Electronic check", "Mailed check"]
    signup_channel: Literal["Online", "Retail", "Referral", "Partner"]
    tenure_months: float = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., gt=0, le=1000)
    total_charges: float = Field(..., ge=0)
    support_tickets_90d: int = Field(..., ge=0)
    avg_usage_gb: float | None = Field(None, ge=0)
    late_payments_12m: int = Field(..., ge=0)
    has_streaming: Literal[0, 1]
    has_security: Literal[0, 1]
    num_products: int = Field(..., ge=1, le=10)
    days_since_last_login: float = Field(..., ge=0)
    satisfaction_score: float | None = Field(None, ge=1, le=10)

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "age": 34, "region": "South", "contract_type": "Month-to-month",
        "payment_method": "Electronic check", "signup_channel": "Online", "tenure_months": 4,
        "monthly_charges": 95.5, "total_charges": 380, "support_tickets_90d": 3, "avg_usage_gb": 80,
        "late_payments_12m": 2, "has_streaming": 1, "has_security": 0, "num_products": 4,
        "days_since_last_login": 40, "satisfaction_score": 3.5}}}


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: Literal["low", "medium", "high"]
    threshold: float


def _risk(p: float) -> str:
    return "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"


def _predict(customers: List[Customer]) -> List[Prediction]:
    b = state["bundle"]
    df = pd.DataFrame([c.model_dump() for c in customers])
    proba = b["pipeline"].predict_proba(df)[:, 1]
    return [Prediction(churn_probability=round(float(p), 4), churn_prediction=bool(p >= b["threshold"]),
                       risk_level=_risk(p), threshold=round(b["threshold"], 4)) for p in proba]


@app.get("/health")
def health():
    b = state["bundle"]
    return {"status": "ok", "model": b["model_name"], "trained_at": b["trained_at"]}


@app.get("/model-info")
def model_info():
    b = state["bundle"]
    return {"model": b["model_name"], "threshold": b["threshold"], "test_metrics": b["metrics_test"],
            "features": b["features"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return _predict([customer])[0]


@app.post("/predict-batch", response_model=List[Prediction])
def predict_batch(customers: List[Customer]):
    if not 0 < len(customers) <= 5000:
        raise HTTPException(422, "Send between 1 and 5000 customers per request.")
    return _predict(customers)
