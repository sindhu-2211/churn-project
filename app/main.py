"""Step 8: Churn scoring API.

Run locally from the project root:  uvicorn app.main:app --reload
Then open http://localhost:8000/docs to try it in the browser.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app import db
from app.schemas import Customer, Prediction
from src.explain import ChurnExplainer
from src.features import add_features

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "models" / "serving_config.json").read_text())
PIPE = joblib.load(ROOT / "models" / f"{CONFIG['model']}.joblib")
EXPLAINER = ChurnExplainer(PIPE, pd.read_csv(ROOT / "models" / "shap_background.csv"))
CONN = db.connect()

app = FastAPI(title="Customer Churn API",
              description="Scores a telecom customer's churn risk and explains why.")


def risk_tier(p: float) -> str:
    if p >= CONFIG["high_risk_min"]:
        return "high"
    if p >= CONFIG["medium_risk_min"]:
        return "medium"
    return "low"


@app.get("/health")
def health():
    return {"status": "ok", "model": CONFIG["model"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    raw = customer.model_dump()
    df = pd.DataFrame([raw])
    proba = float(PIPE.predict_proba(add_features(df))[0, 1])
    tier = risk_tier(proba)
    # Reasons are only meaningful when there is real risk to explain
    reasons = EXPLAINER.top_reasons(df) if tier != "low" else []
    db.log_prediction(CONN, raw, proba, tier, CONFIG["model"])
    return Prediction(customerID=customer.customerID, churn_probability=round(proba, 4),
                      risk_tier=tier, top_reasons=reasons, model=CONFIG["model"])


@app.get("/monitoring/summary")
def monitoring_summary():
    """Quick drift check: compare these numbers week to week."""
    return db.summary(CONN)
