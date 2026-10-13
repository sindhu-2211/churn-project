"""Log every prediction to SQLite so we can monitor drift later
(e.g. is the share of high-risk customers or the average bill changing?)."""
import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.environ.get("PREDICTIONS_DB", "predictions.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    customer_id TEXT,
    model TEXT NOT NULL,
    churn_probability REAL NOT NULL,
    risk_tier TEXT NOT NULL,
    features_json TEXT NOT NULL
)"""


def connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(SCHEMA)
    return conn


def log_prediction(conn, customer: dict, proba: float, tier: str, model: str):
    conn.execute(
        "INSERT INTO predictions (created_at, customer_id, model, churn_probability,"
        " risk_tier, features_json) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), customer.get("customerID"), model,
         proba, tier, json.dumps(customer)))
    conn.commit()


def summary(conn) -> dict:
    n, avg_p, avg_bill = conn.execute(
        "SELECT COUNT(*), AVG(churn_probability),"
        " AVG(json_extract(features_json, '$.MonthlyCharges')) FROM predictions").fetchone()
    tiers = dict(conn.execute(
        "SELECT risk_tier, COUNT(*) FROM predictions GROUP BY risk_tier").fetchall())
    return {"n_predictions": n, "avg_churn_probability": avg_p,
            "avg_monthly_charges": avg_bill, "tier_counts": tiers}
