"""Step 4: Feature engineering and encoding.

`add_features` is a pure function of a single customer's row, so the exact same
code runs at training time and inside the API later (no train/serve skew).
`build_preprocessor` returns the encoder; it is fit on the training set only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Tenure buckets: churn risk drops sharply after the first year, not linearly
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=TENURE_BINS,
                                 labels=TENURE_LABELS).astype(str)
    # How many add-on services the customer has: more services = more "locked in"
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1) + (df["InternetService"] != "No")
    # Interaction: a high bill hurts most when the customer can leave any month
    is_m2m = (df["Contract"] == "Month-to-month").astype(int)
    df["m2m_x_monthly_charges"] = is_m2m * df["MonthlyCharges"]
    # Price-change proxy: current bill vs their lifetime average bill.
    # Positive = they now pay more than they used to (the closest thing to a
    # usage trend this dataset allows, since it has no monthly history)
    avg_monthly = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["charge_vs_avg"] = np.where(df["tenure"] > 0, df["MonthlyCharges"] - avg_monthly, 0.0)
    df["auto_pay"] = df["PaymentMethod"].str.contains("automatic").map({True: "Yes", False: "No"})
    df["fiber_no_support"] = ((df["InternetService"] == "Fiber optic")
                              & (df["TechSupport"] == "No")).map({True: "Yes", False: "No"})
    return df


NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "num_services",
           "m2m_x_monthly_charges", "charge_vs_avg"]
CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
               "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
               "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
               "Contract", "PaperlessBilling", "PaymentMethod",
               "tenure_bucket", "auto_pay", "fiber_no_support"]


def build_preprocessor() -> ColumnTransformer:
    # One-hot (not label) encoding: categories like PaymentMethod have no order.
    # handle_unknown="ignore" so an unseen category in production doesn't crash the API.
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
    ])
