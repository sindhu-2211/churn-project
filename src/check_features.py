"""Sanity-check the engineered features on the training set."""
from pathlib import Path
import pandas as pd
from src.features import add_features, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
train = add_features(pd.read_csv(ROOT / "data/processed/train.csv"))

for col in ["tenure_bucket", "num_services", "auto_pay", "fiber_no_support"]:
    print(f"\nChurn rate (%) by {col}:")
    print((train.groupby(col)["churn"].agg(["size", "mean"]).assign(mean=lambda t: (t["mean"]*100).round(1))))

train["charge_trend"] = pd.cut(train["charge_vs_avg"], [-1e9, -2, 2, 1e9], labels=["paying less", "flat", "paying more"])
print("\nChurn rate (%) by charge_vs_avg band:")
print(train.groupby("charge_trend", observed=True)["churn"].agg(["size", "mean"]).assign(mean=lambda t: (t["mean"]*100).round(1)))

X = build_preprocessor().fit_transform(train)
print(f"\nEncoded feature matrix: {X.shape[0]} rows x {X.shape[1]} columns")
