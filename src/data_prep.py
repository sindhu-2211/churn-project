"""Step 3: Clean the raw data and make a train/test split.

Run from the project root:  python -m src.data_prep
Writes data/processed/train.csv and data/processed/test.csv.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "telco_churn.csv"
OUT = ROOT / "data" / "processed"

TARGET = "churn"
RANDOM_STATE = 42


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and obvious data-quality issues. Safe to run on new customers too."""
    df = df.copy()
    # Blank TotalCharges only occur for tenure == 0 (not billed yet), so 0 is the true value
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", "0", regex=True))
    # SeniorCitizen is 0/1 while every other yes/no column is text; make it consistent
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    if "Churn" in df.columns:
        df[TARGET] = (df.pop("Churn") == "Yes").astype(int)
    return df


def main():
    df = clean(pd.read_csv(RAW))
    # stratify keeps the 26.5% churn rate identical in train and test
    train, test = train_test_split(df, test_size=0.2, stratify=df[TARGET],
                                   random_state=RANDOM_STATE)
    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    print(f"train: {len(train)} rows, churn rate {train[TARGET].mean():.1%}")
    print(f"test:  {len(test)} rows, churn rate {test[TARGET].mean():.1%}")
    print(f"missing values after cleaning: {int(df.isna().sum().sum())}")


if __name__ == "__main__":
    main()
