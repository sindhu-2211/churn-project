"""Step 8 prep: save what the API needs besides the model itself.

Run from the project root:  python -m src.export_serving
- models/shap_background.csv: 200 training customers, the SHAP reference set
- models/serving_config.json: risk-tier cutoffs

Class weights make the raw probabilities run high (the model was told churners
matter ~2.8x more), so tiers are set by rank on the training set instead of
fixed 0.5/0.7 cut-offs: top 10% = high, next 20% = medium, rest = low.
That matches a retention team that can call ~10% of customers.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_prep import TARGET
from src.features import add_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "logistic_regression"


def main():
    pipe = joblib.load(ROOT / "models" / f"{MODEL_NAME}.joblib")
    train = pd.read_csv(ROOT / "data/processed/train.csv").drop(columns=TARGET)
    proba = pipe.predict_proba(add_features(train))[:, 1]
    config = {
        "model": MODEL_NAME,
        "high_risk_min": float(np.quantile(proba, 0.90)),
        "medium_risk_min": float(np.quantile(proba, 0.70)),
    }
    (ROOT / "models" / "serving_config.json").write_text(json.dumps(config, indent=2))
    train.sample(200, random_state=0).to_csv(ROOT / "models" / "shap_background.csv", index=False)
    print(config)


if __name__ == "__main__":
    main()
