"""Step 7: Explain predictions with SHAP.

Run from the project root:  python -m src.explain
- Global view: which features drive churn across all test customers.
- Local view: top reasons for one customer (reused by the API in step 8).

SHAP values are additive: base value + sum of SHAP values = the model's
log-odds for that customer. Positive pushes towards churn, negative away.
One-hot columns are summed back into their original feature, so a reason
reads "Contract = Month-to-month" rather than five separate dummy columns.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src.data_prep import TARGET
from src.features import CATEGORICAL, NUMERIC, add_features

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"

READABLE = {
    "tenure": "Tenure (months)", "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges", "num_services": "Number of services",
    "m2m_x_monthly_charges": "Month-to-month x monthly charges",
    "charge_vs_avg": "Bill vs lifetime average", "tenure_bucket": "Tenure bucket",
    "auto_pay": "Auto-pay", "fiber_no_support": "Fiber without tech support",
}


def feature_groups(prep):
    """Map each original feature to the encoded column indices it produced."""
    groups, i = {}, 0
    for col in NUMERIC:
        groups[col] = [i]
        i += 1
    for col, cats in zip(CATEGORICAL, prep.named_transformers_["cat"].categories_):
        n = 1 if len(cats) == 2 else len(cats)   # drop="if_binary" keeps one column
        groups[col] = list(range(i, i + n))
        i += n
    return groups


class ChurnExplainer:
    def __init__(self, pipe, background: pd.DataFrame):
        self.prep, self.model = pipe.named_steps["prep"], pipe.named_steps["model"]
        bg = self.prep.transform(add_features(background))
        self.explainer = shap.LinearExplainer(
            self.model, shap.maskers.Independent(bg, max_samples=len(bg)))
        self.groups = feature_groups(self.prep)

    def grouped_shap(self, df_feat: pd.DataFrame) -> pd.DataFrame:
        sv = self.explainer.shap_values(self.prep.transform(df_feat))
        sv = np.asarray(sv)
        return pd.DataFrame({f: sv[:, idx].sum(axis=1) for f, idx in self.groups.items()},
                            index=df_feat.index)

    def top_reasons(self, customer: pd.DataFrame, k: int = 3) -> list[dict]:
        """Top k features pushing this customer TOWARDS churn, in plain words."""
        feat = add_features(customer)
        row = self.grouped_shap(feat).iloc[0].sort_values(ascending=False)
        reasons = []
        for f, val in row.head(k).items():
            if val <= 0:
                break
            v = feat.iloc[0][f]
            v = round(float(v), 1) if f in NUMERIC else v
            reasons.append({"feature": READABLE.get(f, f), "value": v,
                            "impact": round(float(val), 3)})
        return reasons


def main():
    import matplotlib  # only needed for charts, not inside the API
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pipe = joblib.load(ROOT / "models" / "logistic_regression.joblib")
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    test = pd.read_csv(ROOT / "data/processed/test.csv")
    ex = ChurnExplainer(pipe, train.drop(columns=TARGET).sample(500, random_state=0))

    test_feat = add_features(test.drop(columns=TARGET))
    shap_df = ex.grouped_shap(test_feat)
    importance = shap_df.abs().mean().sort_values(ascending=False)
    importance.rename("mean_abs_shap").to_csv(ROOT / "reports" / "shap_importance.csv")
    print("Mean |SHAP| (log-odds) by feature, top 12:")
    print(importance.head(12).round(3).to_string())

    # Global chart
    top = importance.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([READABLE.get(f, f) for f in top.index], top.values, color="#3b6ea5")
    ax.set_xlabel("Average impact on churn log-odds (mean |SHAP|)")
    ax.set_title("What drives churn predictions (Logistic Regression, test set)")
    plt.tight_layout()
    plt.savefig(FIG / "shap_importance.png", dpi=120)

    # Direction chart: how contract/tenure values push the prediction
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, f in zip(axes, ["Contract", "InternetService"]):
        d = pd.DataFrame({"value": test_feat[f], "shap": shap_df[f]})
        d.groupby("value")["shap"].mean().sort_values().plot(kind="barh", ax=ax, color="#3b6ea5")
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title(f"{f}: push towards (+) or away from (-) churn")
        ax.set_xlabel("mean SHAP (log-odds)")
        ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIG / "shap_direction.png", dpi=120)

    # Local explanations for three example customers
    proba = pipe.predict_proba(test_feat)[:, 1]
    picks = {"highest risk": int(np.argmax(proba)),
             "median risk": int(np.argsort(proba)[len(proba) // 2]),
             "lowest risk": int(np.argmin(proba))}
    for label, i in picks.items():
        cust = test.drop(columns=TARGET).iloc[[i]]
        print(f"\n{label}: {cust['customerID'].iloc[0]}  P(churn)={proba[i]:.2f}  "
              f"actually churned={bool(test[TARGET].iloc[i])}")
        for r in ex.top_reasons(cust):
            print(f"   {r['feature']} = {r['value']}  (+{r['impact']})")
    print(f"\nSaved charts to {FIG}")


if __name__ == "__main__":
    main()
