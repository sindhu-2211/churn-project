"""Step 2: Exploratory data analysis of the Telco churn dataset.

Run from the project root:  python src/eda.py
Prints key numbers and saves charts to reports/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "telco_churn.csv"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["churn_flag"] = (df["Churn"] == "Yes").astype(int)

print(f"Shape: {df.shape}")
print(f"Overall churn rate: {df['churn_flag'].mean():.1%}")

# Data-quality check: TotalCharges is read as text because some rows are blank
blank_tc = (df["TotalCharges"].str.strip() == "").sum()
print(f"Blank TotalCharges rows: {blank_tc} (tenure of those rows: "
      f"{sorted(df.loc[df['TotalCharges'].str.strip() == '', 'tenure'].unique())})")
print(f"Duplicate customerIDs: {df['customerID'].duplicated().sum()}")


def churn_by(col):
    t = df.groupby(col)["churn_flag"].agg(customers="size", churn_rate="mean")
    t["churn_rate"] = (t["churn_rate"] * 100).round(1)
    return t.sort_values("churn_rate", ascending=False)


for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport",
            "OnlineSecurity", "SeniorCitizen", "PaperlessBilling"]:
    print(f"\nChurn rate (%) by {col}:\n{churn_by(col)}")

bins = [0, 6, 12, 24, 48, 72]
labels = ["0-6", "7-12", "13-24", "25-48", "49-72"]
df["tenure_band"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
print(f"\nChurn rate (%) by tenure band:\n{churn_by('tenure_band').sort_index()}")

print("\nMedian MonthlyCharges by churn:")
print(df.groupby("Churn")["MonthlyCharges"].median())

# --- Charts ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col in zip(axes, ["Contract", "tenure_band", "InternetService"]):
    rates = df.groupby(col, observed=True)["churn_flag"].mean() * 100
    rates.plot(kind="bar", ax=ax, color="#3b6ea5", label="churn rate")
    ax.axhline(df["churn_flag"].mean() * 100, ls="--", color="grey", label="overall")
    ax.set_title(f"Churn rate by {col}")
    ax.set_ylabel("% churned")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
plt.tight_layout()
plt.savefig(FIG / "churn_by_segment.png", dpi=120)

fig, ax = plt.subplots(figsize=(7, 4.5))
for label, grp in df.groupby("Churn"):
    ax.hist(grp["MonthlyCharges"], bins=30, alpha=0.6, label=f"Churn={label}", density=True)
ax.set_title("Monthly charges: churners vs stayers")
ax.set_xlabel("Monthly charges ($)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "monthly_charges_by_churn.png", dpi=120)
print(f"\nSaved charts to {FIG}")
