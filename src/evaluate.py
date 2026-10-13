"""Step 6: Final evaluation on the held-out test set.

Run from the project root:  python -m src.evaluate
This is the first and only time the test set is scored.
Writes reports/test_results.csv and charts to reports/figures/.
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from src.data_prep import TARGET
from src.features import add_features

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
MODELS = {"Logistic Regression": "logistic_regression", "XGBoost": "xgboost",
          "Random Forest": "random_forest"}


def capture_at(y, score, frac):
    """Share of all churners found if we contact the top `frac` riskiest customers."""
    n = int(round(len(y) * frac))
    top = np.argsort(-score, kind="stable")[:n]
    return y[top].sum() / y.sum(), y[top].mean()


def main():
    test = add_features(pd.read_csv(ROOT / "data/processed/test.csv"))
    y = test.pop(TARGET).to_numpy()
    base_rate = y.mean()

    scores = {name: joblib.load(ROOT / "models" / f"{f}.joblib").predict_proba(test)[:, 1]
              for name, f in MODELS.items()}
    # Rule-based baselines a retention team could use without any model
    m2m = (test["Contract"] == "Month-to-month").to_numpy()
    rule_simple = m2m.astype(float)                                 # yes/no flag
    rule_ranked = m2m * 1000 - test["tenure"].to_numpy()            # m2m, newest first
    scores["Rule: month-to-month, newest first"] = rule_ranked

    rows = []
    for name, s in scores.items():
        row = {"model": name, "roc_auc": roc_auc_score(y, s),
               "pr_auc": average_precision_score(y, s)}
        for frac in [0.10, 0.20, 0.30]:
            cap, prec = capture_at(y, s, frac)
            row[f"capture_top{int(frac*100)}"] = cap
            row[f"precision_top{int(frac*100)}"] = prec
        rows.append(row)
    res = pd.DataFrame(rows)
    res["lift_top10"] = res["precision_top10"] / base_rate
    res.to_csv(ROOT / "reports" / "test_results.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"Test set: {len(y)} customers, {y.sum()} churners ({base_rate:.1%})\n")
    print(res[["model", "roc_auc", "pr_auc", "capture_top10", "precision_top10",
               "lift_top10", "capture_top20", "capture_top30"]].round(3).to_string(index=False))

    # The plain yes/no rule flags every month-to-month customer
    flagged = m2m.mean()
    cap_rule, prec_rule = y[m2m].sum() / y.sum(), y[m2m].mean()
    cap_lr, prec_lr = capture_at(y, scores["Logistic Regression"], flagged)
    print(f"\nYes/no rule flags {flagged:.0%} of customers: catches {cap_rule:.1%} of churners, "
          f"precision {prec_rule:.1%}")
    print(f"LogReg contacting the same {flagged:.0%}:      catches {cap_lr:.1%} of churners, "
          f"precision {prec_lr:.1%}")
    # How many calls does each approach need to catch half the churners?
    for name in ["Logistic Regression", "Rule: month-to-month, newest first"]:
        order = np.argsort(-scores[name], kind="stable")
        need = np.searchsorted(np.cumsum(y[order]), 0.5 * y.sum()) + 1
        print(f"{name}: contacts needed to catch 50% of churners = {need} ({need/len(y):.0%})")

    # --- Charts ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for name, s in scores.items():
        ls = "--" if name.startswith("Rule") else "-"
        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(fpr, tpr, ls, label=f"{name} ({roc_auc_score(y, s):.3f})")
        p, r, _ = precision_recall_curve(y, s)
        axes[1].plot(r, p, ls, label=f"{name} ({average_precision_score(y, s):.3f})")
        order = np.argsort(-s, kind="stable")
        axes[2].plot(np.arange(1, len(y) + 1) / len(y) * 100,
                     np.cumsum(y[order]) / y.sum() * 100, ls, label=name)
    axes[0].plot([0, 1], [0, 1], ":", color="grey", label="random")
    axes[1].axhline(base_rate, ls=":", color="grey", label="random")
    axes[2].plot([0, 100], [0, 100], ":", color="grey", label="random")
    axes[2].axvline(10, color="grey", lw=0.8)
    axes[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    axes[2].set(title="Gains: % of churners caught vs % of customers contacted",
                xlabel="% of customers contacted (riskiest first)", ylabel="% of churners caught")
    for ax in axes:
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "test_evaluation.png", dpi=120)
    print(f"\nSaved {FIG / 'test_evaluation.png'}")


if __name__ == "__main__":
    main()
