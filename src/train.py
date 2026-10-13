"""Step 5: Train and compare three models with 5-fold cross-validation.

Run from the project root:  python -m src.train
Only the training set is used here; the test set is saved for step 6.
Writes models/<name>.joblib and reports/cv_results.csv.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_prep import RANDOM_STATE, TARGET
from src.features import add_features, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]


def make_models(pos_weight: float) -> dict:
    # Class imbalance handled with class weights rather than SMOTE: it keeps the
    # real data untouched and errors on churners cost ~2.8x more during training.
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", C=0.5, max_iter=2000),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
    }


def main():
    train = add_features(pd.read_csv(ROOT / "data/processed/train.csv"))
    y = train.pop(TARGET)
    pos_weight = (y == 0).sum() / (y == 1).sum()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    (ROOT / "models").mkdir(exist_ok=True)
    for name, model in make_models(pos_weight).items():
        # Preprocessor sits inside the pipeline, so each CV fold fits its own encoder
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        scores = cross_validate(pipe, train, y, cv=cv,
                                scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision",
                                         "recall": "recall", "precision": "precision"})
        row = {"model": name}
        for m in ["roc_auc", "pr_auc", "recall", "precision"]:
            row[m] = scores[f"test_{m}"].mean()
            row[f"{m}_std"] = scores[f"test_{m}"].std()
        rows.append(row)
        print(f"{name:20s} ROC-AUC {row['roc_auc']:.3f} ± {row['roc_auc_std']:.3f} | "
              f"PR-AUC {row['pr_auc']:.3f} ± {row['pr_auc_std']:.3f} | "
              f"recall {row['recall']:.2f} | precision {row['precision']:.2f}")
        pipe.fit(train, y)  # refit on the full training set for step 6
        joblib.dump(pipe, ROOT / "models" / f"{name}.joblib")

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "reports" / "cv_results.csv", index=False)
    print(f"\nPositive-class weight used: {pos_weight:.2f}")
    # Prefer the simplest model whose PR-AUC is within one std of the best
    # (rows are ordered simplest first)
    best = results.loc[results["pr_auc"].idxmax()]
    chosen = results[results["pr_auc"] >= best["pr_auc"] - best["pr_auc_std"]].iloc[0]
    print(f"Highest PR-AUC: {best['model']}; simplest model within 1 std: {chosen['model']}")


if __name__ == "__main__":
    main()
