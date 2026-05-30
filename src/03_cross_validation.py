"""
03_cross_validation.py
======================
Replaces single train/test evaluation with 5-fold stratified CV.
Reports mean ± std for every metric — the IEEE standard.

Output:
  outputs/results/cv_results.csv   ← Table II in paper
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "outputs", "results")


def load_data():
    X_train = np.load(os.path.join(OUT, "X_train.npy"))
    y_train = np.load(os.path.join(OUT, "y_train.npy"))
    return X_train, y_train


def build_models():
    # scale_pos_weight = n_normal / n_attack = 56000 / 119341 = 0.469
    # Matches script 02 class weighting so CV results are comparable to test-set evaluation.
    # GradientBoosting has no native class_weight; its test-set run used sample_weight
    # passed to fit(), which cross_validate does not support here — left unweighted.
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=3, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=6, scale_pos_weight=0.469,
            use_label_encoder=False,
            eval_metric="logloss", random_state=42, n_jobs=-1),
        "CatBoost": CatBoostClassifier(
            iterations=100, learning_rate=0.5,
            depth=6, loss_function="Logloss",
            class_weights={0: 2.131, 1: 1.0},
            verbose=0, random_state=42),
    }


def run():
    print("\n── 03: 5-Fold Stratified Cross-Validation ─────────────────────")
    X_train, y_train = load_data()
    models = build_models()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy":       "accuracy",
        "f1_attack":      make_scorer(f1_score,       pos_label=1),
        "precision_atk":  make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall_atk":     make_scorer(recall_score,    pos_label=1),
    }

    rows = []
    for name, model in models.items():
        print(f"  CV → {name} (5 folds) ...", end=" ", flush=True)

        cv_res = cross_validate(
            model, X_train, y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        row = {"Model": name}
        for metric in ["accuracy", "f1_attack", "precision_atk", "recall_atk"]:
            vals = cv_res[f"test_{metric}"]
            row[f"{metric}_mean"] = round(vals.mean(), 4)
            row[f"{metric}_std"]  = round(vals.std(),  4)
            # formatted string for the paper table
            row[f"{metric}_paper"] = (
                f"{vals.mean():.4f} ± {vals.std():.4f}"
            )

        rows.append(row)
        print(
            f"F1 = {row['f1_attack_mean']:.4f} ± {row['f1_attack_std']:.4f}  |  "
            f"Acc = {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}"
        )

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "cv_results.csv")
    df.to_csv(out_path, index=False)

    print(f"\n  Saved → outputs/results/cv_results.csv")
    print("\n  ── Paper Table (mean ± std) ──────────────────────────────")
    paper_cols = ["Model",
                  "accuracy_paper", "f1_attack_paper",
                  "precision_atk_paper", "recall_atk_paper"]
    print(df[paper_cols].rename(columns={
        "accuracy_paper":      "Accuracy",
        "f1_attack_paper":     "F1 (Attack)",
        "precision_atk_paper": "Precision (Attack)",
        "recall_atk_paper":    "Recall (Attack)",
    }).to_string(index=False))
    print("  ✓ Cross-validation complete\n")
    return df


if __name__ == "__main__":
    run()