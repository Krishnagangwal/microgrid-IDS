"""
02_train_models.py
==================
Trains Decision Tree, Gradient Boosting, CatBoost, and XGBoost
on the preprocessed training data.

KEY FIX: Class-weighted training to handle UNSW-NB15's imbalanced splits.
  Train: 68% attack / 32% normal
  Test:  55% attack / 45% normal
  Without correction, models develop attack-prediction bias → FPR ~0.30
  With scale_pos_weight = n_normal/n_attack = 0.469, FPR drops to ~0.03

Also runs threshold sensitivity analysis for the operational guidance
section of the paper.

Saves:
  outputs/models/{dt,gb,xgb,cat,xgb_opt,ensemble}.pkl
  outputs/results/baseline_metrics.csv      ← Table I  (default threshold)
  outputs/results/threshold_analysis.csv    ← Table II (threshold sweep)
  outputs/figures/fig_threshold_sweep.pdf   ← Figure: FPR/Recall tradeoff
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              classification_report, confusion_matrix)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT    = os.path.join(BASE, "outputs", "results")
FIGS   = os.path.join(BASE, "outputs", "figures")
MODELS = os.path.join(BASE, "outputs", "models")
os.makedirs(FIGS, exist_ok=True)


def load_data():
    X_train = np.load(os.path.join(OUT, "X_train.npy"))
    X_test  = np.load(os.path.join(OUT, "X_test.npy"))
    y_train = np.load(os.path.join(OUT, "y_train.npy"))
    y_test  = np.load(os.path.join(OUT, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def compute_class_weight(y_train):
    """
    scale_pos_weight = n_negative / n_positive
    Corrects for the 2.13:1 attack:normal ratio in UNSW-NB15 training set.
    Equivalent to class_weight='balanced' but explicit and auditable.
    """
    n_normal = (y_train == 0).sum()
    n_attack = (y_train == 1).sum()
    spw = n_normal / n_attack
    print(f"  Class balance → Normal: {n_normal:,}  Attack: {n_attack:,}  "
          f"ratio: {n_attack/n_normal:.2f}:1")
    print(f"  scale_pos_weight = {n_normal}/{n_attack} = {spw:.4f}")
    print(f"  (Penalises false positives {1/spw:.1f}x more — "
          f"corrects attack-prediction bias)")
    return spw, n_normal, n_attack


def build_models(spw):
    """
    All models use class weighting to correct training set imbalance.
    Hyperparameters kept minimal to avoid overfitting — justified by
    the ablation study in script 08.
    """
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",   # sklearn equivalent of scale_pos_weight
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
            # Note: GBoost has no native class_weight; use sample_weight below
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=spw,      # key fix
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=100,
            learning_rate=0.5,
            depth=6,
            loss_function="Logloss",
            class_weights={0: 1/spw, 1: 1.0},   # key fix
            verbose=0,
            random_state=42
        ),
        "XGBoost (Optimised)": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,      # key fix
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),
    }


def get_sample_weights(y_train, spw):
    """
    For GradientBoosting which lacks native class_weight,
    pass per-sample weights manually.
    """
    weights = np.where(y_train == 1, 1.0, 1.0 / spw)
    return weights


def evaluate(model, X_test, y_test, name="", threshold=0.5):
    """Evaluate at a given probability threshold (default 0.5)."""
    if threshold == 0.5 or not hasattr(model, "predict_proba"):
        y_pred = model.predict(X_test)
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_test, y_pred, pos_label=1)
    f1   = f1_score(y_test, y_pred, pos_label=1)
    fpr  = fp / (fp + tn)
    acc  = accuracy_score(y_test, y_pred)

    return {
        "Model":            name,
        "Threshold":        threshold,
        "Accuracy":         round(acc,  4),
        "Precision_Attack": round(prec, 4),
        "Recall_Attack":    round(rec,  4),
        "F1_Attack":        round(f1,   4),
        "FPR":              round(fpr,  4),
        "TN": int(tn), "FP": int(fp),
        "FN": int(fn), "TP": int(tp),
    }


def threshold_sweep(model, X_test, y_test, name, thresholds):
    """Sweep probability thresholds and return FPR/Recall pairs."""
    if not hasattr(model, "predict_proba"):
        return []
    y_prob = model.predict_proba(X_test)[:, 1]
    rows = []
    for t in thresholds:
        m = evaluate(model, X_test, y_test, name, threshold=t)
        rows.append(m)
    return rows


def run():
    print("\n── 02: Train Models (with class-weight correction) ─────────────")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    spw, n_normal, n_attack = compute_class_weight(y_train)
    sample_weights = get_sample_weights(y_train, spw)

    models   = build_models(spw)
    results  = []
    fitted   = {}

    SAVE_NAMES = {
        "Decision Tree":        "decision_tree.pkl",
        "Gradient Boosting":    "gradient_boosting.pkl",
        "XGBoost":              "xgboost.pkl",
        "CatBoost":             "catboost.pkl",
        "XGBoost (Optimised)":  "xgboost_optimised.pkl",
    }

    for name, model in models.items():
        print(f"\n  Training {name} ...", end=" ", flush=True)

        # GradientBoosting needs sample_weight passed to fit()
        if name == "Gradient Boosting":
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        m = evaluate(model, X_test, y_test, name)
        results.append(m)
        fitted[name] = model

        path = os.path.join(MODELS, SAVE_NAMES[name])
        joblib.dump(model, path)
        print(f"Acc={m['Accuracy']}  F1={m['F1_Attack']}  "
              f"Prec={m['Precision_Attack']}  Rec={m['Recall_Attack']}  "
              f"FPR={m['FPR']}")

    # ── Ensemble Voting ──────────────────────────────────────────────────────
    print(f"\n  Training Ensemble (Voting) ...", end=" ", flush=True)
    ensemble = VotingClassifier(
        estimators=[
            ("gb",  fitted["Gradient Boosting"]),
            ("xgb", fitted["XGBoost"]),
            ("cat", fitted["CatBoost"]),
        ],
        voting="soft"
    )
    ensemble.fit(X_train, y_train, sample_weight=sample_weights)
    m = evaluate(ensemble, X_test, y_test, "Ensemble (Voting)")
    results.append(m)
    joblib.dump(ensemble, os.path.join(MODELS, "ensemble.pkl"))
    print(f"Acc={m['Accuracy']}  F1={m['F1_Attack']}  "
          f"Prec={m['Precision_Attack']}  Rec={m['Recall_Attack']}  "
          f"FPR={m['FPR']}")

    # ── Save baseline metrics ────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, "baseline_metrics.csv"), index=False)

    print(f"\n  ── Baseline Results (class-weighted, threshold=0.5) ────────")
    print(df[["Model","Accuracy","Precision_Attack",
              "Recall_Attack","F1_Attack","FPR"]].to_string(index=False))

    # ── Threshold sweep on best model (XGBoost Optimised) ───────────────────
    print(f"\n  Running threshold sensitivity sweep (XGBoost Optimised) ...")
    thresholds = np.arange(0.10, 0.91, 0.05).round(2)
    sweep_rows = threshold_sweep(
        fitted["XGBoost (Optimised)"], X_test, y_test,
        "XGBoost (Optimised)", thresholds
    )
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(os.path.join(OUT, "threshold_analysis.csv"), index=False)

    # ── Figure: Threshold sweep plot ────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    ax1.plot(sweep_df["Threshold"], sweep_df["Recall_Attack"],
             color="#1D9E75", marker="o", ms=4, linewidth=1.5,
             label="Recall (attack)")
    ax1.plot(sweep_df["Threshold"], sweep_df["Precision_Attack"],
             color="#534AB7", marker="s", ms=4, linewidth=1.5,
             label="Precision (attack)")
    ax1.plot(sweep_df["Threshold"], sweep_df["F1_Attack"],
             color="#185FA5", marker="^", ms=4, linewidth=1.5,
             label="F1 (attack)")
    ax2.plot(sweep_df["Threshold"], sweep_df["FPR"],
             color="#E24B4A", marker="D", ms=4, linewidth=1.5,
             linestyle="--", label="FPR (right axis)")

    # mark the default threshold
    ax1.axvline(0.5, color="gray", linestyle=":", linewidth=1,
                label="Default (0.5)")

    # find the threshold where FPR first drops below 0.05
    low_fpr = sweep_df[sweep_df["FPR"] <= 0.05]
    if not low_fpr.empty:
        t_rt = low_fpr.iloc[-1]["Threshold"]
        ax1.axvline(t_rt, color="orange", linestyle="--", linewidth=1,
                    label=f"FPR≤5% at t={t_rt}")

    ax1.set_xlabel("Classification threshold", fontsize=10)
    ax1.set_ylabel("Precision / Recall / F1", fontsize=10)
    ax2.set_ylabel("False Positive Rate (FPR)", fontsize=10, color="#E24B4A")
    ax2.tick_params(axis="y", labelcolor="#E24B4A")
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=8, loc="center left")

    ax1.set_title(
        "XGBoost (Optimised): Detection Performance vs. Classification Threshold\n"
        "Operators can tune this tradeoff for their deployment context",
        fontsize=9, pad=10
    )
    plt.tight_layout()
    fig_path = os.path.join(FIGS, "fig_threshold_sweep.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → outputs/figures/fig_threshold_sweep.pdf")
    print(f"  Saved → outputs/results/threshold_analysis.csv")

    # ── Print operational recommendations ────────────────────────────────────
    print(f"\n  ── Operational threshold recommendations ───────────────────")
    if not low_fpr.empty:
        row_rt = low_fpr.iloc[-1]
        print(f"  For FPR ≤ 5%: threshold = {row_rt['Threshold']}  "
              f"→ Recall={row_rt['Recall_Attack']}  "
              f"F1={row_rt['F1_Attack']}")
    row_50 = sweep_df[sweep_df["Threshold"] == 0.50]
    if not row_50.empty:
        r = row_50.iloc[0]
        print(f"  Default (0.50):  FPR={r['FPR']}  "
              f"Recall={r['Recall_Attack']}  F1={r['F1_Attack']}")

    print("""
  Write in paper (Operational Guidance section):
  "The classification threshold governs the precision-recall tradeoff.
  At the default threshold of 0.5, the model achieves F1=[X] with
  FPR=[Y]%. For high-criticality microgrids (e.g., hospital islanded
  networks), operators may increase the threshold to prioritise
  precision and reduce alert fatigue. For maximum threat coverage,
  lowering the threshold to [T] captures [R]% of attacks at FPR=[F]%,
  providing a configurable deployment envelope."
""")
    print("  ✓ Training complete\n")
    return fitted, ensemble


if __name__ == "__main__":
    run()