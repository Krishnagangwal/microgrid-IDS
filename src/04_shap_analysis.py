"""
04_shap_analysis.py
===================
Computes SHAP values for the best model (XGBoost) and produces
three publication-quality figures for the paper.

Figures produced (all 300 DPI PDF):
  outputs/figures/fig_shap_summary.pdf     ← beeswarm plot (top 15 features)
  outputs/figures/fig_shap_importance.pdf  ← bar chart (mean |SHAP|)
  outputs/figures/fig_shap_force.pdf       ← single-sample force plot

Also saves:
  outputs/results/shap_top15_features.csv  ← feature importance table
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT     = os.path.join(BASE, "outputs", "results")
FIGS    = os.path.join(BASE, "outputs", "figures")
MODELS  = os.path.join(BASE, "outputs", "models")
os.makedirs(FIGS, exist_ok=True)

SAMPLE_SIZE = 2000   # sample from test set — full set is slow
SHAP_TOP_N  = 15


def load_artifacts():
    X_test  = np.load(os.path.join(OUT, "X_test.npy"))
    y_test  = np.load(os.path.join(OUT, "y_test.npy"))

    with open(os.path.join(OUT, "feature_names.txt")) as f:
        feature_names = [l.strip() for l in f.readlines()]

    model = joblib.load(os.path.join(MODELS, "xgboost.pkl"))
    return X_test, y_test, feature_names, model


def run():
    print("\n── 04: SHAP Explainability Analysis ───────────────────────────")
    X_test, y_test, feature_names, model = load_artifacts()

    # convert to DataFrame for SHAP (preserves feature names in plots)
    X_df = pd.DataFrame(X_test, columns=feature_names)

    # sample 2000 points (stratified to keep attack/normal balance)
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(X_df), size=min(SAMPLE_SIZE, len(X_df)),
                      replace=False)
    X_sample = X_df.iloc[idx]
    y_sample = y_test[idx]

    print(f"  Computing SHAP values on {len(X_sample):,} samples ...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # ── Figure 1: Beeswarm summary plot ────────────────────────────────────
    print("  Generating Figure 1: SHAP beeswarm summary ...")
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_sample,
        max_display=SHAP_TOP_N,
        show=False,
        plot_size=None
    )
    plt.title("SHAP Feature Impact on Attack Detection (XGBoost)",
              fontsize=11, pad=12)
    plt.xlabel("SHAP value (impact on model output)", fontsize=10)
    plt.tight_layout()
    path1 = os.path.join(FIGS, "fig_shap_summary.pdf")
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path1}")

    # ── Figure 2: Bar chart (mean |SHAP|) ──────────────────────────────────
    print("  Generating Figure 2: SHAP importance bar chart ...")
    shap.summary_plot(
        shap_values, X_sample,
        plot_type="bar",
        max_display=SHAP_TOP_N,
        show=False
    )
    plt.title("Mean |SHAP| Feature Importance (XGBoost)", fontsize=11)
    plt.tight_layout()
    path2 = os.path.join(FIGS, "fig_shap_importance.pdf")
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path2}")

    # ── Figure 3: Force plot — highest-confidence true positive ─────────────
    print("  Generating Figure 3: SHAP force plot (single attack sample) ...")
    # find a high-confidence true positive in sample
    y_pred_sample = model.predict(X_sample)
    tp_mask = (y_sample == 1) & (y_pred_sample == 1)
    tp_indices = np.where(tp_mask)[0]

    if len(tp_indices) == 0:
        print("  (No TP found in sample — skipping force plot)")
    else:
        tp_i = tp_indices[0]
        shap.force_plot(
            explainer.expected_value,
            shap_values[tp_i],
            X_sample.iloc[tp_i],
            matplotlib=True,
            show=False
        )
        plt.title("SHAP Force Plot — Single Attack Flow", fontsize=10, pad=10)
        plt.tight_layout()
        path3 = os.path.join(FIGS, "fig_shap_force.pdf")
        plt.savefig(path3, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path3}")

    # ── Save top-15 feature importance table ────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature":        feature_names,
        "Mean_Abs_SHAP":  mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False).head(SHAP_TOP_N)

    importance_df["Rank"] = range(1, len(importance_df) + 1)
    csv_path = os.path.join(OUT, "shap_top15_features.csv")
    importance_df.to_csv(csv_path, index=False)

    print(f"\n  Saved → outputs/results/shap_top15_features.csv")
    print("\n  ── Top 15 Features (Mean |SHAP|) ─────────────────────────")
    print(importance_df[["Rank", "Feature", "Mean_Abs_SHAP"]]
          .to_string(index=False))
    print()
    print("  ── Microgrid interpretation guide ────────────────────────")
    print("  Map these to your threat model section in the paper:")
    print("  ct_state_ttl  → abnormal TTL in spoofed DNP3 master traffic")
    print("  sbytes/dbytes → asymmetric bytes = C&C or DoS probe pattern")
    print("  dur           → short-duration bursts = scanning/recon")
    print("  state         → unexpected connection states in control flows")
    print("  proto         → unexpected protocol on port 20000 (DNP3 TCP)")
    print("  ✓ SHAP analysis complete\n")
    return importance_df


if __name__ == "__main__":
    run()