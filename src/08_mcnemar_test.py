"""
08_mcnemar_test.py
==================
Performs McNemar's statistical significance test between the top-2
models (Gradient Boosting vs XGBoost).

Without this, IEEE reviewers will say:
  "The authors cannot claim one model outperforms another without
   statistical evidence."

Output:
  outputs/results/mcnemar_results.csv
  Console: chi2, p-value, interpretation
"""

import os
import numpy as np
import pandas as pd
import joblib
from mlxtend.evaluate import mcnemar_table, mcnemar

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT    = os.path.join(BASE, "outputs", "results")
MODELS = os.path.join(BASE, "outputs", "models")


def run():
    print("\n── 08: McNemar Statistical Significance Test ──────────────────")

    X_test = np.load(os.path.join(OUT, "X_test.npy"))
    y_test = np.load(os.path.join(OUT, "y_test.npy"))

    # load the two best models
    gb_model  = joblib.load(os.path.join(MODELS, "gradient_boosting.pkl"))
    xgb_model = joblib.load(os.path.join(MODELS, "xgboost.pkl"))

    print("  Getting predictions from Gradient Boosting and XGBoost ...")
    y_pred_gb  = gb_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)

    # ── contingency table ───────────────────────────────────────────────────
    # tb[i,j] = number of samples where:
    #   i=0,j=0 → both wrong
    #   i=0,j=1 → GB wrong, XGB correct
    #   i=1,j=0 → GB correct, XGB wrong
    #   i=1,j=1 → both correct
    tb = mcnemar_table(y_target=y_test,
                       y_model1=y_pred_gb,
                       y_model2=y_pred_xgb)

    chi2, p_val = mcnemar(ary=tb, corrected=True)

    print(f"\n  Contingency table:")
    print(f"                       XGBoost Correct   XGBoost Wrong")
    print(f"  GBoost Correct   [{tb[1,1]:>12,}]       [{tb[1,0]:>9,}]")
    print(f"  GBoost Wrong     [{tb[0,1]:>12,}]       [{tb[0,0]:>9,}]")

    print(f"\n  McNemar's test (continuity-corrected):")
    print(f"    chi² = {chi2:.4f}")
    print(f"    p    = {p_val:.4f}")

    ALPHA = 0.05
    if p_val < ALPHA:
        interpretation = (
            f"Statistically SIGNIFICANT (p={p_val:.4f} < α={ALPHA}). "
            "The two models differ in their error patterns. "
            "Gradient Boosting and XGBoost make different mistakes — "
            "their combination in an ensemble is well-motivated."
        )
    else:
        interpretation = (
            f"NOT statistically significant (p={p_val:.4f} ≥ α={ALPHA}). "
            "Gradient Boosting and XGBoost are statistically equivalent "
            "at this performance level. Selection between them should be "
            "based on deployment criteria: latency, interpretability, "
            "or hardware constraints — not F1 score alone."
        )

    print(f"\n  Interpretation: {interpretation}")

    # save results
    results = pd.DataFrame([{
        "Model_1":           "Gradient Boosting",
        "Model_2":           "XGBoost",
        "chi2":              round(chi2, 4),
        "p_value":           round(p_val, 4),
        "alpha":             ALPHA,
        "significant":       p_val < ALPHA,
        "interpretation":    interpretation,
        "both_correct":      int(tb[1, 1]),
        "gb_only_correct":   int(tb[1, 0]),
        "xgb_only_correct":  int(tb[0, 1]),
        "both_wrong":        int(tb[0, 0]),
    }])
    out_path = os.path.join(OUT, "mcnemar_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  Saved → outputs/results/mcnemar_results.csv")

    print("\n  ── How to write this in the paper ──────────────────────────")
    print("  'To determine whether the performance difference between")
    print("  Gradient Boosting and XGBoost is statistically meaningful,")
    print("  we applied McNemar's test (continuity-corrected) on their")
    print(f"  predictions (χ²={chi2:.4f}, p={p_val:.4f}). This indicates")
    if p_val < ALPHA:
        print("  a statistically significant difference in error patterns,")
        print("  supporting their combination in the ensemble classifier.'")
    else:
        print("  statistical equivalence; model selection should therefore")
        print("  be guided by deployment criteria (latency, hardware).'")
    print("  ✓ McNemar test complete\n")
    return results


if __name__ == "__main__":
    run()