"""
07_multiclass_analysis.py
=========================
Uses the attack_cat labels (retained but unused in training) to
compute per-attack-category detection rates.

This answers the operational question:
  "Which threat types does the IDS actually catch?"
Maps UNSW-NB15 categories to microgrid threat types.

Output:
  outputs/results/per_category_detection.csv   ← Table IV in paper
  outputs/figures/fig_category_detection.pdf
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT    = os.path.join(BASE, "outputs", "results")
FIGS   = os.path.join(BASE, "outputs", "figures")
MODELS = os.path.join(BASE, "outputs", "models")

# Map UNSW-NB15 attack categories → microgrid threat context
MICROGRID_MAP = {
    "DoS":           "DoS on DNP3 master station",
    "Reconnaissance":"Network reconnaissance / port scanning",
    "Exploits":      "Protocol exploit / command injection",
    "Fuzzers":       "DNP3 protocol fuzzing",
    "Backdoors":     "Persistent backdoor / C2 channel",
    "Generic":       "Generic network attack",
    "Analysis":      "Traffic analysis / probing",
    "Shellcode":     "Remote code execution attempt",
    "Worms":         "Self-propagating malware",
    "Normal":        "(Normal traffic — not an attack)",
}


def run():
    print("\n── 07: Per-Attack-Category Analysis ───────────────────────────")

    X_test  = np.load(os.path.join(OUT, "X_test.npy"))
    y_test  = np.load(os.path.join(OUT, "y_test.npy"))
    cat_test = pd.read_csv(os.path.join(OUT, "cat_test.csv")).iloc[:, 0]

    model = joblib.load(os.path.join(MODELS, "xgboost.pkl"))
    y_pred = model.predict(X_test)

    # analyse only attack samples (label==1)
    attack_mask = y_test == 1
    cat_attack   = cat_test[attack_mask].reset_index(drop=True)
    pred_attack  = y_pred[attack_mask]

    rows = []
    for cat in sorted(cat_attack.unique()):
        if cat.strip().lower() == "normal":
            continue
        mask     = cat_attack == cat
        n_total  = mask.sum()
        n_detect = (pred_attack[mask] == 1).sum()
        n_miss   = n_total - n_detect
        det_rate = n_detect / n_total if n_total > 0 else 0

        microgrid = MICROGRID_MAP.get(cat, cat)
        rows.append({
            "Attack_Category":    cat,
            "Microgrid_Threat":   microgrid,
            "Total_Samples":      int(n_total),
            "Detected":           int(n_detect),
            "Missed":             int(n_miss),
            "Detection_Rate":     round(det_rate, 4),
            "Miss_Rate":          round(1 - det_rate, 4),
        })
        print(f"  {cat:15s} → {det_rate:.3f} detection  "
              f"({n_detect:,}/{n_total:,}  missed={n_miss:,})")

    df = pd.DataFrame(rows).sort_values("Detection_Rate", ascending=False)
    csv_path = os.path.join(OUT, "per_category_detection.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved → outputs/results/per_category_detection.csv")

    # ── Figure: horizontal bar chart ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.55)))

    colors = ["#2ecc71" if r >= 0.90 else
              "#f39c12" if r >= 0.80 else
              "#e74c3c"
              for r in df["Detection_Rate"]]

    bars = ax.barh(df["Attack_Category"], df["Detection_Rate"],
                   color=colors, height=0.6)

    ax.axvline(0.90, color="gray", linestyle="--",
               linewidth=1, label="90% threshold")
    ax.set_xlabel("Detection Rate (True Positive Rate per category)", fontsize=10)
    ax.set_title("Per-Attack-Category Detection Rate (XGBoost)",
                 fontsize=10, pad=10)
    ax.set_xlim(0, 1.08)
    ax.legend(fontsize=9)

    for bar, row in zip(bars, df.itertuples()):
        ax.text(bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{row.Detection_Rate:.1%}  (n={row.Total_Samples:,})",
                va="center", ha="left", fontsize=8, color="gray")

    plt.tight_layout()
    fig_path = os.path.join(FIGS, "fig_category_detection.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → outputs/figures/fig_category_detection.pdf")

    # ── Operational insight ──────────────────────────────────────────────────
    worst = df.iloc[-1]
    best  = df.iloc[0]
    print("\n  ── Key findings for paper ──────────────────────────────────")
    print(f"  Highest detection: {best['Attack_Category']} "
          f"({best['Detection_Rate']:.1%})")
    print(f"  Lowest detection:  {worst['Attack_Category']} "
          f"({worst['Detection_Rate']:.1%})")
    print("  Write: 'DoS and reconnaissance threats — the most critical")
    print("  for microgrid availability — achieve detection rates of X%")
    print("  and Y% respectively. The lower detection rate for [worst]")
    print("  reflects [explain: low sample count / feature overlap].")
    print("  Future work will focus on improving [worst] detection")
    print("  through protocol-specific feature engineering.'")
    print("  ✓ Category analysis complete\n")
    return df


if __name__ == "__main__":
    run()