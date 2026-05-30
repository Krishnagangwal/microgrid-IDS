"""
09_ablation.py
==============
Ablation study for the XGBoost classifier. Three experiments, all using
5-fold stratified CV on the training set (same split as script 03):

  A. n_estimators sweep [10, 25, 50, 100, 150, 200, 300, 500]
     — identifies the elbow/plateau in detection performance

  B. Preprocessing variants
     — no standardisation | min-max normalisation | z-score (current)
     — X_train.npy is z-score; raw features recovered via inverse_transform

  C. Class weight variants
     — no weighting (spw=1.0) | current (spw=0.469) | aggressive (spw=0.25)
     — reports both F1 and FPR to show the precision-recall tradeoff

Saves:
  outputs/results/ablation_results.csv
  outputs/figures/fig_ablation.pdf
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT    = os.path.join(BASE, "outputs", "results")
FIGS   = os.path.join(BASE, "outputs", "figures")
MODELS = os.path.join(BASE, "outputs", "models")
os.makedirs(FIGS, exist_ok=True)

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Base XGBoost hyperparameters — same as script 02
BASE_PARAMS = dict(
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=0.469,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

N_ESTIMATORS_SWEEP = [10, 25, 50, 100, 150, 200, 300, 500]


def _fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


f1_scorer  = make_scorer(f1_score, pos_label=1)
fpr_scorer = make_scorer(_fpr)

SCORING = {"f1": f1_scorer, "fpr": fpr_scorer}


def _cv(model, X, y):
    """5-fold CV → (f1_mean, f1_std, fpr_mean, fpr_std)."""
    res = cross_validate(model, X, y, cv=SKF, scoring=SCORING,
                         n_jobs=-1, return_train_score=False)
    return (
        float(res["test_f1"].mean()),  float(res["test_f1"].std()),
        float(res["test_fpr"].mean()), float(res["test_fpr"].std()),
    )


def _row(experiment, variant, f1m, f1s, fprm, fprs):
    return {
        "Experiment": experiment,
        "Variant":    variant,
        "F1_mean":    round(f1m,  4),
        "F1_std":     round(f1s,  4),
        "FPR_mean":   round(fprm, 4),
        "FPR_std":    round(fprs, 4),
    }


# ── Experiment A: n_estimators sweep ────────────────────────────────────────

def run_n_estimators(X_train, y_train):
    print("\n  A. n_estimators sweep")
    rows = []
    for n in N_ESTIMATORS_SWEEP:
        model = XGBClassifier(n_estimators=n, **BASE_PARAMS)
        f1m, f1s, fprm, fprs = _cv(model, X_train, y_train)
        rows.append(_row("n_estimators", str(n), f1m, f1s, fprm, fprs))
        print(f"    n_estimators={n:>3d}  "
              f"F1={f1m:.4f}±{f1s:.4f}  FPR={fprm:.4f}±{fprs:.4f}")
    return rows


# ── Experiment B: preprocessing variants ────────────────────────────────────

def run_preprocessing(X_train, y_train, scaler):
    print("\n  B. Preprocessing variants")
    X_raw   = scaler.inverse_transform(X_train)   # recover un-scaled features
    X_minmax = MinMaxScaler().fit_transform(X_raw)

    variants = [
        ("No standardisation",     X_raw),
        ("Min-max normalisation",  X_minmax),
        ("Z-score (current)",      X_train),
    ]
    rows = []
    for name, X in variants:
        model = XGBClassifier(n_estimators=100, **BASE_PARAMS)
        f1m, f1s, fprm, fprs = _cv(model, X, y_train)
        rows.append(_row("preprocessing", name, f1m, f1s, fprm, fprs))
        print(f"    {name:<30s}  "
              f"F1={f1m:.4f}±{f1s:.4f}  FPR={fprm:.4f}±{fprs:.4f}")
    return rows


# ── Experiment C: class weight variants ─────────────────────────────────────

def run_class_weight(X_train, y_train):
    print("\n  C. Class weight variants (scale_pos_weight)")
    variants = [
        ("No weighting  (spw=1.00)", 1.00),
        ("Current       (spw=0.469)", 0.469),
        ("Aggressive    (spw=0.25)",  0.25),
    ]
    rows = []
    for name, spw in variants:
        params = {**BASE_PARAMS, "scale_pos_weight": spw}
        model  = XGBClassifier(n_estimators=100, **params)
        f1m, f1s, fprm, fprs = _cv(model, X_train, y_train)
        rows.append(_row("class_weight", name.strip(), f1m, f1s, fprm, fprs))
        print(f"    {name:<35s}  "
              f"F1={f1m:.4f}±{f1s:.4f}  FPR={fprm:.4f}±{fprs:.4f}")
    return rows


# ── Figure ───────────────────────────────────────────────────────────────────

def make_figure(sweep_df, prep_df, weight_df):
    fig = plt.figure(figsize=(14, 11))
    gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)

    ax_top  = fig.add_subplot(gs[0, :])   # full-width top
    ax_prep = fig.add_subplot(gs[1, 0])
    ax_wt   = fig.add_subplot(gs[1, 1])

    # ── Panel A: n_estimators line plot ─────────────────────────────────────
    n_vals = sweep_df["Variant"].astype(int).values
    f1m    = sweep_df["F1_mean"].values
    f1s    = sweep_df["F1_std"].values

    ax_top.plot(n_vals, f1m, color="#185FA5", marker="o",
                ms=5, linewidth=1.8, label="F1 (mean)")
    ax_top.fill_between(n_vals, f1m - f1s, f1m + f1s,
                         alpha=0.15, color="#185FA5", label="±1 std")

    # mark the elbow: first n where F1 is within 0.002 of the plateau
    plateau = f1m.max()
    elbow_idx = next((i for i, v in enumerate(f1m)
                      if v >= plateau - 0.002), len(f1m) - 1)
    ax_top.axvline(n_vals[elbow_idx], color="orange", linestyle="--",
                   linewidth=1.2,
                   label=f"Plateau at n={n_vals[elbow_idx]}")

    ax_top.set_xlabel("n_estimators", fontsize=10)
    ax_top.set_ylabel("F1 (attack, 5-fold CV)", fontsize=10)
    ax_top.set_title("A.  n_estimators Sweep — XGBoost", fontsize=10, pad=8)
    ax_top.legend(fontsize=8)
    ax_top.set_ylim(f1m.min() - 0.01, min(1.0, f1m.max() + 0.01))
    ax_top.grid(alpha=0.3, linewidth=0.5)
    ax_top.set_xticks(n_vals)

    # ── Panel B: preprocessing bar chart ────────────────────────────────────
    labels_b  = [t.replace(" normalisation", "\nnormalisation")
                   .replace(" standardisation", "\nstandardisation")
                 for t in prep_df["Variant"]]
    f1m_b = prep_df["F1_mean"].values
    f1s_b = prep_df["F1_std"].values
    colors_b  = ["#E24B4A", "#BA7517", "#1D9E75"]

    bars_b = ax_prep.bar(labels_b, f1m_b, color=colors_b,
                          width=0.5, yerr=f1s_b,
                          capsize=4, error_kw={"linewidth": 1})
    ax_prep.set_ylabel("F1 (attack, 5-fold CV)", fontsize=10)
    ax_prep.set_title("B.  Preprocessing Variants", fontsize=10, pad=8)
    ymin_b = max(0, f1m_b.min() - 0.02)
    ax_prep.set_ylim(ymin_b, min(1.0, f1m_b.max() + 0.02))
    for bar, val in zip(bars_b, f1m_b):
        ax_prep.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", va="bottom",
                     fontsize=8, color="black")
    ax_prep.grid(axis="y", alpha=0.3, linewidth=0.5)

    # ── Panel C: class weight grouped bars (F1 + FPR) ───────────────────────
    labels_c = [v.split("(")[0].strip() for v in weight_df["Variant"]]
    f1m_c    = weight_df["F1_mean"].values
    fprm_c   = weight_df["FPR_mean"].values
    f1s_c    = weight_df["F1_std"].values
    fprs_c   = weight_df["FPR_std"].values

    x        = np.arange(len(labels_c))
    width    = 0.32

    ax_wt2   = ax_wt.twinx()

    bars_f1  = ax_wt.bar(x - width / 2, f1m_c, width,
                          color="#185FA5", alpha=0.85,
                          label="F1", yerr=f1s_c,
                          capsize=3, error_kw={"linewidth": 1})
    bars_fpr = ax_wt2.bar(x + width / 2, fprm_c, width,
                           color="#E24B4A", alpha=0.75,
                           label="FPR", yerr=fprs_c,
                           capsize=3, error_kw={"linewidth": 1})

    ax_wt.set_ylabel("F1 (attack)", fontsize=10, color="#185FA5")
    ax_wt.tick_params(axis="y", labelcolor="#185FA5")
    ax_wt2.set_ylabel("FPR", fontsize=10, color="#E24B4A")
    ax_wt2.tick_params(axis="y", labelcolor="#E24B4A")

    ax_wt.set_title("C.  Class Weight Variants", fontsize=10, pad=8)
    ax_wt.set_xticks(x)
    ax_wt.set_xticklabels(labels_c, fontsize=8)

    lines1, labels1 = ax_wt.get_legend_handles_labels()
    lines2, labels2 = ax_wt2.get_legend_handles_labels()
    ax_wt.legend(lines1 + lines2, labels1 + labels2,
                 fontsize=8, loc="upper right")
    ax_wt.grid(axis="y", alpha=0.3, linewidth=0.5)

    fig.suptitle("Ablation Study — XGBoost on UNSW-NB15 (5-fold CV)",
                 fontsize=12, y=1.01)

    path = os.path.join(FIGS, "fig_ablation.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


# ── run() ────────────────────────────────────────────────────────────────────

def run():
    print("\n── 09: Ablation Study ──────────────────────────────────────────")

    X_train = np.load(os.path.join(OUT, "X_train.npy"))
    y_train = np.load(os.path.join(OUT, "y_train.npy"))
    scaler  = joblib.load(os.path.join(MODELS, "scaler.pkl"))

    rows_a = run_n_estimators(X_train, y_train)
    rows_b = run_preprocessing(X_train, y_train, scaler)
    rows_c = run_class_weight(X_train, y_train)

    all_rows = rows_a + rows_b + rows_c
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved → outputs/results/ablation_results.csv")

    # ── figure ───────────────────────────────────────────────────────────────
    sweep_df  = df[df["Experiment"] == "n_estimators"].reset_index(drop=True)
    prep_df   = df[df["Experiment"] == "preprocessing"].reset_index(drop=True)
    weight_df = df[df["Experiment"] == "class_weight"].reset_index(drop=True)

    fig_path = make_figure(sweep_df, prep_df, weight_df)
    print(f"  Saved → outputs/figures/fig_ablation.pdf")

    # ── terminal summary table ────────────────────────────────────────────────
    print("\n  ── Summary Table ────────────────────────────────────────────")
    for exp, label in [("n_estimators", "A. n_estimators sweep"),
                        ("preprocessing", "B. Preprocessing"),
                        ("class_weight",  "C. Class weight")]:
        sub = df[df["Experiment"] == exp][
            ["Variant", "F1_mean", "F1_std", "FPR_mean", "FPR_std"]
        ].copy()
        sub["F1"]  = sub.apply(lambda r: f"{r.F1_mean:.4f} ± {r.F1_std:.4f}",  axis=1)
        sub["FPR"] = sub.apply(lambda r: f"{r.FPR_mean:.4f} ± {r.FPR_std:.4f}", axis=1)
        print(f"\n  {label}")
        print(sub[["Variant", "F1", "FPR"]].to_string(index=False))

    print("\n  ✓ Ablation study complete\n")
    return df


if __name__ == "__main__":
    run()
