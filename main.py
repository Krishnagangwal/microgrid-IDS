"""
main.py
=======
Runs the complete IEEE-grade IDS evaluation pipeline end-to-end.
Run this single file to reproduce every number in the paper.

Usage:
    python main.py

Steps:
    01 — Load & preprocess UNSW-NB15
    02 — Train all models (DT, GBoost, XGBoost, CatBoost, Ensemble)
    03 — 5-fold stratified cross-validation
    04 — SHAP explainability analysis
    05 — Inference latency benchmark
    06 — ROC & Precision-Recall curves
    07 — Per-attack-category detection rates
    08 — McNemar statistical significance test
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import importlib


STEPS = [
    ("01_load_and_preprocess",  "Preprocessing"),
    ("02_train_models",         "Model training"),
    ("03_cross_validation",     "5-fold cross-validation"),
    ("04_shap_analysis",        "SHAP explainability"),
    ("05_latency_benchmark",    "Latency benchmark"),
    ("06_roc_pr_curves",        "ROC / PR curves"),
    ("07_multiclass_analysis",  "Per-category analysis"),
    ("08_mcnemar_test",         "McNemar significance test"),
]


def main():
    print("=" * 62)
    print("  Microgrid IDS — Full IEEE Evaluation Pipeline")
    print("=" * 62)

    total_start = time.perf_counter()

    for module_name, label in STEPS:
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(module_name)
            mod.run()
        except FileNotFoundError as e:
            print(f"\n  [ERROR] {e}")
            print("  → Download UNSW-NB15 CSVs and place in data/ folder.")
            print("  → See README.md for download instructions.")
            sys.exit(1)
        except Exception as e:
            print(f"\n  [ERROR in {module_name}] {e}")
            raise
        elapsed = time.perf_counter() - t0
        print(f"  ✓ {label} — {elapsed:.1f}s")

    total = time.perf_counter() - total_start
    print("=" * 62)
    print(f"  Pipeline complete in {total:.1f}s")
    print()
    print("  Outputs:")
    print("    outputs/results/  ← all CSV metric tables")
    print("    outputs/figures/  ← all PDF figures (300 DPI)")
    print("    outputs/models/   ← all saved model .pkl files")
    print()
    print("  Paper figures ready:")
    print("    fig_shap_summary.pdf     → Figure (SHAP beeswarm)")
    print("    fig_shap_importance.pdf  → Figure (feature importance)")
    print("    fig_roc_pr.pdf           → Figure (ROC + PR curves)")
    print("    fig_latency.pdf          → Figure (latency benchmark)")
    print("    fig_category_detection.pdf → Figure (per-category)")
    print("=" * 62)


if __name__ == "__main__":
    main()