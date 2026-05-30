# Microgrid IDS


**Machine Learning-Based Intrusion Detection for Critical Microgrid Networks**

---

## Quickstart

```bash
# 1. Clone / open folder in VS Code

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download UNSW-NB15 dataset (see below)

# 5. Run full pipeline
python main.py
```

---

## Dataset download

1. Go to: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Download:
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`
3. Place both files in the `data/` folder

---

## Project structure

```
microgrid_ids/
├── data/                          ← place UNSW-NB15 CSVs here
├── src/
│   ├── 01_load_and_preprocess.py  ← P1-P4 pipeline, no leakage
│   ├── 02_train_models.py         ← DT, GBoost, XGBoost, CatBoost
│   ├── 03_cross_validation.py     ← 5-fold stratified CV (mean ± std)
│   ├── 04_shap_analysis.py        ← SHAP beeswarm + importance + force
│   ├── 05_latency_benchmark.py    ← per-flow inference latency
│   ├── 06_roc_pr_curves.py        ← AUC-ROC + PR curves
│   ├── 07_multiclass_analysis.py  ← per-attack-category detection
│   └── 08_mcnemar_test.py         ← statistical significance test
├── outputs/
│   ├── figures/                   ← 300 DPI PDFs for paper
│   ├── models/                    ← saved .pkl files
│   └── results/                   ← CSV metric tables
├── main.py                        ← run everything
├── requirements.txt
└── README.md
```

---

## Outputs

| File | Paper table/figure |
|---|---|
| `results/baseline_metrics.csv` | Table I — baseline results |
| `results/cv_results.csv` | Table II — CV mean ± std |
| `results/latency_benchmark.csv` | Table III — latency |
| `results/per_category_detection.csv` | Table IV — per-category |
| `results/auc_scores.csv` | AUC-ROC values |
| `results/mcnemar_results.csv` | Statistical significance |
| `results/shap_top15_features.csv` | Top-15 SHAP features |
| `figures/fig_shap_summary.pdf` | Figure — SHAP beeswarm |
| `figures/fig_shap_importance.pdf` | Figure — feature importance |
| `figures/fig_roc_pr.pdf` | Figure — ROC + PR curves |
| `figures/fig_latency.pdf` | Figure — latency benchmark |
| `figures/fig_category_detection.pdf` | Figure — per-category |

---

## Reproducing individual steps

```bash
python src/01_load_and_preprocess.py
python src/03_cross_validation.py
python src/04_shap_analysis.py
# etc.
```

Each script is self-contained and loads from `outputs/results/`.

---

## Environment

Tested on Python 3.10+. Key library versions in `requirements.txt`.
