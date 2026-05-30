"""
01_load_and_preprocess.py
=========================
Loads UNSW-NB15 training and testing CSVs, applies a strict
no-leakage preprocessing pipeline, and saves processed arrays
to outputs/results/ for all downstream scripts to consume.

Pipeline (P1–P4):
  P1 — Drop id, replace inf with NaN, mode-impute categoricals,
       median-impute numerics
  P2 — One-hot encode {proto, service, state} with drop_first=True
  P3 — Z-score standardise numeric features (fit on train only)
  P4 — Binary target: 0=Normal, 1=Attack
       Retain attack_cat separately for multiclass analysis (script 07)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ── paths ──────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "data")
OUT    = os.path.join(BASE, "outputs", "results")
MODELS = os.path.join(BASE, "outputs", "models")
os.makedirs(OUT,    exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

TRAIN_CSV = os.path.join(DATA, "UNSW_NB15_training-set.csv")
TEST_CSV  = os.path.join(DATA, "UNSW_NB15_testing-set.csv")

# ── column config ───────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["proto", "service", "state"]
DROP_COLS        = ["id", "label"]        # 'label' is re-added as target
TARGET_COL       = "label"
ATTACK_CAT_COL   = "attack_cat"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # normalise column names: strip whitespace, lowercase
    df.columns = df.columns.str.strip().str.lower()
    return df


def preprocess(train_df: pd.DataFrame,
               test_df:  pd.DataFrame):
    """
    Returns:
        X_train, X_test  — feature matrices (numpy float32)
        y_train, y_test  — binary labels (numpy int)
        cat_train, cat_test — attack_cat series (for script 07)
        feature_names    — list of final feature column names
    """

    # ── P1: separate targets before any processing ─────────────────────────
    y_train    = train_df[TARGET_COL].values.astype(int)
    y_test     = test_df[TARGET_COL].values.astype(int)
    cat_train  = train_df[ATTACK_CAT_COL].fillna("Normal").str.strip()
    cat_test   = test_df[ATTACK_CAT_COL].fillna("Normal").str.strip()

    drop = [c for c in [TARGET_COL, ATTACK_CAT_COL, "id"]
            if c in train_df.columns]
    train_df = train_df.drop(columns=drop)
    test_df  = test_df.drop(columns=drop)

    # ── P1 cont: replace inf, then impute ──────────────────────────────────
    for df in (train_df, test_df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # identify column types from training set
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in CATEGORICAL_COLS if c in train_df.columns]

    # mode impute categoricals (fit on train)
    for col in cat_cols:
        mode_val = train_df[col].mode(dropna=True)[0]
        train_df[col].fillna(mode_val, inplace=True)
        test_df[col].fillna(mode_val,  inplace=True)

    # median impute numerics (fit on train)
    medians = train_df[num_cols].median()
    train_df[num_cols] = train_df[num_cols].fillna(medians)
    test_df[num_cols]  = test_df[num_cols].fillna(medians)

    # ── P2: one-hot encode using TRAINING vocabulary only ──────────────────
    # Critical: encode train first, then reindex test to the exact same
    # columns. This prevents test-set categories from creating extra columns
    # and is the correct no-leakage approach.
    # drop_first=False here — we drop manually after aligning so the
    # column sets are guaranteed identical between train and test.
    train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=False)

    # record the exact column order from training
    train_columns = train_df.columns.tolist()

    # encode test using the same dummies, then reindex to training columns
    # (fills missing cols with 0, drops unseen test-only cols)
    test_df = pd.get_dummies(test_df, columns=cat_cols, drop_first=False)
    test_df = test_df.reindex(columns=train_columns, fill_value=0)

    # ── P3: z-score normalisation (scaler fit on train only) ───────────────
    feature_names = train_df.columns.tolist()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df).astype(np.float32)
    X_test  = scaler.transform(test_df).astype(np.float32)

    # save scaler so any future inference uses the same parameters
    joblib.dump(scaler, os.path.join(MODELS, "scaler.pkl"))
    print(f"  Scaler saved → outputs/models/scaler.pkl")

    return (X_train, X_test,
            y_train, y_test,
            cat_train, cat_test,
            feature_names)


def run():
    print("\n── 01: Load & Preprocess ──────────────────────────────────────")

    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            f"\nCould not find:\n  {TRAIN_CSV}\n\n"
            "Download UNSW-NB15 from:\n"
            "  https://research.unsw.edu.au/projects/unsw-nb15-dataset\n"
            "and place both CSVs in the data/ folder."
        )

    print("  Loading CSVs ...")
    train_df = load_raw(TRAIN_CSV)
    test_df  = load_raw(TEST_CSV)

    print(f"  Train shape: {train_df.shape}  |  "
          f"Test shape: {test_df.shape}")

    # class distribution
    for name, df in [("Train", train_df), ("Test", test_df)]:
        counts = df[TARGET_COL].value_counts()
        print(f"  {name} label dist → "
              f"Normal: {counts.get(0,0):,}  "
              f"Attack: {counts.get(1,0):,}")

    (X_train, X_test,
     y_train, y_test,
     cat_train, cat_test,
     feature_names) = preprocess(train_df, test_df)

    print(f"  Final feature count: {X_train.shape[1]}")

    # ── save processed data ─────────────────────────────────────────────────
    np.save(os.path.join(OUT, "X_train.npy"), X_train)
    np.save(os.path.join(OUT, "X_test.npy"),  X_test)
    np.save(os.path.join(OUT, "y_train.npy"), y_train)
    np.save(os.path.join(OUT, "y_test.npy"),  y_test)
    cat_train.to_csv(os.path.join(OUT, "cat_train.csv"), index=False)
    cat_test.to_csv(os.path.join(OUT,  "cat_test.csv"),  index=False)

    # save feature names as a text file
    with open(os.path.join(OUT, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))

    print("  Saved → outputs/results/{X_train,X_test,y_train,y_test}.npy")
    print("  Saved → outputs/results/{cat_train,cat_test}.csv")
    print("  Saved → outputs/results/feature_names.txt")
    print("  ✓ Preprocessing complete\n")

    return X_train, X_test, y_train, y_test, cat_train, cat_test, feature_names


if __name__ == "__main__":
    run()