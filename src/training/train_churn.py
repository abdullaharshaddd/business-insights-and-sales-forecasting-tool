"""
train_churn.py
==============
Business Insights & Sales Forecasting Tool
Phase 3 & 4 — Hyperparameter Tuning + Model Training

Pipeline:
  1. Load churn feature table
  2. Preprocess (encode categoricals, scale numerics)
  3. Handle class imbalance with SMOTE
  4. Hyperparameter grid search (learning_rate × batch_size × dropout)
  5. Train final model with best params on 70/15/15 split
  6. Save model, scaler, and hp results table
"""

import os
import yaml
import json
import warnings
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.model_selection   import train_test_split, StratifiedKFold
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.metrics           import (
    roc_auc_score, precision_score, recall_score,
    f1_score, classification_report,
)
from imblearn.over_sampling    import SMOTE
import joblib
import tensorflow as tf

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

VAL_CFG  = cfg["validation"]
HP_GRID  = cfg["hp_grid"]
FINAL_HP = cfg["final_hp"]

FEATURES_PATH = os.path.join(BASE_DIR, cfg["paths"]["churn_features"])
MODEL_DIR     = os.path.join(BASE_DIR, cfg["paths"]["model_dir"])
TABLES_DIR    = os.path.join(BASE_DIR, cfg["paths"]["reports_tables"])
FIGURES_DIR   = os.path.join(BASE_DIR, cfg["paths"]["reports_figures"])

# Ensure output directories exist
for d in [MODEL_DIR, TABLES_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Feature columns ─────────────────────────────────────────────────────────
NUMERIC_COLS  = [
    "recency", "frequency", "monetary",
    "avg_basket_size", "product_variety", "avg_unit_price",
    "r_score", "f_score", "m_score", "rfm_score",
]
LABEL_COL     = "churned"
CAT_COLS      = ["country"]    # will be label-encoded


# ── 1. Data Loading & Preprocessing ─────────────────────────────────────────

def load_and_preprocess():
    print("[Training] Loading feature table...")
    df = pd.read_csv(FEATURES_PATH, dtype={"customerid": str})

    # Fill any NaN in numeric columns with median
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Encode country
    le = LabelEncoder()
    df["country_enc"] = le.fit_transform(df["country"].astype(str))

    FEATURE_COLS = NUMERIC_COLS + ["country_enc"]

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values
    segments = df["rfm_segment"].values  # for bias check later

    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Class balance — Churned: {y.mean():.1%}  Retained: {(1-y).mean():.1%}")
    return X, y, segments, le, FEATURE_COLS


# ── 2. Train / Val / Test Split (70 / 15 / 15) ──────────────────────────────

def make_splits(X, y, segments):
    rs = VAL_CFG["random_state"]
    test_sz = VAL_CFG["test_size"]
    val_sz  = VAL_CFG["val_size"]

    X_tmp, X_test, y_tmp, y_test, seg_tmp, seg_test = train_test_split(
        X, y, segments, test_size=test_sz, random_state=rs, stratify=y
    )
    rel_val = val_sz / (1 - test_sz)
    X_train, X_val, y_train, y_val, seg_train, seg_val = train_test_split(
        X_tmp, y_tmp, seg_tmp, test_size=rel_val, random_state=rs, stratify=y_tmp
    )
    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    return (X_train, y_train, seg_train,
            X_val,   y_val,   seg_val,
            X_test,  y_test,  seg_test)


# ── 3. Scaling ───────────────────────────────────────────────────────────────

def scale(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    return X_train_s, X_val_s, X_test_s, scaler


# ── 4. SMOTE (apply only to training set) ────────────────────────────────────

def apply_smote(X_train, y_train):
    if cfg["imbalance"]["use_smote"]:
        sm = SMOTE(random_state=cfg["imbalance"]["random_state"])
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"  SMOTE — Churned: {y_res.sum()} | Retained: {(y_res==0).sum()}")
        return X_res, y_res
    return X_train, y_train


# ── 5. Hyperparameter Grid Search ────────────────────────────────────────────

def hp_grid_search(X_train, y_train, X_val, y_val, input_dim):
    """Run a condensed HP grid and return results as a DataFrame."""
    from src.models.churn_model import build_churn_model

    results = []
    combos  = [
        (lr, bs, dr, ep, lyr)
        for lr  in HP_GRID["learning_rate"]
        for bs  in HP_GRID["batch_size"]
        for dr  in HP_GRID["dropout"]
        for ep  in HP_GRID["epochs"]
        for lyr in HP_GRID["layers"]
    ]

    print(f"\n[HP Search] Total combinations: {len(combos)} — running on small epoch budget")
    # Limit to a representative subset for time efficiency (academic demonstration)
    import random; random.seed(42)
    combos_sample = random.sample(combos, min(12, len(combos)))

    for lr, bs, dr, ep, lyr in combos_sample:
        tf.keras.backend.clear_session()
        m = build_churn_model(input_dim=input_dim, layers=lyr, dropout=dr, learning_rate=lr)
        hist = m.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=ep,
            batch_size=bs,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        )
        val_auc = max(hist.history.get("val_auc_roc", [0]))
        results.append({
            "learning_rate": lr,
            "batch_size":    bs,
            "dropout":       dr,
            "epochs":        ep,
            "layers":        str(lyr),
            "val_auc_roc":   round(val_auc, 4),
        })
        print(f"  LR={lr} BS={bs} DR={dr} EP={ep} L={lyr} → val_AUC={val_auc:.4f}")

    df_results = pd.DataFrame(results).sort_values("val_auc_roc", ascending=False)
    path = os.path.join(TABLES_DIR, "hp_tuning_results.csv")
    df_results.to_csv(path, index=False)
    print(f"\n[HP Search] Results saved → {path}")
    return df_results


# ── 6. Final Training ─────────────────────────────────────────────────────────

def train_final_model(X_train, y_train, X_val, y_val, input_dim):
    from src.models.churn_model import build_churn_model

    tf.keras.backend.clear_session()
    model = build_churn_model(
        input_dim=input_dim,
        layers=FINAL_HP["layers"],
        dropout=FINAL_HP["dropout"],
        learning_rate=FINAL_HP["learning_rate"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_roc", patience=8, restore_best_weights=True, mode="max"
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "churn_dnn_best.h5"),
            monitor="val_auc_roc", save_best_only=True, mode="max", verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=FINAL_HP["epochs"],
        batch_size=FINAL_HP["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save training curves ──
    _plot_training_curves(history)

    # ── Also train baselines for comparison ──
    from src.models.churn_model import get_baseline_models
    baselines = get_baseline_models()
    bl_results = {}
    for name, clf in baselines.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]
        bl_results[name] = {
            "accuracy":  round(float(np.mean(y_pred == y_val)), 4),
            "precision": round(precision_score(y_val, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_val, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_val, y_pred, zero_division=0), 4),
            "auc_roc":   round(roc_auc_score(y_val, y_prob), 4),
        }
        joblib.dump(clf, os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}.pkl"))
        print(f"  [{name}] AUC={bl_results[name]['auc_roc']}")

    # Save baseline comparison
    bl_df = pd.DataFrame(bl_results).T
    bl_df.to_csv(os.path.join(TABLES_DIR, "baseline_comparison.csv"))

    return model, history, baselines


def _plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curve"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["auc_roc"],     label="Train AUC")
    axes[1].plot(history.history["val_auc_roc"], label="Val AUC")
    axes[1].set_title("AUC-ROC Curve"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Training curves → {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" BISFT — Customer Churn Model Training")
    print("=" * 60)

    X, y, segments, le, FEATURE_COLS = load_and_preprocess()
    (X_train, y_train, seg_train,
     X_val,   y_val,   seg_val,
     X_test,  y_test,  seg_test) = make_splits(X, y, segments)

    X_train_s, X_val_s, X_test_s, scaler = scale(X_train, X_val, X_test)
    X_train_sm, y_train_sm = apply_smote(X_train_s, y_train)

    input_dim = X_train_sm.shape[1]

    # Save test set for evaluation
    np.save(os.path.join(MODEL_DIR, "X_test.npy"),   X_test_s)
    np.save(os.path.join(MODEL_DIR, "y_test.npy"),   y_test)
    np.save(os.path.join(MODEL_DIR, "seg_test.npy"), np.array(seg_test))
    json.dump(FEATURE_COLS, open(os.path.join(MODEL_DIR, "feature_cols.json"), "w"))

    # HP grid search
    print("\n[Phase 3] Running Hyperparameter Grid Search...")
    hp_df = hp_grid_search(X_train_sm, y_train_sm, X_val_s, y_val, input_dim)
    print("\nTop 5 HP Combinations:")
    print(hp_df.head(5).to_string(index=False))

    # Final model
    print("\n[Phase 3] Training Final Model with best hyperparameters...")
    model, history, baselines = train_final_model(
        X_train_sm, y_train_sm, X_val_s, y_val, input_dim
    )

    print("\n✅ Training complete! Run evaluate_churn.py for full metrics.")


if __name__ == "__main__":
    main()
