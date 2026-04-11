"""
evaluate_churn.py
=================
Business Insights & Sales Forecasting Tool
Phase 4 — Evaluation, Error Analysis & Bias Check

Reads the saved test set and trained DNN, then produces:
  • Classification report (Precision / Recall / F1 per class)
  • AUC-ROC score
  • Confusion matrix heatmap
  • ROC curve plot
  • Precision-Recall curve plot
  • Bias Check: per-RFM-segment performance breakdown
  • Error analysis: top misclassified customer profiles
  • Comparison table: DNN vs. baseline models
"""

import os
import json
import warnings
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score,
)
import joblib
import tensorflow as tf

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
import yaml
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

MODEL_DIR   = os.path.join(BASE_DIR, cfg["paths"]["model_dir"])
EVAL_DIR    = os.path.join(BASE_DIR, cfg["paths"]["eval_dir"])
FIGURES_DIR = os.path.join(BASE_DIR, cfg["paths"]["reports_figures"])
TABLES_DIR  = os.path.join(BASE_DIR, cfg["paths"]["reports_tables"])

for d in [EVAL_DIR, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

THRESHOLD = 0.5   # classification threshold


# ── Load artifacts ─────────────────────────────────────────────────────────

def load_test_data():
    X_test  = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(MODEL_DIR, "y_test.npy"))
    seg_test = np.load(os.path.join(MODEL_DIR, "seg_test.npy"), allow_pickle=True)
    return X_test, y_test, seg_test


def load_dnn():
    path = os.path.join(MODEL_DIR, "churn_dnn_best.h5")
    return tf.keras.models.load_model(path)


# ── Core Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob, model_name="Model"):
    report = classification_report(y_true, y_pred, target_names=["Retained", "Churned"])
    auc    = roc_auc_score(y_true, y_prob)
    ap     = average_precision_score(y_true, y_prob)
    print(f"\n{'='*55}")
    print(f"  {model_name} — Test Set Performance")
    print(f"{'='*55}")
    print(report)
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Avg Prec : {ap:.4f}")

    row = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc":   round(auc, 4),
        "avg_prec":  round(ap, 4),
    }
    return row


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        ax=ax,
    )
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Confusion matrix → {path}")


def plot_roc_curve(y_true, proba_dict):
    """Plot ROC curves for multiple models on one figure."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y_prob in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Churn Classifier Comparison")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "roc_comparison.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ROC comparison → {path}")


def plot_pr_curve(y_true, proba_dict):
    """Plot Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, y_prob in proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Churn Classifiers")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pr_comparison.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  PR curves → {path}")


# ── Bias Check ────────────────────────────────────────────────────────────────

def bias_check(y_true, y_pred, y_prob, segments):
    """
    Compute per-segment performance to detect model bias.
    Segment column is the RFM segment (Low-Value, Mid-Value, High-Value, Champions).
    """
    print("\n[Bias Check] Per-RFM-Segment Performance")
    print("-" * 50)

    rows = []
    for seg in sorted(set(segments)):
        mask = segments == seg
        if mask.sum() < 10:
            continue
        r = {
            "segment":   seg,
            "n_samples": int(mask.sum()),
            "churn_rate": round(float(y_true[mask].mean()), 3),
            "precision": round(precision_score(y_true[mask], y_pred[mask], zero_division=0), 4),
            "recall":    round(recall_score(y_true[mask], y_pred[mask], zero_division=0), 4),
            "f1":        round(f1_score(y_true[mask], y_pred[mask], zero_division=0), 4),
            "auc_roc":   round(roc_auc_score(y_true[mask], y_prob[mask])
                               if len(np.unique(y_true[mask])) > 1 else float("nan"), 4),
        }
        rows.append(r)
        print(f"  {seg:14s}: F1={r['f1']:.4f}  AUC={r['auc_roc']}  n={r['n_samples']}")

    df_bias = pd.DataFrame(rows)
    df_bias.to_csv(os.path.join(EVAL_DIR, "bias_check_by_segment.csv"), index=False)

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        ax.bar(df_bias["segment"], df_bias[metric], color="steelblue", alpha=0.8)
        ax.set_title(metric.capitalize()); ax.set_xlabel("RFM Segment")
        ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Bias Check — Per-Segment Metrics", y=1.02, fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "bias_check_segments.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Bias check plot → {path}")

    return df_bias


# ── Error Analysis ────────────────────────────────────────────────────────────

def error_analysis(X_test, y_true, y_pred, y_prob, feature_cols):
    """Identify the most confidently wrong predictions."""
    # False Positives: predicted churn but actually retained
    fp_mask = (y_pred == 1) & (y_true == 0)
    # False Negatives: predicted retained but actually churned
    fn_mask = (y_pred == 0) & (y_true == 1)

    df_err = pd.DataFrame(X_test, columns=feature_cols)
    df_err["y_true"]    = y_true
    df_err["y_pred"]    = y_pred
    df_err["churn_prob"]= y_prob
    df_err["error_type"]= "Correct"
    df_err.loc[fp_mask, "error_type"] = "False Positive"
    df_err.loc[fn_mask, "error_type"] = "False Negative"

    # Top False Negatives (missed churners — highest risk blind spots)
    top_fn = df_err[df_err["error_type"] == "False Negative"].nsmallest(20, "churn_prob")
    # Top False Positives (false alarms — most confidently wrong)
    top_fp = df_err[df_err["error_type"] == "False Positive"].nlargest(20, "churn_prob")

    err_summary = pd.concat([top_fn, top_fp])
    err_path = os.path.join(EVAL_DIR, "error_analysis_samples.csv")
    err_summary.to_csv(err_path, index=False)

    print(f"\n[Error Analysis]")
    print(f"  False Positives (predicted churn, actually retained): {fp_mask.sum()}")
    print(f"  False Negatives (missed churners):                    {fn_mask.sum()}")
    print(f"  Top error samples → {err_path}")

    # FP vs FN feature profile comparison
    if fp_mask.sum() > 0 and fn_mask.sum() > 0:
        profile = pd.DataFrame({
            "False Positive avg": df_err[fp_mask][feature_cols].mean(),
            "False Negative avg": df_err[fn_mask][feature_cols].mean(),
        })
        profile.to_csv(os.path.join(EVAL_DIR, "error_profiles.csv"))
        print(f"  Error profiles → {os.path.join(EVAL_DIR, 'error_profiles.csv')}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print(" BISFT — Customer Churn Evaluation")
    print("=" * 55)

    X_test, y_test, seg_test = load_test_data()
    feature_cols = json.load(open(os.path.join(MODEL_DIR, "feature_cols.json")))

    # ── DNN ──────────────────────────────────────────────────────────────
    dnn     = load_dnn()
    dnn_prob = dnn.predict(X_test, verbose=0).ravel()
    dnn_pred = (dnn_prob >= THRESHOLD).astype(int)

    proba_dict = {"DNN (ChurnNet)": dnn_prob}

    # ── Baselines ─────────────────────────────────────────────────────────
    baseline_names = ["logistic_regression", "random_forest", "gradient_boosting"]
    baseline_labels = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    all_metrics = []

    for fname, label in zip(baseline_names, baseline_labels):
        pkl = os.path.join(MODEL_DIR, f"{fname}.pkl")
        if os.path.exists(pkl):
            clf  = joblib.load(pkl)
            prob = clf.predict_proba(X_test)[:, 1]
            pred = (prob >= THRESHOLD).astype(int)
            proba_dict[label] = prob
            row = compute_metrics(y_test, pred, prob, label)
            all_metrics.append(row)
            plot_confusion_matrix(y_test, pred, label)

    # ── DNN Metrics ───────────────────────────────────────────────────────
    row_dnn = compute_metrics(y_test, dnn_pred, dnn_prob, "DNN (ChurnNet)")
    all_metrics.append(row_dnn)
    plot_confusion_matrix(y_test, dnn_pred, "DNN_ChurnNet")

    # ── Comparison Plot ───────────────────────────────────────────────────
    plot_roc_curve(y_test, proba_dict)
    plot_pr_curve(y_test, proba_dict)

    # ── Final Metrics Table ───────────────────────────────────────────────
    df_metrics = pd.DataFrame(all_metrics).set_index("model")
    metrics_path = os.path.join(TABLES_DIR, "final_metrics.csv")
    df_metrics.to_csv(metrics_path)
    print("\n[Metrics] Final comparison table:")
    print(df_metrics.to_string())
    print(f"\n  Saved → {metrics_path}")

    # ── Bias Check ────────────────────────────────────────────────────────
    df_bias = bias_check(y_test, dnn_pred, dnn_prob, seg_test)

    # ── Error Analysis ────────────────────────────────────────────────────
    error_analysis(X_test, y_test, dnn_pred, dnn_prob, feature_cols)

    print("\n✅ Evaluation complete! Check reports/figures/ and evaluation/churn/")


if __name__ == "__main__":
    main()
