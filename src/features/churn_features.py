"""
churn_features.py
=================
Business Insights & Sales Forecasting Tool
Phase 3 — Feature Engineering

Builds a customer-level churn feature table from the cleaned Online Retail
dataset.

Steps:
  1. Load cleaned Online Retail CSV
  2. Compute Recency, Frequency, Monetary (RFM) features per customer
  3. Engineer additional predictive features
  4. Define binary churn label (1 = churned, 0 = retained)
  5. Save feature table to data/processed/churn_features.csv

Churn definition
----------------
A customer is considered churned if they made NO purchase in the
`churn_window_days` period before the snapshot date.
"""

import os
import yaml
import pandas as pd
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

INPUT_PATH    = os.path.join(BASE_DIR, cfg["paths"]["processed_online_retail"])
OUTPUT_PATH   = os.path.join(BASE_DIR, cfg["paths"]["churn_features"])
SNAPSHOT_DATE = pd.Timestamp(cfg["features"]["snapshot_date"])
CHURN_WINDOW  = cfg["features"]["churn_window_days"]


# ── Main Feature Builder ────────────────────────────────────────────────────

def build_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a customer-level DataFrame with RFM + engineered features and churn label."""

    df = df.copy()
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])

    # ── Define training window: 12 months before snapshot ──
    train_end   = SNAPSHOT_DATE - pd.Timedelta(days=CHURN_WINDOW)
    train_start = train_end - pd.Timedelta(days=365)

    df_train = df[(df["invoicedate"] >= train_start) & (df["invoicedate"] <= train_end)]
    df_label = df[(df["invoicedate"] > train_end)    & (df["invoicedate"] <= SNAPSHOT_DATE)]

    # ── RFM features (computed on training window) ──
    rfm = (
        df_train.groupby("customerid")
        .agg(
            recency  = ("invoicedate",  lambda x: (train_end - x.max()).days),
            frequency= ("invoiceno",    "nunique"),
            monetary = ("totalprice",   "sum"),
        )
        .reset_index()
    )

    # ── Additional engineered features ──
    basket_size = (
        df_train.groupby(["customerid", "invoiceno"])["quantity"]
        .sum()
        .reset_index()
        .groupby("customerid")["quantity"]
        .mean()
        .rename("avg_basket_size")
    )

    product_variety = (
        df_train.groupby("customerid")["stockcode"]
        .nunique()
        .rename("product_variety")
    )

    avg_unit_price = (
        df_train.groupby("customerid")["unitprice"]
        .mean()
        .rename("avg_unit_price")
    )

    country_mode = (
        df_train.groupby("customerid")["country"]
        .apply(lambda x: x.mode().iloc[0] if not x.empty else "Unknown")
        .rename("country")
    )

    rfm = (
        rfm
        .join(basket_size,    on="customerid")
        .join(product_variety, on="customerid")
        .join(avg_unit_price,  on="customerid")
        .join(country_mode,   on="customerid")
    )

    # ── Churn label ──
    # Customers who appear in the label window → retained (0); else churned (1)
    retained_customers = set(df_label["customerid"].unique())
    rfm["churned"] = rfm["customerid"].apply(
        lambda cid: 0 if cid in retained_customers else 1
    )

    # ── RFM quantile scores (1–5) using rank to avoid qcut duplicate issues ──
    bins = cfg["features"]["rfm_bins"]

    def safe_qcut_rank(series, n_bins, ascending=True):
        """Rank-based quantile scoring that is robust to ties/duplicates."""
        ranked = series.rank(method="first", ascending=ascending)
        return pd.qcut(ranked, n_bins, labels=range(1, n_bins + 1), duplicates="drop").astype(int)

    rfm["r_score"] = safe_qcut_rank(rfm["recency"],   bins, ascending=False)  # lower recency = better
    rfm["f_score"] = safe_qcut_rank(rfm["frequency"], bins, ascending=True)
    rfm["m_score"] = safe_qcut_rank(rfm["monetary"],  bins, ascending=True)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # ── Segment label (used later for Bias Check) ──
    score_min = int(rfm["rfm_score"].min())
    score_max = int(rfm["rfm_score"].max())
    # Build 4 equal-width bands dynamically from actual score range
    q25 = int(np.percentile(rfm["rfm_score"], 25))
    q50 = int(np.percentile(rfm["rfm_score"], 50))
    q75 = int(np.percentile(rfm["rfm_score"], 75))
    seg_bins   = [score_min - 1, q25, q50, q75, score_max]
    seg_labels = ["Low-Value", "Mid-Value", "High-Value", "Champions"]
    # Ensure unique edges
    seg_bins = sorted(set(seg_bins))
    seg_labels = seg_labels[: len(seg_bins) - 1]
    rfm["rfm_segment"] = pd.cut(
        rfm["rfm_score"],
        bins=seg_bins,
        labels=seg_labels,
        include_lowest=True,
    )


    print(f"  Total customers:  {len(rfm)}")
    print(f"  Churned (1):      {rfm['churned'].sum()} ({rfm['churned'].mean():.1%})")
    print(f"  Retained (0):     {(rfm['churned']==0).sum()} ({(rfm['churned']==0).mean():.1%})")

    return rfm


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print("[Churn Features] Loading cleaned Online Retail data...")
    df = pd.read_csv(INPUT_PATH, dtype={"customerid": str})

    print("[Churn Features] Building features...")
    features = build_churn_features(df)

    features.to_csv(OUTPUT_PATH, index=False)
    print(f"[Churn Features] Saved → {OUTPUT_PATH}")
    print(f"[Churn Features] Feature table shape: {features.shape}")


if __name__ == "__main__":
    main()
