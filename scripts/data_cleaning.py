
import os
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR        = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR  = os.path.join(BASE_DIR, "data", "processed")

ONLINE_RETAIL_PATH = os.path.join(RAW_DIR, "online_retail", "online_retail.xlsx")
OLIST_DIR          = os.path.join(RAW_DIR, "olist")

OUTPUT_ONLINE_RETAIL = os.path.join(PROCESSED_DIR, "online_retail_cleaned.csv")
OUTPUT_OLIST         = os.path.join(PROCESSED_DIR, "olist_merged_cleaned.csv")


# ══════════════════════════════════════════════
# 1. ONLINE RETAIL DATASET
# ══════════════════════════════════════════════

def clean_online_retail(path: str) -> pd.DataFrame:
    """Load and clean the Online Retail (.xlsx) dataset."""
    print("\n[Online Retail] Loading data...")
    df = pd.read_excel(path, dtype={"CustomerID": str})
    print(f"  Raw shape: {df.shape}")

    # ── Standardise column names ──
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ── Drop fully duplicate rows ──
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Duplicates removed: {before - len(df)}")

    # ── Drop rows with missing CustomerID or Description ──
    df.dropna(subset=["customerid", "description"], inplace=True)

    # ── Remove cancelled orders (InvoiceNo starts with 'C') ──
    df = df[~df["invoiceno"].astype(str).str.startswith("C")]

    # ── Remove non-positive Quantity and UnitPrice ──
    df = df[(df["quantity"] > 0) & (df["unitprice"] > 0)]

    # ── Parse InvoiceDate ──
    df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
    df.dropna(subset=["invoicedate"], inplace=True)

    # ── Derived column: TotalPrice ──
    df["totalprice"] = df["quantity"] * df["unitprice"]

    # ── Strip whitespace from string columns ──
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.strip()
            except AttributeError:
                pass

    print(f"  Clean shape: {df.shape}")
    return df


# ══════════════════════════════════════════════
# 2. OLIST DATASET
# ══════════════════════════════════════════════

def load_olist(olist_dir: str) -> dict:
    """Load all Olist CSV files into a dictionary of DataFrames."""
    files = {
        "orders":       "olist_orders_dataset.csv",
        "order_items":  "olist_order_items_dataset.csv",
        "payments":     "olist_order_payments_dataset.csv",
        "reviews":      "olist_order_reviews_dataset.csv",
        "customers":    "olist_customers_dataset.csv",
        "products":     "olist_products_dataset.csv",
        "sellers":      "olist_sellers_dataset.csv",
        "geo":          "olist_geolocation_dataset.csv",
        "translation":  "product_category_name_translation.csv",
    }
    dfs = {}
    for key, fname in files.items():
        fpath = os.path.join(olist_dir, fname)
        dfs[key] = pd.read_csv(fpath)
        print(f"  Loaded {fname}: {dfs[key].shape}")
    return dfs


def clean_olist(olist_dir: str) -> pd.DataFrame:
    """Merge and clean the Olist dataset."""
    print("\n[Olist] Loading data...")
    dfs = load_olist(olist_dir)

    df = dfs["orders"].merge(dfs["order_items"],  on="order_id",    how="left")
    df = df.merge(dfs["payments"],                on="order_id",    how="left")
    df = df.merge(dfs["customers"],               on="customer_id", how="left")
    df = df.merge(dfs["products"],                on="product_id",  how="left")
    df = df.merge(dfs["sellers"],                 on="seller_id",   how="left")
    df = df.merge(dfs["reviews"],                 on="order_id",    how="left")
    df = df.merge(
        dfs["translation"],
        on="product_category_name",
        how="left"
    )
    print(f"  Merged shape: {df.shape}")

    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Duplicates removed: {before - len(df)}")

    df = df[df["order_status"] == "delivered"]

    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "shipping_limit_date",
        "review_creation_date",
        "review_answer_timestamp",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df.dropna(subset=["order_id", "customer_id", "product_id"], inplace=True)

    if "review_score" in df.columns:
        median_score = df["review_score"].median()
        df["review_score"].fillna(median_score, inplace=True)

    geo_cols = [c for c in df.columns if "geolocation" in c]
    df.drop(columns=geo_cols, errors="ignore", inplace=True)

    # ── Derived column: delivery_days ──
    if {"order_purchase_timestamp", "order_delivered_customer_date"}.issubset(df.columns):
        df["delivery_days"] = (
            df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
        ).dt.days

    # ── Rename translated category column for clarity ──
    if "product_category_name_english" in df.columns:
        df.rename(
            columns={"product_category_name_english": "product_category_en"},
            inplace=True,
        )

    # ── Strip whitespace from string columns ──
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.strip()
            except AttributeError:
                pass

    print(f"  Clean shape: {df.shape}")
    return df


# ══════════════════════════════════════════════
# 3. MAIN
# ══════════════════════════════════════════════

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── Clean Online Retail ──
    df_retail = clean_online_retail(ONLINE_RETAIL_PATH)
    df_retail.to_csv(OUTPUT_ONLINE_RETAIL, index=False)
    print(f"\n[Online Retail] Saved → {OUTPUT_ONLINE_RETAIL}")

    # ── Clean Olist ──
    df_olist = clean_olist(OLIST_DIR)
    df_olist.to_csv(OUTPUT_OLIST, index=False)
    print(f"\n[Olist] Saved → {OUTPUT_OLIST}")

    print("\n✅ All cleaning complete.")


if __name__ == "__main__":
    main()
