"""
ETL pipeline: CSV → PostgreSQL
Run after executing database/schema.sql in pgAdmin.

Requirements:
    pip install psycopg2-binary pandas numpy

Usage:
    python database/etl.py
"""

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import numpy as np
from pathlib import Path

# ── Fill these in before running ──────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "BISFT",
    "user":     "postgres",
    "password": "sohaibknows01#",
}
# ─────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent.parent
RAW_OLIST     = BASE_DIR / "data" / "raw" / "olist"
ONLINE_RETAIL = BASE_DIR / "data" / "processed" / "online_retail_cleaned.csv"


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def _nan_to_none(val):
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, str) and val.lower() in ("nan", "none", ""):
        return None
    return val


def _ts(val):
    """pandas Timestamp or NaT → Python datetime or None."""
    if pd.isna(val):
        return None
    return val.to_pydatetime()


def _cid(val):
    """Customer-id from float-or-string → zero-decimal string or None."""
    if _nan_to_none(val) is None:
        return None
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return None


# ═════════════════════════════════════════════════════════════
# ONLINE RETAIL
# ═════════════════════════════════════════════════════════════

def load_online_retail(conn):
    print("\n── Online Retail ────────────────────────────────────")
    df = pd.read_csv(ONLINE_RETAIL, dtype=str)
    df.columns = [c.lower().strip() for c in df.columns]

    # Normalise types
    df["quantity"]    = pd.to_numeric(df["quantity"],    errors="coerce")
    df["unitprice"]   = pd.to_numeric(df["unitprice"],   errors="coerce")
    df["totalprice"]  = pd.to_numeric(df["totalprice"],  errors="coerce")
    df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
    df["customerid"]  = df["customerid"].apply(_cid)

    # Drop rows missing critical keys
    df = df.dropna(subset=["invoiceno", "stockcode", "invoicedate"])
    df = df[df["stockcode"].str.strip() != ""]
    df["invoiceno"] = df["invoiceno"].str.strip()

    with conn.cursor() as cur:

        # 1. retail_customers — unique customerids with a country
        customers = (
            df.dropna(subset=["customerid"])
              .groupby("customerid", sort=False)
              .agg(country=("country", "first"))
              .reset_index()
        )
        rows = list(customers.itertuples(index=False, name=None))
        execute_values(
            cur,
            "INSERT INTO retail_customers (customerid, country) VALUES %s ON CONFLICT DO NOTHING",
            rows,
        )
        conn.commit()
        print(f"  retail_customers       : {len(rows):>8,} rows")

        # 2. retail_products — one row per stockcode
        products = (
            df.groupby("stockcode", sort=False)
              .agg(description=("description", "last"), latest_unit_price=("unitprice", "last"))
              .reset_index()
        )
        rows = [
            (r.stockcode, _nan_to_none(r.description), _nan_to_none(r.latest_unit_price))
            for r in products.itertuples(index=False)
        ]
        execute_values(
            cur,
            "INSERT INTO retail_products (stockcode, description, latest_unit_price) VALUES %s ON CONFLICT DO NOTHING",
            rows,
        )
        conn.commit()
        print(f"  retail_products        : {len(rows):>8,} rows")

        # 3. retail_invoices — one row per invoiceno
        invoices = (
            df.groupby("invoiceno", sort=False)
              .agg(customerid=("customerid", "first"), invoicedate=("invoicedate", "first"))
              .reset_index()
        )
        rows = [
            (
                r.invoiceno,
                _nan_to_none(r.customerid),
                _ts(r.invoicedate),
                str(r.invoiceno).startswith("C"),
            )
            for r in invoices.itertuples(index=False)
        ]
        execute_values(
            cur,
            "INSERT INTO retail_invoices (invoiceno, customerid, invoicedate, is_cancellation) VALUES %s ON CONFLICT DO NOTHING",
            rows,
        )
        conn.commit()
        print(f"  retail_invoices        : {len(rows):>8,} rows")

        # 4. retail_invoice_items — all line items (chunked)
        total = 0
        chunk_size = 5_000
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size]
            rows = [
                (
                    r.invoiceno,
                    r.stockcode,
                    int(r.quantity) if pd.notna(r.quantity) else 0,
                    float(r.unitprice) if pd.notna(r.unitprice) else 0.0,
                    float(r.totalprice) if pd.notna(r.totalprice) else 0.0,
                )
                for r in chunk.itertuples(index=False)
            ]
            execute_values(
                cur,
                """INSERT INTO retail_invoice_items
                   (invoiceno, stockcode, quantity, unitprice, totalprice)
                   VALUES %s""",
                rows,
            )
            total += len(rows)
        conn.commit()
        print(f"  retail_invoice_items   : {total:>8,} rows")


# ═════════════════════════════════════════════════════════════
# OLIST
# ═════════════════════════════════════════════════════════════

def load_olist(conn):
    print("\n── Olist ────────────────────────────────────────────")

    with conn.cursor() as cur:

        # 1. olist_product_category_translation
        df = pd.read_csv(RAW_OLIST / "product_category_name_translation.csv")
        rows = [
            (_nan_to_none(r.product_category_name), _nan_to_none(r.product_category_name_english))
            for r in df.itertuples(index=False)
        ]
        execute_values(
            cur,
            "INSERT INTO olist_product_category_translation (category_name_pt, category_name_en) VALUES %s ON CONFLICT DO NOTHING",
            rows,
        )
        conn.commit()
        print(f"  category_translation   : {len(rows):>8,} rows")

        # 2. olist_geolocation — large file, chunked read
        total = 0
        for chunk in pd.read_csv(
            RAW_OLIST / "olist_geolocation_dataset.csv",
            chunksize=10_000,
            dtype={"geolocation_zip_code_prefix": str},
        ):
            rows = [
                (
                    str(r.geolocation_zip_code_prefix).zfill(5),
                    _nan_to_none(r.geolocation_lat),
                    _nan_to_none(r.geolocation_lng),
                    _nan_to_none(r.geolocation_city),
                    _nan_to_none(r.geolocation_state),
                )
                for r in chunk.itertuples(index=False)
            ]
            execute_values(
                cur,
                "INSERT INTO olist_geolocation (zip_code_prefix, lat, lng, city, state) VALUES %s",
                rows,
            )
            total += len(rows)
        conn.commit()
        print(f"  olist_geolocation      : {total:>8,} rows")

        # 3. olist_sellers
        df = pd.read_csv(
            RAW_OLIST / "olist_sellers_dataset.csv",
            dtype={"seller_zip_code_prefix": str},
        )
        rows = [
            (
                r.seller_id,
                str(r.seller_zip_code_prefix).zfill(5),
                _nan_to_none(r.seller_city),
                _nan_to_none(r.seller_state),
            )
            for r in df.itertuples(index=False)
        ]
        execute_values(
            cur,
            "INSERT INTO olist_sellers (seller_id, zip_code_prefix, city, state) VALUES %s ON CONFLICT DO NOTHING",
            rows,
        )
        conn.commit()
        print(f"  olist_sellers          : {len(rows):>8,} rows")

        # 4. olist_customers
        df = pd.read_csv(
            RAW_OLIST / "olist_customers_dataset.csv",
            dtype={"customer_zip_code_prefix": str},
        )
        rows = [
            (
                r.customer_id,
                r.customer_unique_id,
                str(r.customer_zip_code_prefix).zfill(5),
                _nan_to_none(r.customer_city),
                _nan_to_none(r.customer_state),
            )
            for r in df.itertuples(index=False)
        ]
        execute_values(
            cur,
            "INSERT INTO olist_customers (customer_id, customer_unique_id, zip_code_prefix, city, state) VALUES %s ON CONFLICT DO NOTHING",
            rows,
        )
        conn.commit()
        print(f"  olist_customers        : {len(rows):>8,} rows")

        # 5. olist_products
        df = pd.read_csv(RAW_OLIST / "olist_products_dataset.csv")
        rows = [
            (
                r.product_id,
                _nan_to_none(r.product_category_name),
                _nan_to_none(r.product_name_lenght),
                _nan_to_none(r.product_description_lenght),
                _nan_to_none(r.product_photos_qty),
                _nan_to_none(r.product_weight_g),
                _nan_to_none(r.product_length_cm),
                _nan_to_none(r.product_height_cm),
                _nan_to_none(r.product_width_cm),
            )
            for r in df.itertuples(index=False)
        ]
        execute_values(
            cur,
            """INSERT INTO olist_products
               (product_id, category_name_pt, name_length, description_length,
                photos_qty, weight_g, length_cm, height_cm, width_cm)
               VALUES %s ON CONFLICT DO NOTHING""",
            rows,
        )
        conn.commit()
        print(f"  olist_products         : {len(rows):>8,} rows")

        # 6. olist_orders
        df = pd.read_csv(RAW_OLIST / "olist_orders_dataset.csv")
        ts_cols = [
            "order_purchase_timestamp", "order_approved_at",
            "order_delivered_carrier_date", "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
        for col in ts_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")

        rows = [
            (
                r.order_id, r.customer_id, r.order_status,
                _ts(r.order_purchase_timestamp),
                _ts(r.order_approved_at),
                _ts(r.order_delivered_carrier_date),
                _ts(r.order_delivered_customer_date),
                _ts(r.order_estimated_delivery_date),
            )
            for r in df.itertuples(index=False)
        ]
        execute_values(
            cur,
            """INSERT INTO olist_orders
               (order_id, customer_id, order_status, order_purchase_timestamp,
                order_approved_at, order_delivered_carrier_date,
                order_delivered_customer_date, order_estimated_delivery_date)
               VALUES %s ON CONFLICT DO NOTHING""",
            rows,
        )
        conn.commit()
        print(f"  olist_orders           : {len(rows):>8,} rows")

        # 7. olist_order_items
        df = pd.read_csv(RAW_OLIST / "olist_order_items_dataset.csv")
        df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"], errors="coerce")
        rows = [
            (
                r.order_id, int(r.order_item_id),
                _nan_to_none(r.product_id), _nan_to_none(r.seller_id),
                _ts(r.shipping_limit_date),
                _nan_to_none(r.price), _nan_to_none(r.freight_value),
            )
            for r in df.itertuples(index=False)
        ]
        execute_values(
            cur,
            """INSERT INTO olist_order_items
               (order_id, order_item_id, product_id, seller_id,
                shipping_limit_date, price, freight_value)
               VALUES %s ON CONFLICT DO NOTHING""",
            rows,
        )
        conn.commit()
        print(f"  olist_order_items      : {len(rows):>8,} rows")

        # 8. olist_order_payments
        df = pd.read_csv(RAW_OLIST / "olist_order_payments_dataset.csv")
        rows = [
            (
                r.order_id, int(r.payment_sequential),
                _nan_to_none(r.payment_type),
                _nan_to_none(r.payment_installments),
                _nan_to_none(r.payment_value),
            )
            for r in df.itertuples(index=False)
        ]
        execute_values(
            cur,
            """INSERT INTO olist_order_payments
               (order_id, payment_sequential, payment_type, payment_installments, payment_value)
               VALUES %s ON CONFLICT DO NOTHING""",
            rows,
        )
        conn.commit()
        print(f"  olist_order_payments   : {len(rows):>8,} rows")

        # 9. olist_order_reviews — deduplicate on review_id (keep last occurrence)
        df = pd.read_csv(RAW_OLIST / "olist_order_reviews_dataset.csv")
        df["review_creation_date"]    = pd.to_datetime(df["review_creation_date"],    errors="coerce")
        df["review_answer_timestamp"] = pd.to_datetime(df["review_answer_timestamp"], errors="coerce")
        df = df.drop_duplicates(subset=["review_id"], keep="last")

        # Fetch valid order_ids so we can skip orphaned reviews (avoids FK violation)
        cur.execute("SELECT order_id FROM olist_orders")
        valid_orders = {row[0] for row in cur.fetchall()}
        df = df[df["order_id"].isin(valid_orders)]

        rows = [
            (
                r.review_id, r.order_id,
                _nan_to_none(r.review_score),
                _nan_to_none(r.review_comment_title),
                _nan_to_none(r.review_comment_message),
                _ts(r.review_creation_date),
                _ts(r.review_answer_timestamp),
            )
            for r in df.itertuples(index=False)
        ]
        execute_values(
            cur,
            """INSERT INTO olist_order_reviews
               (review_id, order_id, review_score, review_comment_title,
                review_comment_message, review_creation_date, review_answer_timestamp)
               VALUES %s ON CONFLICT DO NOTHING""",
            rows,
        )
        conn.commit()
        print(f"  olist_order_reviews    : {len(rows):>8,} rows")


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════

def main():
    print("Connecting to PostgreSQL...")
    conn = get_conn()
    try:
        load_online_retail(conn)
        load_olist(conn)
        print("\n✓ ETL complete — all data loaded into PostgreSQL.")
    except Exception as e:
        conn.rollback()
        print(f"\n✗ ETL failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
