-- ============================================================
-- BISFT — PostgreSQL Schema
-- Run this entire file in pgAdmin Query Editor once, in order.
-- Create the database first:  CREATE DATABASE bisft;
-- Then connect to bisft and run this file.
-- ============================================================


-- ─────────────────────────────────────────────────────────────
-- SECTION 1: ONLINE RETAIL  (UK gift retailer)
-- Source: data/processed/online_retail_cleaned.csv
-- ─────────────────────────────────────────────────────────────

CREATE TABLE retail_customers (
    customerid          VARCHAR(20)     PRIMARY KEY,
    country             VARCHAR(100)    NOT NULL
);

CREATE TABLE retail_products (
    stockcode           VARCHAR(20)     PRIMARY KEY,
    description         TEXT,
    latest_unit_price   NUMERIC(10, 2)
);

-- invoiceno starting with 'C' are cancellation invoices
CREATE TABLE retail_invoices (
    invoiceno           VARCHAR(20)     PRIMARY KEY,
    customerid          VARCHAR(20)     REFERENCES retail_customers(customerid),
    invoicedate         TIMESTAMP       NOT NULL,
    is_cancellation     BOOLEAN         NOT NULL DEFAULT FALSE
);

CREATE TABLE retail_invoice_items (
    id                  SERIAL          PRIMARY KEY,
    invoiceno           VARCHAR(20)     NOT NULL REFERENCES retail_invoices(invoiceno),
    stockcode           VARCHAR(20)     REFERENCES retail_products(stockcode),
    description         TEXT,
    quantity            INTEGER         NOT NULL,
    unitprice           NUMERIC(10, 2)  NOT NULL,
    totalprice          NUMERIC(12, 2)  NOT NULL
);

CREATE INDEX idx_rii_invoiceno  ON retail_invoice_items(invoiceno);
CREATE INDEX idx_rii_stockcode  ON retail_invoice_items(stockcode);
CREATE INDEX idx_ri_customerid  ON retail_invoices(customerid);
CREATE INDEX idx_ri_date        ON retail_invoices(invoicedate);


-- ─────────────────────────────────────────────────────────────
-- SECTION 2: OLIST  (Brazilian e-commerce platform)
-- Source: data/raw/olist/*.csv
-- ─────────────────────────────────────────────────────────────

CREATE TABLE olist_product_category_translation (
    category_name_pt    VARCHAR(100)    PRIMARY KEY,
    category_name_en    VARCHAR(100)
);

-- One zip code prefix can map to multiple lat/lng points
CREATE TABLE olist_geolocation (
    id                  SERIAL          PRIMARY KEY,
    zip_code_prefix     VARCHAR(10)     NOT NULL,
    lat                 NUMERIC(10, 6),
    lng                 NUMERIC(10, 6),
    city                VARCHAR(100),
    state               CHAR(2)
);

CREATE INDEX idx_geo_zip ON olist_geolocation(zip_code_prefix);

CREATE TABLE olist_sellers (
    seller_id           VARCHAR(50)     PRIMARY KEY,
    zip_code_prefix     VARCHAR(10),
    city                VARCHAR(100),
    state               CHAR(2)
);

-- customer_id is order-scoped; customer_unique_id is the real person
CREATE TABLE olist_customers (
    customer_id         VARCHAR(50)     PRIMARY KEY,
    customer_unique_id  VARCHAR(50)     NOT NULL,
    zip_code_prefix     VARCHAR(10),
    city                VARCHAR(100),
    state               CHAR(2)
);

CREATE INDEX idx_cust_unique_id ON olist_customers(customer_unique_id);

-- category_name_pt is stored as-is; join with olist_product_category_translation for English name
CREATE TABLE olist_products (
    product_id          VARCHAR(50)     PRIMARY KEY,
    category_name_pt    VARCHAR(100),
    name_length         INTEGER,
    description_length  INTEGER,
    photos_qty          INTEGER,
    weight_g            INTEGER,
    length_cm           NUMERIC(8, 2),
    height_cm           NUMERIC(8, 2),
    width_cm            NUMERIC(8, 2)
);

CREATE TABLE olist_orders (
    order_id                        VARCHAR(50)     PRIMARY KEY,
    customer_id                     VARCHAR(50)     NOT NULL REFERENCES olist_customers(customer_id),
    order_status                    VARCHAR(30)     NOT NULL,
    order_purchase_timestamp        TIMESTAMP,
    order_approved_at               TIMESTAMP,
    order_delivered_carrier_date    TIMESTAMP,
    order_delivered_customer_date   TIMESTAMP,
    order_estimated_delivery_date   TIMESTAMP
);

CREATE INDEX idx_orders_customer_id ON olist_orders(customer_id);
CREATE INDEX idx_orders_purchase_ts ON olist_orders(order_purchase_timestamp);
CREATE INDEX idx_orders_status      ON olist_orders(order_status);

CREATE TABLE olist_order_items (
    order_id            VARCHAR(50)     NOT NULL REFERENCES olist_orders(order_id),
    order_item_id       INTEGER         NOT NULL,
    product_id          VARCHAR(50)     REFERENCES olist_products(product_id),
    seller_id           VARCHAR(50)     REFERENCES olist_sellers(seller_id),
    shipping_limit_date TIMESTAMP,
    price               NUMERIC(10, 2),
    freight_value       NUMERIC(10, 2),
    PRIMARY KEY (order_id, order_item_id)
);

CREATE INDEX idx_items_product_id ON olist_order_items(product_id);
CREATE INDEX idx_items_seller_id  ON olist_order_items(seller_id);

CREATE TABLE olist_order_payments (
    order_id                VARCHAR(50)     NOT NULL REFERENCES olist_orders(order_id),
    payment_sequential      INTEGER         NOT NULL,
    payment_type            VARCHAR(30),
    payment_installments    INTEGER,
    payment_value           NUMERIC(10, 2),
    PRIMARY KEY (order_id, payment_sequential)
);

CREATE TABLE olist_order_reviews (
    review_id               VARCHAR(50)     PRIMARY KEY,
    order_id                VARCHAR(50)     NOT NULL REFERENCES olist_orders(order_id),
    review_score            SMALLINT        CHECK (review_score BETWEEN 1 AND 5),
    review_comment_title    TEXT,
    review_comment_message  TEXT,
    review_creation_date    TIMESTAMP,
    review_answer_timestamp TIMESTAMP
);

CREATE INDEX idx_reviews_order_id ON olist_order_reviews(order_id);
