"""
Knowledge Ingestion Pipeline for Enterprise RAG
================================================
Run ONCE to embed business knowledge into ChromaDB vector store.
Re-run whenever KPI definitions or model registry are updated.

Usage:
    python -m src.chatbot.ingest_knowledge
"""

import json
import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

VECTOR_DB_PATH = "data/vector_db"
KPI_PATH       = "config/kpi_definitions.json"
MODEL_PATH     = "config/model_registry.json"

EMBED_FN = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def build_documents() -> list[dict]:
    """Convert all structured business knowledge into searchable text chunks."""
    docs = []

    # ── KPI Definitions ────────────────────────────────────────────────────────
    with open(KPI_PATH) as f:
        kpis = json.load(f)

    for key, kpi in kpis.items():
        if key.startswith("_"):
            continue
        label    = kpi.get("label", key)
        defn     = kpi.get("definition", "")
        formula  = kpi.get("sql_formula", "")
        unit     = kpi.get("unit", "")
        category = kpi.get("category", "")

        text = (
            f"KPI: {label}\n"
            f"Category: {category}\n"
            f"Definition: {defn}\n"
            f"Unit: {unit}\n"
            f"SQL Formula: {formula}"
        )
        docs.append({
            "id":   f"kpi_{key}",
            "text": text,
            "meta": {"type": "kpi", "key": key, "category": category}
        })

    # ── Model Registry ─────────────────────────────────────────────────────────
    with open(MODEL_PATH) as f:
        registry = json.load(f)

    for key, model in registry.items():
        if key.startswith("_"):
            continue
        name    = model.get("name", key)
        purpose = model.get("purpose", "")
        reason  = model.get("selection_reason", "")
        mtype   = model.get("type", "")
        output  = model.get("output", model.get("output_columns", ""))

        features_text = ""
        if "features" in model:
            features_text = "Features:\n" + "\n".join(
                f"  - {f['name']}: {f['description']}" for f in model["features"]
            )

        eval_text = ""
        if "evaluation" in model and "segment_results" in model["evaluation"]:
            segments = model["evaluation"]["segment_results"]
            eval_text = "Segment Performance:\n" + "\n".join(
                f"  - {seg}: Churn Rate {v['churn_rate']*100:.1f}%, AUC {v['auc_roc']:.4f}"
                for seg, v in segments.items()
            )
        elif "evaluation" in model and "performance" in model["evaluation"]:
            p = model["evaluation"]["performance"]
            eval_text = f"Performance: RMSE={p.get('rmse','N/A')}, MAE={p.get('mae','N/A')}, MAPE={p.get('mape','N/A')}"

        text = (
            f"ML Model: {name}\n"
            f"Type: {mtype}\n"
            f"Purpose: {purpose}\n"
            f"Selection Reason: {reason}\n"
            f"Output: {output}\n"
            f"{features_text}\n"
            f"{eval_text}"
        )
        docs.append({
            "id":   f"model_{key}",
            "text": text.strip(),
            "meta": {"type": "model", "key": key}
        })

    # ── Business Rules ─────────────────────────────────────────────────────────
    rules = [
        {
            "id":   "rule_revenue_formula",
            "text": (
                "Business Rule: Revenue Calculation\n"
                "ALWAYS compute revenue as SUM(price + freight_value) from order_items.\n"
                "NEVER use SUM(price) alone — this omits shipping charges and understates revenue.\n"
                "Table: order_items | Columns: price, freight_value\n"
                "Join with orders on order_id to filter by status."
            ),
            "meta": {"type": "rule", "key": "revenue_formula"}
        },
        {
            "id":   "rule_churn_definition",
            "text": (
                "Business Rule: Churn Definition\n"
                "A customer is considered churned if they have NOT placed any order within\n"
                "30 days of the snapshot date: 2011-12-10.\n"
                "Use this definition for all churn-related SQL queries and DNN model predictions.\n"
                "Snapshot date is fixed at 2011-12-10 for this dataset."
            ),
            "meta": {"type": "rule", "key": "churn_definition"}
        },
        {
            "id":   "rule_category_join",
            "text": (
                "Business Rule: Product Category Names\n"
                "Product categories are stored in Portuguese in the products table.\n"
                "Always JOIN product_category_name_translation on product_category_name\n"
                "to retrieve the English category name (product_category_name_english).\n"
                "Tables: products → product_category_name_translation"
            ),
            "meta": {"type": "rule", "key": "category_join"}
        },
        {
            "id":   "rule_delivered_filter",
            "text": (
                "Business Rule: Revenue Should Filter Delivered Orders\n"
                "When calculating revenue, always filter orders with:\n"
                "WHERE o.order_status = 'delivered'\n"
                "to exclude canceled, pending, or in-transit orders from financial metrics."
            ),
            "meta": {"type": "rule", "key": "delivered_filter"}
        },
        {
            "id":   "rule_rfm_segments",
            "text": (
                "Business Rule: Customer Segments (RFM)\n"
                "Customers are segmented by RFM score into 4 groups:\n"
                "  - Champions:  High Recency + High Frequency + High Monetary\n"
                "  - High-Value: Above-average monetary, moderate frequency\n"
                "  - Mid-Value:  Average across all RFM dimensions\n"
                "  - Low-Value:  Below-average across all RFM dimensions\n"
                "Segment data is available from the churn_features.csv and churn model evaluation files.\n"
                "Use tool_get_churn_risk_by_segment to get real segment-level churn rates."
            ),
            "meta": {"type": "rule", "key": "rfm_segments"}
        },
    ]
    docs.extend(rules)
    return docs


def ingest():
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    client     = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    # Delete and recreate for fresh ingestion
    try:
        client.delete_collection("bisft_knowledge")
    except Exception:
        pass

    collection = client.create_collection(
        name="bisft_knowledge",
        embedding_function=EMBED_FN,
    )

    docs = build_documents()
    print(f"Ingesting {len(docs)} documents into ChromaDB at '{VECTOR_DB_PATH}'...")

    collection.add(
        ids       = [d["id"]   for d in docs],
        documents = [d["text"] for d in docs],
        metadatas = [d["meta"] for d in docs],
    )
    print(f"Done. {collection.count()} documents embedded and persisted.")
    return collection.count()


if __name__ == "__main__":
    ingest()
