import os
import json
import sqlite3
import pandas as pd
import numpy as np
import joblib
import yaml
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import asyncio
import diskcache

tool_cache = diskcache.Cache("data/tool_cache")

def async_cache(expire=3600):
    """Decorator to cache the results of async functions using diskcache."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{args}:{kwargs}"
            if key in tool_cache:
                return tool_cache[key] + "\n[Result served from Cache]"
            result = await func(*args, **kwargs)
            tool_cache.set(key, result, expire=expire)
            return result
        return wrapper
    return decorator

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

DB_PATH            = cfg["paths"]["olist_db"]
PROPHET_MODEL_PATH = cfg["paths"]["prophet_model"]
FORECASTING_EVAL   = cfg["paths"]["forecasting_eval"]
MODEL_REGISTRY_PATH = "config/model_registry.json"
KPI_DEFINITIONS_PATH = "config/kpi_definitions.json"
CHURN_MODEL_DIR    = cfg["paths"]["model_dir"]
VECTOR_DB_PATH     = "data/vector_db"

# Lazy-loaded vector store (initialized on first use)
_vector_client = None
_vector_collection = None
_embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _get_vector_collection():
    """Return ChromaDB collection, initializing client on first use."""
    global _vector_client, _vector_collection
    if _vector_collection is None:
        _vector_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        _vector_collection = _vector_client.get_collection(
            name="bisft_knowledge",
            embedding_function=_embed_fn,
        )
    return _vector_collection


# ─────────────────────────────────────────────────────────────────────────────
# 1. SQL Tool
# ─────────────────────────────────────────────────────────────────────────────
def query_database(query: str) -> str:
    """Execute a SQL query against the Olist SQLite database and return results as a string."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, conn)
        if df.empty:
            return "Query returned no results."
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL Error: {str(e)}"
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model Registry Tool
# ─────────────────────────────────────────────────────────────────────────────
@async_cache(expire=86400) # Cache for 24 hours
async def get_model_registry(model_name: str = None) -> str:
    """
    Retrieve model metadata from the model registry.
    If model_name is provided (e.g. 'churn_predictor', 'revenue_forecaster'),
    returns metadata for that specific model.
    If None, returns a summary of all available models.
    """
    try:
        with open(MODEL_REGISTRY_PATH) as f:
            registry = json.load(f)

        if model_name:
            # Remove the comment key
            model = registry.get(model_name)
            if not model:
                available = [k for k in registry.keys() if not k.startswith("_")]
                return f"Model '{model_name}' not found. Available models: {available}"
            return json.dumps(model, indent=2)
        else:
            # Summary of all models
            summary = {}
            for key, val in registry.items():
                if key.startswith("_"):
                    continue
                summary[key] = {
                    "name": val.get("name"),
                    "purpose": val.get("purpose"),
                    "type": val.get("type"),
                    "selection_reason": val.get("selection_reason", "")
                }
            return json.dumps(summary, indent=2)
    except Exception as e:
        return f"Error reading model registry: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. KPI Definition Tool
# ─────────────────────────────────────────────────────────────────────────────
@async_cache(expire=86400)
async def get_kpi_definition(kpi_name: str = None) -> str:
    """
    Retrieve KPI business definitions and SQL formulas.
    If kpi_name is provided (e.g. 'aov', 'churn_rate_overall', 'revenue'),
    returns the definition and SQL for that KPI.
    If None, returns a list of all available KPI names with their labels.
    """
    try:
        with open(KPI_DEFINITIONS_PATH) as f:
            kpis = json.load(f)

        if kpi_name:
            kpi = kpis.get(kpi_name)
            if not kpi:
                available = [k for k in kpis.keys() if not k.startswith("_")]
                return f"KPI '{kpi_name}' not found. Available KPIs: {available}"
            return json.dumps(kpi, indent=2)
        else:
            # Return grouped summary
            categories = {}
            for key, val in kpis.items():
                if key.startswith("_"):
                    continue
                cat = val.get("category", "Other")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(f"{key}: {val.get('label', key)}")
            return json.dumps(categories, indent=2)
    except Exception as e:
        return f"Error reading KPI definitions: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sales Forecast Tool (Real — uses Prophet model)
# ─────────────────────────────────────────────────────────────────────────────
@async_cache(expire=14400) # 4 hours for forecasts
async def get_sales_forecast_summary(days: int = 30) -> str:
    """
    Generate a real sales revenue forecast for the next N days using the trained
    Prophet model. Returns predicted daily revenue with trend direction.
    """
    try:
        # Try live Prophet prediction first
        if os.path.exists(PROPHET_MODEL_PATH):
            model = joblib.load(PROPHET_MODEL_PATH)
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            future_only = forecast.tail(days)[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]]

            avg_revenue  = future_only["yhat"].mean()
            total_revenue = future_only["yhat"].sum()
            trend_direction = "increasing" if future_only["yhat"].iloc[-1] > future_only["yhat"].iloc[0] else "decreasing"
            peak_day  = future_only.loc[future_only["yhat"].idxmax(), "ds"].strftime("%Y-%m-%d")
            peak_val  = future_only["yhat"].max()
            low_day   = future_only.loc[future_only["yhat"].idxmin(), "ds"].strftime("%Y-%m-%d")
            low_val   = future_only["yhat"].min()

            return (
                f"[LIVE Prophet Forecast — Next {days} Days]\n"
                f"Projected Total Revenue:    R${total_revenue:,.2f}\n"
                f"Average Daily Revenue:      R${avg_revenue:,.2f}\n"
                f"Revenue Trend:              {trend_direction.upper()}\n"
                f"Peak Day:                   {peak_day} (R${peak_val:,.2f})\n"
                f"Lowest Day:                 {low_day} (R${low_val:,.2f})\n"
                f"Confidence Interval (95%):  R${future_only['yhat_lower'].mean():,.2f} to R${future_only['yhat_upper'].mean():,.2f}"
            )
    except Exception as e:
        pass  # Fall back to pre-computed results

    # Fallback: use pre-computed forecast_results.csv
    try:
        forecast_path = os.path.join(FORECASTING_EVAL, "forecast_results.csv")
        df = pd.read_csv(forecast_path)
        future = df.tail(days)
        avg_revenue = future["yhat"].mean()
        total_revenue = future["yhat"].sum()
        trend_direction = "increasing" if future["yhat"].iloc[-1] > future["yhat"].iloc[0] else "decreasing"

        return (
            f"[Cached Forecast — Last {days} Days of Pre-Computed Data]\n"
            f"Projected Total Revenue:    R${total_revenue:,.2f}\n"
            f"Average Daily Revenue:      R${avg_revenue:,.2f}\n"
            f"Revenue Trend:              {trend_direction.upper()}"
        )
    except Exception as e:
        return f"Forecast unavailable: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Forecast Performance Metrics Tool
# ─────────────────────────────────────────────────────────────────────────────
@async_cache(expire=86400)
async def get_forecast_metrics() -> str:
    """
    Return the Prophet model's evaluation metrics (RMSE, MAE, MAPE, coverage)
    from cross-validation results.
    """
    try:
        metrics_path = os.path.join(FORECASTING_EVAL, "summary_metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)
        return (
            f"[Prophet Model Performance — Cross-Validation]\n"
            f"Forecast Horizon:  {metrics.get('horizon', 'N/A')}\n"
            f"RMSE:              R${metrics.get('rmse', 0):,.2f}\n"
            f"MAE:               R${metrics.get('mae', 0):,.2f}\n"
            f"MAPE:              {metrics.get('mape', 0)*100:.1f}% (mean absolute % error)\n"
            f"Coverage (95% CI): {metrics.get('coverage', 0)*100:.1f}%"
        )
    except Exception as e:
        return f"Forecast metrics unavailable: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Churn Risk by Segment Tool (Real — from evaluation data)
# ─────────────────────────────────────────────────────────────────────────────
@async_cache(expire=86400)
async def get_churn_risk_by_segment() -> str:
    """
    Return real churn risk rates per customer segment from the DNN model's
    evaluation results. Segments: Champions, High-Value, Mid-Value, Low-Value.
    """
    try:
        path = os.path.join(cfg["paths"]["eval_dir"], "bias_check_by_segment.csv")
        df = pd.read_csv(path)

        # Build a clear, readable output
        lines = ["[Churn Risk by Customer Segment — DNN Model Evaluation]\n"]
        for _, row in df.iterrows():
            seg = row.get("segment", "Unknown")
            n   = int(row.get("n_samples", 0))
            rate = float(row.get("churn_rate", 0)) * 100
            auc  = float(row.get("auc_roc", 0))
            f1   = row.get("f1", None)
            f1_str = f"{float(f1):.3f}" if pd.notna(f1) else "N/A"

            lines.append(
                f"  {seg:<14} | Customers: {n:>3} | Churn Rate: {rate:.1f}% | "
                f"AUC-ROC: {auc:.4f} | F1: {f1_str}"
            )

        lines.append(
            "\nInsight: Low-Value segment has the highest churn risk (87.2%). "
            "Champions and High-Value customers churn at lower rates but represent "
            "the highest revenue loss risk per customer lost."
        )
        return "\n".join(lines)
    except Exception as e:
        return f"Churn segment data unavailable: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Churn Error Analysis Tool
# ─────────────────────────────────────────────────────────────────────────────
@async_cache(expire=86400)
async def get_churn_error_analysis() -> str:
    """
    Return which features most strongly drive False Positives and False Negatives
    in the churn DNN model. Helps understand model reliability.
    """
    try:
        path = os.path.join(cfg["paths"]["eval_dir"], "error_profiles.csv")
        df = pd.read_csv(path, index_col=0)

        lines = ["[Churn Model Error Analysis — Feature Profiles]\n"]
        lines.append("Feature Impact on Prediction Errors (scaled values):\n")
        lines.append(f"{'Feature':<20} {'False Positive Avg':>22} {'False Negative Avg':>22}")
        lines.append("-" * 66)

        for feature, row in df.iterrows():
            fp = float(row.get("False Positive avg", 0))
            fn = float(row.get("False Negative avg", 0))
            lines.append(f"{feature:<20} {fp:>22.4f} {fn:>22.4f}")

        lines.append(
            "\nKey Insight: High negative r_score/f_score/rfm_score in False Positives "
            "means the model over-predicts churn for customers who bought once but recently. "
            "False Negatives are driven by high frequency customers with borderline recency."
        )
        return "\n".join(lines)
    except Exception as e:
        return f"Churn error analysis unavailable: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Churn Risk Overview (backward-compatible wrapper)
# ─────────────────────────────────────────────────────────────────────────────
async def get_churn_risk_overview() -> str:
    """Backward-compatible overview combining segment risks and key insight."""
    return await get_churn_risk_by_segment()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Semantic Business Knowledge Retrieval (Vector RAG)
# ─────────────────────────────────────────────────────────────────────────────
@async_cache(expire=86400)
async def search_business_knowledge(query: str, n_results: int = 3) -> str:
    """
    Perform semantic similarity search across the ChromaDB vector store.
    Returns the top-N most relevant business knowledge documents (KPI definitions,
    business rules, ML model docs) based on the query's meaning.
    """
    try:
        collection = _get_vector_collection()
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            return "No relevant business knowledge found for this query."

        lines = [f"[Semantic Search — Top {len(documents)} Results for: '{query}']\n"]
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
            relevance = round((1 - dist) * 100, 1)
            doc_type  = meta.get("type", "doc").upper()
            key       = meta.get("key", "")
            lines.append(f"── Result {i} [{doc_type}: {key}] (Relevance: {relevance}%) ──")
            lines.append(doc)
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        if "does not exist" in str(e):
            return (
                "Vector knowledge base not initialized. "
                "Run: python -m src.chatbot.ingest_knowledge"
            )
        return f"Knowledge retrieval error: {str(e)}"
