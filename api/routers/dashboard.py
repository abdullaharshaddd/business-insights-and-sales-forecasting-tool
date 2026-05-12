"""
Dashboard Router — All KPI Endpoints
All data served from BISFT PostgreSQL database — no CSV files.
"""
from fastapi import APIRouter, HTTPException
import traceback

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


def _get_kpi_engine():
    """Lazy-load KPIEngine to avoid import errors at startup."""
    from src.analytics.kpi_engine import KPIEngine
    return KPIEngine()


def _to_native(val):
    """Convert numpy types to native Python."""
    if val is None:
        return None
    if hasattr(val, "item"):
        return val.item()
    return val


@router.get("/kpis")
def get_all_kpis():
    """Return all core KPI values — scalars + tabular data."""
    try:
        engine = _get_kpi_engine()

        # Scalar KPIs
        scalar_kpis = [
            "revenue", "revenue_delivered", "total_orders", "delivered_orders",
            "pending_orders", "canceled_orders", "aov", "aov_delivered",
            "on_time_delivery_rate", "avg_review_score", "cancellation_rate",
            "freight_ratio", "avg_payment_installments", "total_customers",
            "total_products", "total_sellers", "total_revenue_retail",
            "retail_orders", "retail_customers"
        ]

        result = {}
        for kpi_id in scalar_kpis:
            try:
                df = engine.calculate_kpi(kpi_id)
                val = df.iloc[0, 0] if not df.empty else None
                val = _to_native(val)
                result[kpi_id] = {
                    "value": round(float(val), 2) if val is not None else None,
                    "label": engine.definitions[kpi_id].get("label", kpi_id),
                    "unit": engine.definitions[kpi_id].get("unit", ""),
                    "category": engine.definitions[kpi_id].get("category", ""),
                }
            except Exception as e:
                result[kpi_id] = {"value": None, "label": kpi_id, "unit": "", "error": str(e)}

        # Tabular KPIs
        row_kpis = engine.get_row_kpis()
        for kpi_id, data in row_kpis.items():
            result[kpi_id] = {
                "data": data,
                "label": engine.definitions[kpi_id].get("label", kpi_id),
                "unit": engine.definitions[kpi_id].get("unit", ""),
                "category": engine.definitions[kpi_id].get("category", ""),
            }

        return {"status": "ok", "data": result}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"KPI Engine error: {str(e)}")


@router.get("/kpis/{kpi_id}")
def get_single_kpi(kpi_id: str):
    """Return a single KPI by ID."""
    try:
        engine = _get_kpi_engine()
        df = engine.calculate_kpi(kpi_id)
        returns_rows = engine.definitions[kpi_id].get("returns") == "rows"

        if returns_rows:
            return {"status": "ok", "data": df.to_dict(orient="records")}
        else:
            val = df.iloc[0, 0] if not df.empty else None
            val = _to_native(val)
            return {
                "status": "ok",
                "data": {
                    "value": round(float(val), 2) if val is not None else None,
                    "label": engine.definitions[kpi_id].get("label", kpi_id),
                    "unit": engine.definitions[kpi_id].get("unit", ""),
                }
            }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"KPI '{kpi_id}' not found: {str(e)}")


@router.get("/kpis/category/{category}")
def get_kpis_by_category(category: str):
    """Return all KPIs for a specific category (Revenue, Orders, Logistics, etc.)."""
    try:
        engine = _get_kpi_engine()
        result = {}

        for kpi_id, kpi_def in engine.definitions.items():
            if kpi_def.get("category", "").lower() == category.lower():
                try:
                    df = engine.calculate_kpi(kpi_id)
                    returns_rows = kpi_def.get("returns") == "rows"
                    if returns_rows:
                        result[kpi_id] = {"data": df.to_dict(orient="records")}
                    else:
                        val = df.iloc[0, 0] if not df.empty else None
                        val = _to_native(val)
                        result[kpi_id] = {
                            "value": round(float(val), 2) if val is not None else None,
                        }
                except Exception:
                    result[kpi_id] = {"error": "Failed to calculate"}

        return {"status": "ok", "category": category, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def system_status():
    """Health check — verify DB and model files are accessible."""
    import os, yaml

    try:
        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)

        from sqlalchemy import create_engine, text
        engine = create_engine(cfg["database"]["url"])
        db_ok = False
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                db_ok = True
        except:
            db_ok = False

        checks = {
            "database": db_ok,
            "prophet_model": os.path.exists(cfg["paths"].get("prophet_model", "")),
            "churn_model": os.path.exists(os.path.join(cfg["paths"]["model_dir"], "churn_dnn_best.h5")),
            "vector_db": os.path.exists("data/vector_db"),
            "groq_key": bool(os.getenv("GROQ_API_KEY") or _read_env_key()),
        }
        all_ok = all(checks.values())
        return {"status": "healthy" if all_ok else "degraded", "checks": checks}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def _read_env_key():
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("GROQ_API_KEY"):
                    return line.split("=")[1].strip()
    except Exception:
        pass
    return None