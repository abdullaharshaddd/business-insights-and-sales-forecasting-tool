"""
Dashboard Router — KPI Endpoints
Calls the deterministic KPIEngine for all core business metrics.
"""
from fastapi import APIRouter, HTTPException
import traceback

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


def _get_kpi_engine():
    """Lazy-load KPIEngine to avoid import errors at startup."""
    from src.analytics.kpi_engine import KPIEngine
    return KPIEngine()


@router.get("/kpis")
def get_all_kpis():
    """Return all core KPI values for the dashboard summary cards."""
    try:
        engine = _get_kpi_engine()

        # Core scalar KPIs
        scalar_kpis = [
            "revenue", "aov", "on_time_delivery_rate",
            "avg_review_score", "cancellation_rate", "freight_ratio",
            "avg_payment_installments"
        ]

        result = {}
        for kpi_id in scalar_kpis:
            try:
                df = engine.calculate_kpi(kpi_id)
                val = df.iloc[0, 0]
                # Convert numpy types to native Python
                if hasattr(val, "item"):
                    val = val.item()
                result[kpi_id] = {
                    "value": round(float(val), 2) if val is not None else None,
                    "label": engine.definitions.get(kpi_id, {}).get("label", kpi_id),
                    "unit": engine.definitions.get(kpi_id, {}).get("unit", ""),
                }
            except Exception as e:
                result[kpi_id] = {"value": None, "label": kpi_id, "unit": "", "error": str(e)}

        # Top categories (tabular)
        try:
            df_cat = engine.calculate_kpi("top_categories_by_revenue")
            result["top_categories"] = df_cat.head(5).to_dict(orient="records")
        except Exception:
            result["top_categories"] = []

        # Geographic distribution (tabular)
        try:
            df_geo = engine.calculate_kpi("customer_geographic_concentration")
            result["geo_distribution"] = df_geo.head(10).to_dict(orient="records")
        except Exception:
            result["geo_distribution"] = []

        return {"status": "ok", "data": result}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"KPI Engine error: {str(e)}")


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
