"""
Forecasting Router — Prophet Forecast & Metrics
"""
from fastapi import APIRouter, HTTPException, Query
import traceback
import os
import json
import joblib
import pandas as pd

router = APIRouter(prefix="/api/forecast", tags=["Forecasting"])


@router.get("")
def get_forecast(days: int = Query(default=30, ge=7, le=90)):
    """
    Return forecast data points for charting.
    Tries live Prophet model first, falls back to pre-computed CSV.
    """
    try:
        import yaml
        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)

        prophet_path = cfg["paths"].get("prophet_model", "models/forecasting/prophet_model.pkl")
        forecast_csv = os.path.join(cfg["paths"].get("forecasting_eval", "evaluation/forecasting"), "forecast_results.csv")

        # Try live Prophet
        if os.path.exists(prophet_path):
            try:
                model = joblib.load(prophet_path)
                future = model.make_future_dataframe(periods=days)
                forecast = model.predict(future)
                future_only = forecast.tail(days)[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]]
                points = []
                for _, row in future_only.iterrows():
                    points.append({
                        "date": row["ds"].strftime("%Y-%m-%d"),
                        "yhat": round(float(row["yhat"]), 2),
                        "yhat_lower": round(float(row["yhat_lower"]), 2),
                        "yhat_upper": round(float(row["yhat_upper"]), 2),
                        "trend": round(float(row["trend"]), 2),
                    })
                summary = _build_summary(points)
                return {"status": "ok", "source": "live_prophet", "days": days, "points": points, "summary": summary}
            except Exception as e:
                print(f"[Forecast] Live Prophet failed: {e}, falling back to CSV")

        # Fallback: pre-computed CSV
        if os.path.exists(forecast_csv):
            df = pd.read_csv(forecast_csv)
            df["ds"] = pd.to_datetime(df["ds"])
            future_only = df.tail(days)
            points = []
            for _, row in future_only.iterrows():
                points.append({
                    "date": row["ds"].strftime("%Y-%m-%d"),
                    "yhat": round(float(row.get("yhat", 0)), 2),
                    "yhat_lower": round(float(row.get("yhat_lower", 0)), 2),
                    "yhat_upper": round(float(row.get("yhat_upper", 0)), 2),
                    "trend": round(float(row.get("trend", row.get("yhat", 0))), 2),
                })
            summary = _build_summary(points)
            return {"status": "ok", "source": "cached_csv", "days": days, "points": points, "summary": summary}

        raise HTTPException(status_code=404, detail="No forecast data available. Train the Prophet model first.")

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
def get_forecast_metrics():
    """Return Prophet cross-validation performance metrics."""
    try:
        import yaml
        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)

        eval_dir = cfg["paths"].get("forecasting_eval", "evaluation/forecasting")
        metrics_path = os.path.join(eval_dir, "summary_metrics.json")

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            return {"status": "ok", "metrics": metrics}

        # Fallback: try metrics.csv
        metrics_csv = os.path.join(eval_dir, "metrics.csv")
        if os.path.exists(metrics_csv):
            df = pd.read_csv(metrics_csv)
            return {"status": "ok", "metrics": df.to_dict(orient="records")}

        # Hardcoded fallback (from memory.md evaluation results)
        return {
            "status": "ok",
            "source": "hardcoded_fallback",
            "metrics": {
                "rmse": 14819.0,
                "mae": 10589.0,
                "mape": 0.36,
                "coverage": 0.867,
                "horizon": "30 days",
            }
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _build_summary(points: list) -> dict:
    if not points:
        return {}
    yhats = [p["yhat"] for p in points]
    total = sum(yhats)
    avg = total / len(yhats)
    trend = "INCREASING" if yhats[-1] > yhats[0] else "DECREASING"
    peak_idx = yhats.index(max(yhats))
    return {
        "total_revenue": round(total, 2),
        "avg_daily_revenue": round(avg, 2),
        "trend": trend,
        "peak_day": points[peak_idx]["date"],
        "peak_value": round(max(yhats), 2),
        "min_value": round(min(yhats), 2),
    }
