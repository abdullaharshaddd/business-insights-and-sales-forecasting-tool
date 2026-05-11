"""
Forecasting Router — Random Forest Forecast & Metrics
"""
from fastapi import APIRouter, HTTPException, Query
import traceback
import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta

router = APIRouter(prefix="/api/forecast", tags=["Forecasting"])

@router.get("")
def get_forecast(days: int = Query(default=30, ge=7, le=90)):
    """
    Return forecast data points for charting using Random Forest.
    """
    try:
        model_path = "models/forecasting/rf_model.pkl"
        data_path = "data/processed/processed_for_forecasting.csv"
        
        if not os.path.exists(model_path) or not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="Model or data not found. Train the RF model first.")
            
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # We need historical sales to compute lags for the future
        historical_sales = df['daily_sales'].tolist()
        last_date = df['date'].iloc[-1]
        
        # Get medians for exogenous variables
        exog_vars = {
            'avg_freight': df['avg_freight'].median(),
            'total_freight': df['total_freight'].median(),
            'unique_customers': df['unique_customers'].median(),
            'unique_orders': df['unique_orders'].median(),
            'unique_products': df['unique_products'].median(),
            'unique_sellers': df['unique_sellers'].median(),
        }
        
        points = []
        current_sales_history = historical_sales.copy()
        
        for i in range(1, days + 1):
            future_date = last_date + timedelta(days=i)
            
            # Compute temporal features
            day_of_week = future_date.dayofweek
            is_weekend = 1 if day_of_week in [5, 6] else 0
            month = future_date.month
            quarter = future_date.quarter
            is_month_end = 1 if future_date.is_month_end else 0
            
            # Compute lags and rolling stats from current_sales_history
            sales_lag_1 = current_sales_history[-1]
            sales_lag_7 = current_sales_history[-7]
            sales_lag_30 = current_sales_history[-30]
            
            last_7 = current_sales_history[-7:]
            last_30 = current_sales_history[-30:]
            
            sales_rolling_mean_7 = np.mean(last_7)
            sales_rolling_std_7 = np.std(last_7)
            sales_rolling_mean_30 = np.mean(last_30)
            
            # Construct feature array
            # Must match exact order of features used in training.
            features_path = "models/forecasting/rf_features.json"
            if os.path.exists(features_path):
                with open(features_path) as f:
                    feature_names = json.load(f)['features']
            else:
                # Fallback to hardcoded order from rf_feature_engineering
                feature_names = ['avg_freight', 'total_freight', 'unique_customers', 'unique_orders', 
                                 'unique_products', 'unique_sellers', 'day_of_week', 'is_weekend', 
                                 'month', 'quarter', 'is_month_end', 'sales_lag_1', 'sales_lag_7', 
                                 'sales_lag_30', 'sales_rolling_mean_7', 'sales_rolling_std_7', 
                                 'sales_rolling_mean_30']
            
            row_dict = {
                'avg_freight': exog_vars['avg_freight'],
                'total_freight': exog_vars['total_freight'],
                'unique_customers': exog_vars['unique_customers'],
                'unique_orders': exog_vars['unique_orders'],
                'unique_products': exog_vars['unique_products'],
                'unique_sellers': exog_vars['unique_sellers'],
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'month': month,
                'quarter': quarter,
                'is_month_end': is_month_end,
                'sales_lag_1': sales_lag_1,
                'sales_lag_7': sales_lag_7,
                'sales_lag_30': sales_lag_30,
                'sales_rolling_mean_7': sales_rolling_mean_7,
                'sales_rolling_std_7': sales_rolling_std_7,
                'sales_rolling_mean_30': sales_rolling_mean_30
            }
            
            X_future = pd.DataFrame([row_dict], columns=feature_names)
            
            # Predict
            pred_sales = model.predict(X_future)[0]
            
            # RF doesn't output confidence intervals natively like Prophet,
            # but we can simulate them or just use the prediction.
            # We'll mock bounds as +/- 15% for visual purposes
            lower = pred_sales * 0.85
            upper = pred_sales * 1.15
            
            points.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "yhat": round(float(pred_sales), 2),
                "yhat_lower": round(float(lower), 2),
                "yhat_upper": round(float(upper), 2),
                "trend": round(float(sales_rolling_mean_7), 2)  # Proxy for trend
            })
            
            # Append prediction to history for next day's lags
            current_sales_history.append(pred_sales)
            
        summary = _build_summary(points)
        return {"status": "ok", "source": "live_rf", "days": days, "points": points, "summary": summary}
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
def get_forecast_metrics():
    """Return Random Forest evaluation metrics."""
    try:
        metrics_path = "evaluation/forecasting/rf_summary_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            return {"status": "ok", "metrics": metrics}
            
        return {
            "status": "ok",
            "source": "fallback",
            "metrics": {
                "rmse": 0,
                "mae": 0,
                "r2": 0,
                "horizon": "30 days"
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
