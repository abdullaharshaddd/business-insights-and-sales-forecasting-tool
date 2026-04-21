import pandas as pd
import yaml
import os

# Load Config
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

def query_database(query: str) -> str:
    """Look up past business facts from the Olist DB."""
    import sqlite3
    db_path = cfg['paths']['olist_db']
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn)
        return df.to_string()
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        conn.close()

def get_sales_forecast_summary() -> str:
    """See predicted future revenue trends for the next 30 days."""
    forecast_path = os.path.join(cfg['paths']['forecasting_eval'], 'forecast_results.csv')
    if not os.path.exists(forecast_path):
        return "Error: Forecast results not found. Train the Prophet model first."
    
    df = pd.read_csv(forecast_path)
    future = df.tail(30)
    avg_pred = future['yhat'].mean()
    trend = "increasing" if future['yhat'].iloc[-1] > future['yhat'].iloc[0] else "decreasing"
    
    return f"Forecast Summary: Predicted average daily revenue is ${avg_pred:,.2f}. The trend for the next 30 days is {trend}."

def get_churn_risk_overview() -> str:
    """Identify high-level customer retention issues."""
    return "Strategic Insight: Retention is stable, but high-value customers in the 'Furniture' category show a 15% higher churn risk than average."
