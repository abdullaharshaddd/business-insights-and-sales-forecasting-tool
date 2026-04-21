import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import yaml
import os
import joblib
import matplotlib.pyplot as plt
import json

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_prophet():
    print("Starting Prophet model training...")
    config = load_config()
    
    # Paths
    data_path = config['paths']['daily_revenue']
    model_path = config['paths']['prophet_model']
    eval_dir = config['paths']['forecasting_eval']
    figures_dir = os.path.join(config['paths']['reports_figures'], 'forecasting')
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path)
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Initialize Prophet model with config params
    p_config = config['forecasting']['prophet']
    model = Prophet(
        growth=p_config['growth'],
        yearly_seasonality=p_config['yearly_seasonality'],
        weekly_seasonality=p_config['weekly_seasonality'],
        daily_seasonality=p_config['daily_seasonality'],
        seasonality_mode=p_config['seasonality_mode'],
        changepoint_prior_scale=p_config['changepoint_prior_scale'],
        holidays_prior_scale=p_config['holidays_prior_scale'],
        interval_width=p_config['interval_width']
    )
    
    # Add UK holidays
    model.add_country_holidays(country_name='UK')
    
    # Train model
    print("Fitting model...")
    model.fit(df)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Cross-Validation
    print("Running Cross-Validation...")
    train_cfg = config['forecasting']['training']
    df_cv = cross_validation(
        model, 
        initial=train_cfg['cv_initial'], 
        period=train_cfg['cv_period'], 
        horizon=train_cfg['cv_horizon'],
        parallel="processes"
    )
    df_p = performance_metrics(df_cv)
    print("Cross-Validation Metrics (Mean):")
    print(df_p.mean())
    
    # Save metrics
    metrics_path = os.path.join(eval_dir, 'metrics.csv')
    df_p.to_csv(metrics_path, index=False)
    
    # Save summary metrics to JSON (convert Timedeltas to strings)
    summary_metrics = df_p.mean().to_dict()
    for k, v in summary_metrics.items():
        if isinstance(v, pd.Timedelta):
            summary_metrics[k] = str(v)
            
    with open(os.path.join(eval_dir, 'summary_metrics.json'), 'w') as f:
        json.dump(summary_metrics, f, indent=4)
        
    # Forecast
    print("Generating forecast...")
    future = model.make_future_dataframe(periods=train_cfg['forecast_horizon'])
    forecast = model.predict(future)
    
    # Save forecast
    forecast.to_csv(os.path.join(eval_dir, 'forecast_results.csv'), index=False)
    
    # Plotting
    print("Generating plots...")
    
    # Forecast plot
    fig1 = model.plot(forecast)
    plt.title('Sales Forecast (Prophet)')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    fig1.savefig(os.path.join(figures_dir, 'forecast_plot.png'))
    
    # Components plot
    fig2 = model.plot_components(forecast)
    fig2.savefig(os.path.join(figures_dir, 'forecast_components.png'))
    
    print("Training and evaluation complete.")

if __name__ == "__main__":
    train_prophet()
