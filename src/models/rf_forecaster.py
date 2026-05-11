import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

def train_rf_forecaster():
    print("Starting Random Forest Model Training...")
    
    # Paths
    data_path = "data/processed/processed_for_forecasting.csv"
    model_dir = "models/forecasting"
    eval_dir = "evaluation/forecasting"
    figures_dir = "reports/figures/forecasting"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run feature engineering first.")
        return
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    
    # Target and Features
    target_col = 'daily_sales'
    features = [c for c in df.columns if c != target_col]
    
    X = df[features]
    y = df[target_col]
    
    # Train/Test Split (Sequential)
    # Let's keep the last 30 days for final testing to match the prediction horizon
    test_size = 30
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Model and Hyperparameter Tuning
    print("Performing Hyperparameter Tuning with TimeSeriesSplit...")
    rf = RandomForestRegressor(random_state=42)
    
    # Grid handles Overfitting by regularizing max_depth and min_samples_split, 
    # and handles Underfitting by increasing n_estimators
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    rf_random = RandomizedSearchCV(
        estimator=rf, 
        param_distributions=param_grid, 
        n_iter=20, 
        cv=tscv, 
        scoring='neg_mean_squared_error',
        verbose=2, 
        random_state=42, 
        n_jobs=-1
    )
    
    rf_random.fit(X_train, y_train)
    
    best_model = rf_random.best_estimator_
    print(f"Best Parameters: {rf_random.best_params_}")
    
    # Evaluation
    print("Evaluating model on test set...")
    y_pred = best_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R²: {r2:.2f}")
    
    # Save Metrics
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'horizon': f'{test_size} days',
        'best_params': rf_random.best_params_
    }
    
    with open(os.path.join(eval_dir, 'rf_summary_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save Model
    model_path = os.path.join(model_dir, 'rf_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names for frontend
    with open(os.path.join(model_dir, 'rf_features.json'), 'w') as f:
        json.dump({'features': features}, f)
        
    # Generate Forecast Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_train.index[-60:], y_train.iloc[-60:], label="Train Sales")
    plt.plot(y_test.index, y_test, label="True Test Sales", color="green")
    plt.plot(y_test.index, y_pred, label="Random Forest Forecast", color="red", linestyle="--")
    plt.title('30-Day Sales Forecast vs Actual (Random Forest)')
    plt.xlabel('Date')
    plt.ylabel('Daily Sales')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, 'rf_forecast_plot.png'))
    print("Forecast plot saved.")
    
    print("Random Forest Training Complete.")

if __name__ == "__main__":
    train_rf_forecaster()
