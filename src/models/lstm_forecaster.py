"""
lstm_forecaster.py
==================
Business Insights & Sales Forecasting Tool
Phase 4 — Time Series Forecasting (Multivariate LSTM)

This script implements a multivariate LSTM model to forecast daily sales
for the next 30 days based on historical performance, freight costs, 
and customer satisfaction.

Architecture
------------
Input (60, N) ──► LSTM(100, return_sequences=True) ──► Dropout(0.2)
               ──► LSTM(50) ──► Dropout(0.2)
               ──► Dense(30, Linear) [Output: 30-day forecast]
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "olist_merged_cleaned.csv")
SAVE_DIR    = os.path.join(BASE_DIR, "models", "lstm")

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Feature Engineering & Aggregation ──────────────────────────────────────

def prepare_time_series_data(df: pd.DataFrame):
    """
    Aggregate transaction data into daily multivariate time series.
    """
    print("[LSTM] Aggregating data into daily buckets...")
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    # Aggregate by day
    # We use 'price' as sales target and others as exogenous variables
    daily_df = df.groupby(df['order_purchase_timestamp'].dt.date).agg({
        'price':         'sum',           # Total Sales
        'freight_value': 'mean',          # Avg Freight
        'review_score':  'mean',          # Avg Satisfaction
        'order_id':      'nunique'        # Order Volume
    }).rename(columns={'price': 'total_sales'})
    
    daily_df.index = pd.to_datetime(daily_df.index)
    daily_df = daily_df.sort_index()

    # Fill potential missing dates with 0/interpolation
    all_days = pd.date_range(start=daily_df.index.min(), end=daily_df.index.max(), freq='D')
    daily_df = daily_df.reindex(all_days)
    daily_df['total_sales'] = daily_df['total_sales'].fillna(0)
    daily_df['order_id']    = daily_df['order_id'].fillna(0)
    daily_df['freight_value'] = daily_df['freight_value'].interpolate(method='linear')
    daily_df['review_score']  = daily_df['review_score'].interpolate(method='linear')

    # Add temporal features
    daily_df['day_of_week'] = daily_df.index.dayofweek
    daily_df['month']       = daily_df.index.month
    
    return daily_df

def create_windows(data, lookback=60, horizon=30):
    """
    Create sliding windows for LSTM.
    data: scaled numpy array
    lookback: number of past days to look at
    horizon: number of future days to predict
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        # We predict ONLY sales column (index 0) for the horizon
        y.append(data[i + lookback : i + lookback + horizon, 0]) 
    return np.array(X), np.array(y)

# ── Model Architecture ─────────────────────────────────────────────────────

def build_lstm_model(input_shape, horizon=30):
    """
    Build multivariate LSTM model.
    """
    inputs = Input(shape=input_shape)
    
    # Layer 1
    x = LSTM(100, return_sequences=True, name="lstm_1")(inputs)
    x = Dropout(0.2, name="dropout_1")(x)
    
    # Layer 2
    x = LSTM(50, return_sequences=False, name="lstm_2")(x)
    x = Dropout(0.2, name="dropout_2")(x)
    
    # Output (30 units for 30-day forecast)
    outputs = Dense(horizon, activation='linear', name="output_sales")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="SalesLSTM")
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model

# ── Execution ──────────────────────────────────────────────────────────────

def main():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Processed data not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    daily_df = prepare_time_series_data(df)
    
    # 2. Scaling
    feature_cols = ['total_sales', 'freight_value', 'review_score', 'order_id', 'day_of_week', 'month']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(daily_df[feature_cols])
    
    # 3. Windowing
    lookback = 60
    horizon  = 30
    X, y = create_windows(scaled_data, lookback, horizon)
    print(f"  Input shape: {X.shape}, Target shape: {y.shape}")

    # 4. Train/Test Split (Sequential)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 5. Build Model
    model = build_lstm_model(input_shape=(lookback, len(feature_cols)), horizon=horizon)
    model.summary()

    # 6. Training
    print("\n[LSTM] Starting training...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # 7. Evaluation
    print("\n[LSTM] Evaluating model...")
    predictions = model.predict(X_test)
    
    # Inverse Transform (only for 'total_sales' target)
    # Since we scaled multiple columns, we need to handle inverse-transforming just the target
    # An easier way is to create a dummy scaler or just use the target scaler.
    # For now, we manually unscale based on first column distribution.
    target_min = scaler.data_min_[0]
    target_max = scaler.data_max_[0]
    
    y_test_unscaled = y_test * (target_max - target_min) + target_min
    preds_unscaled  = predictions * (target_max - target_min) + target_min

    # Calculate Metrics for the first day of forecast as a sample
    rmse = np.sqrt(mean_squared_error(y_test_unscaled[:, 0], preds_unscaled[:, 0]))
    mae  = mean_absolute_error(y_test_unscaled[:, 0], preds_unscaled[:, 0])
    
    print(f"  Test RMSE: {rmse:.2f}")
    print(f"  Test MAE:  {mae:.2f}")

    # 8. Save
    model.save(os.path.join(SAVE_DIR, "sales_lstm_model.h5"))
    print(f"\n✅ Model saved to {SAVE_DIR}")

    # 9. Plot Sample (last prediction sequence vs actual)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_unscaled[-1, :], label="True Future Sales", marker='o')
    plt.plot(preds_unscaled[-1, :], label="LSTM Forecast", marker='x')
    plt.title("30-Day Sales Forecast (Final Sample)")
    plt.xlabel("Days Ahead")
    plt.ylabel("Total Sales")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "forecast_plot.png"))
    print(f"  Forecast plot saved to {SAVE_DIR}/forecast_plot.png")

if __name__ == "__main__":
    main()
