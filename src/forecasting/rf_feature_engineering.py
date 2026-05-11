import os
import pandas as pd
import numpy as np

def create_features():
    print("Starting Random Forest Feature Engineering...")
    
    input_path = "data/processed/olist_merged_cleaned.csv"
    output_path = "data/processed/processed_for_forecasting.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
        
    print("Loading data...")
    df = pd.read_csv(input_path)
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['date'] = df['order_purchase_timestamp'].dt.date
    
    print("Aggregating daily metrics...")
    daily_df = df.groupby('date').agg(
        daily_sales=('price', 'sum'),
        avg_freight=('freight_value', 'mean'),
        total_freight=('freight_value', 'sum'),
        unique_customers=('customer_unique_id', 'nunique'),
        unique_orders=('order_id', 'nunique'),
        unique_products=('product_id', 'nunique'),
        unique_sellers=('seller_id', 'nunique')
    ).reset_index()
    
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values('date').set_index('date')
    
    # Generate full date range to ensure no missing dates
    all_days = pd.date_range(start=daily_df.index.min(), end=daily_df.index.max(), freq='D')
    daily_df = daily_df.reindex(all_days)
    daily_df['daily_sales'] = daily_df['daily_sales'].fillna(0)
    
    # Interpolate others or fill with 0
    daily_df['avg_freight'] = daily_df['avg_freight'].interpolate(method='linear').fillna(0)
    daily_df['total_freight'] = daily_df['total_freight'].fillna(0)
    for col in ['unique_customers', 'unique_orders', 'unique_products', 'unique_sellers']:
        daily_df[col] = daily_df[col].fillna(0)
        
    print("Creating temporal features...")
    daily_df['day_of_week'] = daily_df.index.dayofweek
    daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)
    daily_df['month'] = daily_df.index.month
    daily_df['quarter'] = daily_df.index.quarter
    daily_df['is_month_end'] = daily_df.index.is_month_end.astype(int)
    
    print("Creating lag features...")
    # These must NOT leak the future/current into the past.
    daily_df['sales_lag_1'] = daily_df['daily_sales'].shift(1)
    daily_df['sales_lag_7'] = daily_df['daily_sales'].shift(7)
    daily_df['sales_lag_30'] = daily_df['daily_sales'].shift(30)
    
    print("Creating rolling statistics...")
    # Ensure we use shift(1) before rolling to avoid target leakage
    daily_df['sales_rolling_mean_7'] = daily_df['daily_sales'].shift(1).rolling(window=7).mean()
    daily_df['sales_rolling_std_7'] = daily_df['daily_sales'].shift(1).rolling(window=7).std().fillna(0)
    daily_df['sales_rolling_mean_30'] = daily_df['daily_sales'].shift(1).rolling(window=30).mean()
    
    # Handle missing values created by shifting/rolling
    print("Handling missing values...")
    initial_len = len(daily_df)
    daily_df = daily_df.dropna()
    print(f"Dropped {initial_len - len(daily_df)} rows due to NA from lags/rolling.")
    
    # Reset index to have 'date' as a column
    daily_df = daily_df.reset_index().rename(columns={'index': 'date'})
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    daily_df.to_csv(output_path, index=False)
    print(f"Saved processed dataset to {output_path} with shape {daily_df.shape}")

if __name__ == "__main__":
    create_features()
