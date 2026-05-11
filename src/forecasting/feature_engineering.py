import pandas as pd
import yaml
import os

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def engineer_features():
    print("Starting Forecasting Feature Engineering...")
    config = load_config()
    
    raw_path = config['paths']['processed_online_retail']
    output_path = "data/processed/forecasting_features.csv"
    
    # Load data
    df = pd.read_csv(raw_path)
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    df['date'] = df['invoicedate'].dt.date
    
    print(f"Processing {len(df)} transactions...")
    
    # Aggregate daily
    daily_stats = df.groupby('date').agg({
        'totalprice': 'sum',
        'quantity': 'sum',
        'unitprice': 'mean',
        'stockcode': 'nunique'
    }).reset_index()
    
    # Rename for Prophet
    daily_stats.columns = ['ds', 'y', 'total_qty', 'avg_unit_price', 'unique_items']
    
    # Sort by date
    daily_stats = daily_stats.sort_values('ds')
    
    # Handle outliers (e.g. negative quantities/returns if any)
    daily_stats = daily_stats[daily_stats['y'] > 0]
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    daily_stats.to_csv(output_path, index=False)
    
    print(f"Feature engineering complete. Saved to {output_path}")
    print(daily_stats.head())

if __name__ == "__main__":
    engineer_features()
