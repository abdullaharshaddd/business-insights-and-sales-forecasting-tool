import pandas as pd
import yaml
import os

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_for_forecasting():
    print("Starting preprocessing for forecasting...")
    config = load_config()
    
    # Load cleaned online retail data
    data_path = config['paths']['processed_online_retail']
    if not os.path.exists(data_path):
        # Fallback to direct path if config doesn't match current structure exactly
        data_path = 'data/processed/online_retail_cleaned.csv'
        
    df = pd.read_csv(data_path)
    
    # Ensure invoicedate is datetime
    df['invoicedate'] = pd.to_datetime(df['invoicedate'])
    
    # Aggregate by day
    # Prophet requires 'ds' and 'y' columns
    daily_df = df.groupby(df['invoicedate'].dt.date)['totalprice'].sum().reset_index()
    daily_df.columns = ['ds', 'y']
    
    # Ensure ds is datetime
    daily_df['ds'] = pd.to_datetime(daily_df['ds'])
    
    # Save processed data
    output_path = config['paths']['daily_revenue']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    daily_df.to_csv(output_path, index=False)
    
    print(f"Preprocessed data saved to {output_path}")
    print(f"Total rows: {len(daily_df)}")
    print(daily_df.head())

if __name__ == "__main__":
    preprocess_for_forecasting()
