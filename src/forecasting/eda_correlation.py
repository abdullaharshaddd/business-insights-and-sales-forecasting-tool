import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    print("Starting EDA Correlation Analysis...")
    
    # Paths
    olist_data_path = "data/processed/olist_merged_cleaned.csv"
    output_dir = "reports/figures/forecasting/eda"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(olist_data_path):
        print(f"File not found: {olist_data_path}")
        return
        
    print(f"Loading {olist_data_path}...")
    df = pd.read_csv(olist_data_path)
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['date'] = df['order_purchase_timestamp'].dt.date
    
    print("Aggregating daily metrics for correlation analysis...")
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
    
    # Temporal and Lag features
    print("Engineering temporal and lag features...")
    daily_df['day_of_week'] = daily_df.index.dayofweek
    daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)
    daily_df['month'] = daily_df.index.month
    
    daily_df['sales_lag_1'] = daily_df['daily_sales'].shift(1)
    daily_df['sales_lag_7'] = daily_df['daily_sales'].shift(7)
    daily_df['sales_rolling_mean_7'] = daily_df['daily_sales'].rolling(window=7).mean()
    daily_df['sales_rolling_std_7'] = daily_df['daily_sales'].rolling(window=7).std()
    
    # Drop NAs created by shift and rolling
    daily_df = daily_df.dropna()
    
    # Calculate Correlations
    print("Calculating correlations...")
    corr_matrix = daily_df.corr(method='spearman')
    
    # Sort correlations with daily_sales
    sales_corr = corr_matrix['daily_sales'].sort_values(ascending=False)
    print("\n--- Feature Correlation with Daily Sales ---")
    print(sales_corr)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix for Forecasting Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    print(f"\nCorrelation matrix saved to {output_dir}/correlation_matrix.png")

if __name__ == "__main__":
    main()
