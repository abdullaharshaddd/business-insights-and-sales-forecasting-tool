import pandas as pd
import sqlite3
import os
import glob

def preprocess_olist_to_sqlite():
    # Define paths
    raw_data_path = 'data/raw/olist/'
    output_db_path = 'data/processed/olist/olist.db'
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_db_path), exist_ok=True)
    
    # Connect (create) the SQLite database
    conn = sqlite3.connect(output_db_path)
    
    # Find all CSV files in the raw folder
    csv_files = glob.glob(os.path.join(raw_data_path, "*.csv"))
    print(f"Found {len(csv_files)} Olist CSV files. Starting import...")
    
    for file in csv_files:
        # Simplify the table name (e.g., 'olist_customers_dataset' -> 'customers')
        table_name = os.path.basename(file).replace(".csv", "").replace("olist_", "").replace("_dataset", "")
        
        # Read CSV and write to SQL
        df = pd.read_csv(file)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"  - Imported: {table_name} ({len(df)} rows)")
        
    conn.close()
    print(f"✅ Success! Database created at: {output_db_path}")

if __name__ == "__main__":
    preprocess_olist_to_sqlite()
