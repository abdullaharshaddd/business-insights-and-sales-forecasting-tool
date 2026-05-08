from sqlalchemy import create_engine, text
import yaml

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

engine = create_engine(cfg["database"]["url"])

with engine.connect() as conn:
    for table in ["orders", "order_items", "order_reviews", "customers"]:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
        print(f"Table {table}: {count} rows")
