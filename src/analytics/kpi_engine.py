import json
import pandas as pd
import yaml
import os
from sqlalchemy import create_engine, text


class KPIEngine:
    """
    Central engine for calculating deterministic business KPIs from PostgreSQL.
    All queries run against the BISFT database — no CSV files.
    """

    def __init__(self, config_path="config/config.yaml", kpi_path="config/kpi_definitions.json"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.db_url = cfg["database"]["url"]
        self.engine = create_engine(self.db_url)

        with open(kpi_path) as f:
            self.definitions = json.load(f)

    def _get_connection(self):
        return self.engine.connect()

    def calculate_kpi(self, kpi_id, filters: dict = None):
        """Calculate a specific KPI by its ID from the registry."""
        if kpi_id not in self.definitions:
            raise ValueError(f"KPI '{kpi_id}' not found in registry.")

        kpi = self.definitions[kpi_id]
        formula = kpi["sql_formula"]
        returns_rows = kpi.get("returns") == "rows"

        where_clause = ""
        query_params = {}
        if filters:
            conditions = []
            for i, (k, v) in enumerate(filters.items()):
                param_name = f"p{i}"
                if isinstance(v, list) and len(v) == 2:
                    conditions.append(f"{k} >= :{param_name}_start AND {k} <= :{param_name}_end")
                    query_params[f"{param_name}_start"] = v[0]
                    query_params[f"{param_name}_end"] = v[1]
                else:
                    conditions.append(f"{k} = :{param_name}")
                    query_params[param_name] = v
            if conditions:
                where_clause = " WHERE " + " AND ".join(conditions)

        if where_clause:
            if formula.strip().upper().startswith("SELECT"):
                query = f"{formula}{where_clause}"
            else:
                query = f"SELECT * FROM ({formula}) sub{where_clause}"
        else:
            query = formula

        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=query_params)
            return df

    def get_all_kpis_summary(self):
        """Returns a dictionary of all core KPI values."""
        summary = {}
        core_metrics = [
            "revenue", "revenue_delivered", "total_orders", "delivered_orders",
            "canceled_orders", "aov", "on_time_delivery_rate",
            "avg_review_score", "cancellation_rate", "freight_ratio",
            "avg_payment_installments", "total_customers", "total_products",
            "total_sellers", "total_revenue_retail", "retail_orders"
        ]

        for mid in core_metrics:
            try:
                res = self.calculate_kpi(mid)
                val = res.iloc[0, 0] if not res.empty else None
                if hasattr(val, "item"):
                    val = val.item()
                summary[mid] = round(float(val), 2) if val is not None else None
            except Exception:
                summary[mid] = None
        return summary

    def get_row_kpis(self):
        """Returns all tabular KPIs (rows, not scalars)."""
        row_kpis = {}
        row_kpi_ids = [
            "top_categories_by_revenue", "customer_geographic_concentration",
            "top_selling_products", "payment_type_distribution",
            "monthly_revenue_trend", "review_score_distribution",
            "order_status_breakdown"
        ]

        for kpi_id in row_kpi_ids:
            try:
                df = self.calculate_kpi(kpi_id)
                row_kpis[kpi_id] = df.to_dict(orient="records")
            except Exception:
                row_kpis[kpi_id] = []

        return row_kpis


if __name__ == "__main__":
    engine = KPIEngine()
    print("=== KPI Summary ===")
    summary = engine.get_all_kpis_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n=== Row KPIs ===")
    rows = engine.get_row_kpis()
    for kpi_id, data in rows.items():
        print(f"  {kpi_id}: {len(data)} rows")