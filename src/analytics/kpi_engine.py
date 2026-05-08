import json
import pandas as pd
import yaml
import os
from sqlalchemy import create_engine

class KPIEngine:
    """
    Central engine for calculating deterministic business KPIs.
    Ensures that Dashboards and Chatbots use the exact same logic.
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
        
        where_clause = ""
        query_params = {}
        if filters:
            conditions = []
            for i, (k, v) in enumerate(filters.items()):
                param_name = f"p{i}"
                if isinstance(v, list) and len(v) == 2 and k.endswith("_date"):
                    conditions.append(f"{k} >= :{param_name}_start AND {k} <= :{param_name}_end")
                    query_params[f"{param_name}_start"] = v[0]
                    query_params[f"{param_name}_end"] = v[1]
                else:
                    conditions.append(f"{k} = :{param_name}")
                    query_params[param_name] = v
            if conditions:
                where_clause = " WHERE " + " AND ".join(conditions)

        # If it's a simple formula, wrap it in a SELECT
        if not formula.strip().upper().startswith("SELECT"):
            # Try to determine the primary table (defaulting to order_items if ambiguous)
            table = kpi["tables"][0] if kpi["tables"] else "order_items"
            query = f"SELECT {formula} as value FROM {table}{where_clause}"
        else:
            if where_clause:
                # Basic wrapping — note: if formula is complex, this might need refinement
                query = f"SELECT * FROM ({formula}) sub {where_clause}"
            else:
                query = formula

        from sqlalchemy import text
        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn, params=query_params)
            return df

    def get_all_kpis_summary(self):
        """Returns a dictionary of all core KPI values for a dashboard summary."""
        summary = {}
        # Core summary metrics
        core_metrics = ["revenue", "aov", "on_time_delivery_rate", "avg_review_score"]
        
        for mid in core_metrics:
            try:
                res = self.calculate_kpi(mid)
                summary[mid] = res.iloc[0, 0]
            except:
                summary[mid] = None
        return summary

if __name__ == "__main__":
    # Test the engine
    engine = KPIEngine()
    print("Testing KPI Engine...")
    summary = engine.get_all_kpis_summary()
    print(json.dumps(summary, indent=2))
