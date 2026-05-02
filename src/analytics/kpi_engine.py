import sqlite3
import json
import pandas as pd
import yaml
import os

class KPIEngine:
    """
    Central engine for calculating deterministic business KPIs.
    Ensures that Dashboards and Chatbots use the exact same logic.
    """
    def __init__(self, config_path="config/config.yaml", kpi_path="config/kpi_definitions.json"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        self.db_path = cfg["paths"]["olist_db"]
        
        with open(kpi_path) as f:
            self.definitions = json.load(f)

    def _get_connection(self):
        # Use read-only mode for safety
        safe_path = self.db_path.replace("\\", "/")
        return sqlite3.connect(f"file:{safe_path}?mode=ro&uri=true", uri=True)

    def calculate_kpi(self, kpi_id, filters: dict = None):
        """Calculate a specific KPI by its ID from the registry."""
        if kpi_id not in self.definitions:
            raise ValueError(f"KPI '{kpi_id}' not found in registry.")
        
        kpi = self.definitions[kpi_id]
        formula = kpi["sql_formula"]
        
        where_clause = ""
        params = []
        if filters:
            conditions = []
            for k, v in filters.items():
                if isinstance(v, list) and len(v) == 2 and k.endswith("_date"):
                    conditions.append(f"{k} >= ? AND {k} <= ?")
                    params.extend(v)
                else:
                    conditions.append(f"{k} = ?")
                    params.append(v)
            if conditions:
                where_clause = " WHERE " + " AND ".join(conditions)

        # If it's a simple formula, wrap it in a SELECT
        if not formula.strip().upper().startswith("SELECT"):
            # Try to determine the primary table (defaulting to order_items if ambiguous)
            table = kpi["tables"][0] if kpi["tables"] else "order_items"
            query = f"SELECT {formula} as value FROM {table}{where_clause}"
        else:
            if where_clause:
                query = f"SELECT * FROM ({formula}) {where_clause}"
            else:
                query = formula

        conn = self._get_connection()
        try:
            df = pd.read_sql_query(query, conn, params=params)
            return df
        finally:
            conn.close()

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
