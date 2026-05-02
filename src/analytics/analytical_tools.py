"""
Analytical Intelligence Layer
==============================
Deterministic analytical tools for trend analysis, comparisons,
root-cause investigation, and business recommendations.
These power the multi-step reasoning in the Agentic RAG system.
"""

import sqlite3
import json
import pandas as pd
import yaml

with open("config/config.yaml") as f:
    _cfg = yaml.safe_load(f)

DB_PATH = _cfg["paths"]["olist_db"]


def _conn():
    safe = DB_PATH.replace("\\", "/")
    return sqlite3.connect(f"file:{safe}?mode=ro&uri=true", uri=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Revenue Trend Analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyze_revenue_trends() -> str:
    """Monthly revenue trends with MoM growth rates."""
    sql = """
    SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month,
           ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
           COUNT(DISTINCT o.order_id) AS orders
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1 ORDER BY 1
    """
    conn = _conn()
    try:
        df = pd.read_sql_query(sql, conn)
        if df.empty:
            return "No revenue data available."

        df["mom_growth"] = df["revenue"].pct_change() * 100
        df["avg_order_value"] = (df["revenue"] / df["orders"]).round(2)

        recent = df.tail(6)
        overall_trend = "GROWING" if recent["mom_growth"].mean() > 0 else "DECLINING"
        peak = df.loc[df["revenue"].idxmax()]
        low = df.loc[df["revenue"].idxmin()]

        lines = [
            "[Revenue Trend Analysis]\n",
            f"Overall Trend (last 6 months): {overall_trend}",
            f"Avg MoM Growth (last 6 months): {recent['mom_growth'].mean():.1f}%",
            f"Peak Month: {peak['month']} (R${peak['revenue']:,.2f})",
            f"Lowest Month: {low['month']} (R${low['revenue']:,.2f})\n",
            "Monthly Breakdown (last 6):",
        ]
        for _, r in recent.iterrows():
            g = f"+{r['mom_growth']:.1f}%" if r["mom_growth"] > 0 else f"{r['mom_growth']:.1f}%"
            lines.append(f"  {r['month']}: R${r['revenue']:,.2f} | Orders: {int(r['orders'])} | MoM: {g}")

        return "\n".join(lines)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Delivery Performance Analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyze_delivery_performance() -> str:
    """Delivery delays, on-time rates, and avg delivery time by month."""
    sql = """
    SELECT strftime('%Y-%m', order_purchase_timestamp) AS month,
           COUNT(*) AS total_orders,
           SUM(CASE WHEN order_delivered_customer_date <= order_estimated_delivery_date THEN 1 ELSE 0 END) AS on_time,
           ROUND(AVG(julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp)), 1) AS avg_delivery_days,
           ROUND(AVG(julianday(order_delivered_customer_date) - julianday(order_estimated_delivery_date)), 1) AS avg_delay_days
    FROM orders
    WHERE order_status = 'delivered'
      AND order_delivered_customer_date IS NOT NULL
    GROUP BY 1 ORDER BY 1
    """
    conn = _conn()
    try:
        df = pd.read_sql_query(sql, conn)
        if df.empty:
            return "No delivery data available."

        df["on_time_rate"] = (df["on_time"] / df["total_orders"] * 100).round(1)
        recent = df.tail(6)
        trend = "IMPROVING" if recent["on_time_rate"].iloc[-1] > recent["on_time_rate"].iloc[0] else "DECLINING"

        lines = [
            "[Delivery Performance Analysis]\n",
            f"On-Time Delivery Trend: {trend}",
            f"Current On-Time Rate: {recent['on_time_rate'].iloc[-1]:.1f}%",
            f"Avg Delivery Time (recent): {recent['avg_delivery_days'].mean():.1f} days\n",
            "Monthly Breakdown (last 6):",
        ]
        for _, r in recent.iterrows():
            lines.append(
                f"  {r['month']}: On-Time {r['on_time_rate']:.1f}% | "
                f"Avg {r['avg_delivery_days']} days | Delay: {r['avg_delay_days']} days"
            )
        return "\n".join(lines)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Customer Behavior Analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyze_customer_behavior() -> str:
    """Repeat purchase rates, new vs returning customers, cohort trends."""
    sql_repeat = """
    SELECT customer_unique_id, COUNT(DISTINCT order_id) AS order_count
    FROM customers c JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1
    """
    sql_monthly = """
    SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month,
           COUNT(DISTINCT c.customer_unique_id) AS unique_customers,
           COUNT(DISTINCT o.order_id) AS orders
    FROM orders o JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1 ORDER BY 1
    """
    conn = _conn()
    try:
        df_rep = pd.read_sql_query(sql_repeat, conn)
        df_monthly = pd.read_sql_query(sql_monthly, conn)

        total = len(df_rep)
        one_time = len(df_rep[df_rep["order_count"] == 1])
        repeat = total - one_time
        repeat_rate = (repeat / total * 100) if total > 0 else 0

        recent = df_monthly.tail(6)
        cust_trend = "GROWING" if recent["unique_customers"].iloc[-1] > recent["unique_customers"].iloc[0] else "DECLINING"

        lines = [
            "[Customer Behavior Analysis]\n",
            f"Total Unique Customers: {total:,}",
            f"One-Time Buyers: {one_time:,} ({one_time/total*100:.1f}%)",
            f"Repeat Buyers: {repeat:,} ({repeat_rate:.1f}%)",
            f"Customer Acquisition Trend: {cust_trend}\n",
            "Monthly Unique Customers (last 6):",
        ]
        for _, r in recent.iterrows():
            lines.append(f"  {r['month']}: {int(r['unique_customers']):,} customers | {int(r['orders']):,} orders")

        return "\n".join(lines)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Review Score Analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyze_review_scores() -> str:
    """Review score distribution and trends by month."""
    sql = """
    SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS month,
           ROUND(AVG(r.review_score), 2) AS avg_score,
           COUNT(*) AS review_count,
           SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS low_reviews,
           SUM(CASE WHEN r.review_score >= 4 THEN 1 ELSE 0 END) AS high_reviews
    FROM order_reviews r
    JOIN orders o ON r.order_id = o.order_id
    GROUP BY 1 ORDER BY 1
    """
    conn = _conn()
    try:
        df = pd.read_sql_query(sql, conn)
        if df.empty:
            return "No review data."

        df["low_pct"] = (df["low_reviews"] / df["review_count"] * 100).round(1)
        recent = df.tail(6)
        trend = "IMPROVING" if recent["avg_score"].iloc[-1] > recent["avg_score"].iloc[0] else "DECLINING"

        lines = [
            "[Review Score Analysis]\n",
            f"Score Trend: {trend}",
            f"Current Avg Score: {recent['avg_score'].iloc[-1]}/5",
            f"Low Review Rate (≤2 stars): {recent['low_pct'].mean():.1f}%\n",
            "Monthly Breakdown (last 6):",
        ]
        for _, r in recent.iterrows():
            lines.append(f"  {r['month']}: Avg {r['avg_score']}/5 | Low: {r['low_pct']:.1f}% | Reviews: {int(r['review_count'])}")

        return "\n".join(lines)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Category Performance Analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyze_category_performance() -> str:
    """Top and bottom product categories by revenue and growth."""
    sql = """
    SELECT t.product_category_name_english AS category,
           ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
           COUNT(DISTINCT oi.order_id) AS orders,
           ROUND(AVG(r.review_score), 2) AS avg_review
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
    JOIN orders o ON oi.order_id = o.order_id
    LEFT JOIN order_reviews r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1
    HAVING orders >= 10
    ORDER BY revenue DESC
    """
    conn = _conn()
    try:
        df = pd.read_sql_query(sql, conn)
        if df.empty:
            return "No category data."

        top5 = df.head(5)
        bottom5 = df.tail(5)

        lines = ["[Category Performance Analysis]\n", "Top 5 Categories by Revenue:"]
        for _, r in top5.iterrows():
            lines.append(f"  {r['category']}: R${r['revenue']:,.2f} | Orders: {int(r['orders'])} | Review: {r['avg_review']}/5")

        lines.append("\nBottom 5 Categories:")
        for _, r in bottom5.iterrows():
            lines.append(f"  {r['category']}: R${r['revenue']:,.2f} | Orders: {int(r['orders'])} | Review: {r['avg_review']}/5")

        return "\n".join(lines)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Seller Performance Analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyze_seller_performance() -> str:
    """Seller distribution, top sellers, and delivery impact."""
    sql = """
    SELECT s.seller_id,
           COUNT(DISTINCT oi.order_id) AS orders,
           ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
           ROUND(AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp)), 1) AS avg_delivery_days
    FROM sellers s
    JOIN order_items oi ON s.seller_id = oi.seller_id
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL
    GROUP BY 1
    HAVING orders >= 5
    ORDER BY revenue DESC
    """
    conn = _conn()
    try:
        df = pd.read_sql_query(sql, conn)
        if df.empty:
            return "No seller data."

        total_sellers = len(df)
        top10_rev = df.head(10)["revenue"].sum()
        total_rev = df["revenue"].sum()
        concentration = top10_rev / total_rev * 100

        slow = df[df["avg_delivery_days"] > df["avg_delivery_days"].quantile(0.75)]

        lines = [
            "[Seller Performance Analysis]\n",
            f"Active Sellers (≥5 orders): {total_sellers}",
            f"Top 10 Sellers Revenue Share: {concentration:.1f}%",
            f"Avg Delivery Time: {df['avg_delivery_days'].mean():.1f} days",
            f"Slow Sellers (>75th pctl): {len(slow)} sellers, avg {slow['avg_delivery_days'].mean():.1f} days",
        ]
        return "\n".join(lines)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Root Cause Aggregator
# ─────────────────────────────────────────────────────────────────────────────
def investigate_root_causes(topic: str = "general") -> str:
    """Run multiple analyses and aggregate findings for root-cause investigation."""
    sections = []

    if topic in ("general", "revenue", "sales"):
        sections.append(analyze_revenue_trends())
        sections.append(analyze_category_performance())

    if topic in ("general", "delivery", "logistics"):
        sections.append(analyze_delivery_performance())
        sections.append(analyze_seller_performance())

    if topic in ("general", "customer", "retention", "churn"):
        sections.append(analyze_customer_behavior())
        sections.append(analyze_review_scores())

    if topic in ("general",):
        sections.append(analyze_review_scores())

    # Deduplicate
    seen = set()
    unique = []
    for s in sections:
        key = s[:50]
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return "\n\n" + ("\n\n---\n\n".join(unique))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Geographic Analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyze_geographic_distribution() -> str:
    """Revenue and order distribution by state."""
    sql = """
    SELECT c.customer_state AS state,
           COUNT(DISTINCT o.order_id) AS orders,
           ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
           ROUND(AVG(r.review_score), 2) AS avg_review
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN order_reviews r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1 ORDER BY revenue DESC
    LIMIT 10
    """
    conn = _conn()
    try:
        df = pd.read_sql_query(sql, conn)
        if df.empty:
            return "No geographic data."

        lines = ["[Geographic Distribution — Top 10 States]\n"]
        for _, r in df.iterrows():
            lines.append(f"  {r['state']}: R${r['revenue']:,.2f} | Orders: {int(r['orders'])} | Review: {r['avg_review']}/5")

        top3_share = df.head(3)["revenue"].sum() / df["revenue"].sum() * 100
        lines.append(f"\nTop 3 states account for {top3_share:.1f}% of revenue.")
        return "\n".join(lines)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 10. Market Basket Analysis (Category Correlations)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_market_basket() -> str:
    """Identify top product category pairs frequently bought together in the same order."""
    sql = """
    SELECT t1.product_category_name_english AS cat1,
           t2.product_category_name_english AS cat2,
           COUNT(*) AS frequency
    FROM order_items oi1
    JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
    JOIN products p1 ON oi1.product_id = p1.product_id
    JOIN products p2 ON oi2.product_id = p2.product_id
    JOIN product_category_name_translation t1 ON p1.product_category_name = t1.product_category_name
    JOIN product_category_name_translation t2 ON p2.product_category_name = t2.product_category_name
    GROUP BY 1, 2
    ORDER BY frequency DESC
    LIMIT 10
    """
    conn = _conn()
    try:
        df = pd.read_sql_query(sql, conn)
        if df.empty:
            return "No cross-category purchase patterns found."

        lines = ["[Market Basket Analysis — Top Category Pairs]\n"]
        for _, r in df.iterrows():
            lines.append(f"  {r['cat1']} + {r['cat2']} : {r['frequency']} times")
        
        lines.append("\nInsight: Use these pairs for 'Frequently Bought Together' recommendations.")
        return "\n".join(lines)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 11. Customer Lifetime Value (CLV) by Segment
# ─────────────────────────────────────────────────────────────────────────────
def estimate_clv_by_segment() -> str:
    """Estimate historical CLV based on total spend and order frequency."""
    sql = """
    SELECT customer_unique_id,
           COUNT(DISTINCT order_id) AS frequency,
           SUM(price + freight_value) AS total_spent
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1
    """
    conn = _conn()
    try:
        df = pd.read_sql_query(sql, conn)
        if df.empty:
            return "No CLV data available."

        # Simplistic segmentation based on quantiles
        df['segment'] = pd.qcut(df['total_spent'], 4, labels=['Low-Value', 'Mid-Value', 'High-Value', 'Top-Tier'])
        
        clv_stats = df.groupby('segment').agg({
            'customer_unique_id': 'count',
            'total_spent': 'mean',
            'frequency': 'mean'
        }).rename(columns={'customer_unique_id': 'n_customers', 'total_spent': 'avg_clv', 'frequency': 'avg_freq'})

        lines = ["[Estimated CLV by Segment — Historical Basis]\n"]
        for seg, r in clv_stats.iterrows():
            lines.append(
                f"  {seg:<12} | Customers: {int(r['n_customers']):>5} | "
                f"Avg CLV: R${r['avg_clv']:>7.2f} | Avg Freq: {r['avg_freq']:.2f}"
            )
        
        overall_avg = df['total_spent'].mean()
        lines.append(f"\nOverall Marketplace Avg CLV: R${overall_avg:.2f}")
        return "\n".join(lines)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 12. Order Cancellation Analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyze_order_cancellation() -> str:
    """Analyze cancellation rates and potential revenue loss."""
    sql = """
    SELECT order_status,
           COUNT(*) AS order_count,
           strftime('%Y-%m', order_purchase_timestamp) AS month
    FROM orders
    GROUP BY 1, 3
    """
    conn = _conn()
    try:
        df = pd.read_sql_query(sql, conn)
        if df.empty:
            return "No cancellation data."

        pivot = df.pivot(index='month', columns='order_status', values='order_count').fillna(0)
        if 'canceled' not in pivot.columns:
            return "No canceled orders found in data."
            
        pivot['total'] = pivot.sum(axis=1)
        pivot['cancel_rate'] = (pivot['canceled'] / pivot['total'] * 100).round(2)
        
        recent = pivot.tail(6)
        avg_rate = recent['cancel_rate'].mean()

        lines = ["[Order Cancellation Analysis]\n"]
        lines.append(f"Avg Recent Cancellation Rate: {avg_rate:.2f}%")
        lines.append("Monthly Cancellation Trend:")
        for m, r in recent.iterrows():
            lines.append(f"  {m}: {r['canceled']:.0f} canceled / {r['total']:.0f} total ({r['cancel_rate']:.2f}%)")
            
        return "\n".join(lines)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry (for planner to select from)
# ─────────────────────────────────────────────────────────────────────────────
ANALYTICAL_TOOLS = {
    "analyze_revenue_trends": {
        "fn": analyze_revenue_trends,
        "description": "Monthly revenue trends with MoM growth rates",
        "topics": ["revenue", "sales", "growth"],
    },
    "analyze_delivery_performance": {
        "fn": analyze_delivery_performance,
        "description": "Delivery delays, on-time rates, avg delivery time",
        "topics": ["delivery", "logistics", "shipping"],
    },
    "analyze_customer_behavior": {
        "fn": analyze_customer_behavior,
        "description": "Repeat purchase rates, new vs returning customers",
        "topics": ["customer", "retention", "churn", "repeat"],
    },
    "analyze_review_scores": {
        "fn": analyze_review_scores,
        "description": "Review score distribution and trends",
        "topics": ["reviews", "satisfaction", "quality"],
    },
    "analyze_category_performance": {
        "fn": analyze_category_performance,
        "description": "Top/bottom product categories by revenue",
        "topics": ["products", "categories", "assortment"],
    },
    "analyze_seller_performance": {
        "fn": analyze_seller_performance,
        "description": "Seller distribution, delivery impact",
        "topics": ["sellers", "marketplace", "supply"],
    },
    "analyze_geographic_distribution": {
        "fn": analyze_geographic_distribution,
        "description": "Revenue and orders by state/region",
        "topics": ["geography", "region", "state", "location"],
    },
    "analyze_payment_patterns": {
        "fn": analyze_payment_patterns,
        "description": "Payment method distribution and installments",
        "topics": ["payments", "installments", "credit"],
    },
    "analyze_market_basket": {
        "fn": analyze_market_basket,
        "description": "Cross-selling patterns and category correlations",
        "topics": ["cross-sell", "market basket", "recommendations", "association"],
    },
    "estimate_clv_by_segment": {
        "fn": estimate_clv_by_segment,
        "description": "Customer Lifetime Value estimation by value segment",
        "topics": ["clv", "lifetime value", "profitability", "segments"],
    },
    "analyze_order_cancellation": {
        "fn": analyze_order_cancellation,
        "description": "Order cancellation rates and revenue leakage",
        "topics": ["cancellations", "leakage", "returns", "issues"],
    },
    "investigate_root_causes": {
        "fn": investigate_root_causes,
        "description": "Multi-dimensional root cause investigation",
        "topics": ["why", "root cause", "decline", "problem", "issue"],
    },
}
