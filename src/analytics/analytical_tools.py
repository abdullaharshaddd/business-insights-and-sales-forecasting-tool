"""
Analytical Intelligence Layer
==============================
Deterministic analytical tools — all queries run against BISFT PostgreSQL.
No CSV files. No SQLite. No old table names.
"""

from sqlalchemy import create_engine, text
import pandas as pd
import yaml

with open("config/config.yaml") as f:
    _cfg = yaml.safe_load(f)

DB_URL = _cfg["database"]["url"]
engine = create_engine(DB_URL)


def _conn():
    return engine.connect()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Revenue Trend Analysis (Olist)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_revenue_trends() -> str:
    sql = """
    SELECT TO_CHAR(o.order_purchase_timestamp, 'YYYY-MM') AS month,
           ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
           COUNT(DISTINCT oi.order_id) AS orders
    FROM olist_orders o
    JOIN olist_order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered' AND o.order_purchase_timestamp IS NOT NULL
    GROUP BY 1 ORDER BY 1
    """
    with _conn() as conn:
        try:
            df = pd.read_sql_query(text(sql), conn)
            if df.empty:
                return "No revenue data available."

            df["mom_growth"] = df["revenue"].pct_change() * 100
            df["aov"] = (df["revenue"] / df["orders"]).astype(float).round(2)

            recent = df.tail(6)
            overall_trend = "GROWING" if recent["mom_growth"].mean() > 0 else "DECLINING"
            peak = df.loc[df["revenue"].idxmax()]
            low = df.loc[df["revenue"].idxmin()]

            lines = [
                "[Revenue Trend Analysis — Olist]\n",
                f"Overall Trend (last 6 months): {overall_trend}",
                f"Avg MoM Growth (last 6 months): {recent['mom_growth'].mean():.1f}%",
                f"Peak Month: {peak['month']} (R${peak['revenue']:,.2f})",
                f"Lowest Month: {low['month']} (R${low['revenue']:,.2f})\n",
                "Monthly Breakdown (last 6):",
            ]
            for _, r in recent.iterrows():
                g = f"+{r['mom_growth']:.1f}%" if r["mom_growth"] > 0 else f"{r['mom_growth']:.1f}%"
                lines.append(f"  {r['month']}: R${r['revenue']:,.2f} | Orders: {int(r['orders'])} | AOV: R${r['aov']} | MoM: {g}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error analyzing revenue: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Delivery Performance (Olist)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_delivery_performance() -> str:
    sql = """
    SELECT TO_CHAR(order_purchase_timestamp, 'YYYY-MM') AS month,
           COUNT(*) AS total_orders,
           SUM(CASE WHEN order_delivered_customer_date <= order_estimated_delivery_date THEN 1 ELSE 0 END) AS on_time,
           ROUND(AVG(EXTRACT(DAY FROM (order_delivered_customer_date - order_purchase_timestamp)))::numeric, 1) AS avg_days,
           ROUND(AVG(EXTRACT(DAY FROM (order_delivered_customer_date - order_estimated_delivery_date)))::numeric, 1) AS avg_delay
    FROM olist_orders
    WHERE order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL
    GROUP BY 1 ORDER BY 1
    """
    with _conn() as conn:
        try:
            df = pd.read_sql_query(text(sql), conn)
            if df.empty:
                return "No delivery data available."

            df["on_time_rate"] = (df["on_time"] / df["total_orders"] * 100).astype(float).round(1)
            recent = df.tail(6)
            trend = "IMPROVING" if recent["on_time_rate"].iloc[-1] > recent["on_time_rate"].iloc[0] else "DECLINING"

            lines = [
                "[Delivery Performance — Olist]\n",
                f"On-Time Delivery Trend: {trend}",
                f"Current On-Time Rate: {recent['on_time_rate'].iloc[-1]:.1f}%",
                f"Avg Delivery Time (recent): {recent['avg_days'].mean():.1f} days\n",
                "Monthly Breakdown (last 6):",
            ]
            for _, r in recent.iterrows():
                lines.append(
                    f"  {r['month']}: On-Time {r['on_time_rate']:.1f}% | "
                    f"Avg {r['avg_days']} days | Delay: {r['avg_delay']} days"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"Error analyzing delivery: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Customer Behavior (Olist)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_customer_behavior() -> str:
    sql_repeat = """
    SELECT c.customer_unique_id, COUNT(DISTINCT o.order_id) AS order_count
    FROM olist_customers c
    JOIN olist_orders o ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1
    """
    sql_monthly = """
    SELECT TO_CHAR(o.order_purchase_timestamp, 'YYYY-MM') AS month,
           COUNT(DISTINCT c.customer_unique_id) AS unique_customers,
           COUNT(DISTINCT o.order_id) AS orders
    FROM olist_orders o
    JOIN olist_customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered' AND o.order_purchase_timestamp IS NOT NULL
    GROUP BY 1 ORDER BY 1
    """
    with _conn() as conn:
        try:
            df_rep = pd.read_sql_query(text(sql_repeat), conn)
            df_monthly = pd.read_sql_query(text(sql_monthly), conn)

            total = len(df_rep)
            one_time = len(df_rep[df_rep["order_count"] == 1])
            repeat = total - one_time
            repeat_rate = (repeat / total * 100) if total > 0 else 0

            recent = df_monthly.tail(6)
            cust_trend = "GROWING" if recent["unique_customers"].iloc[-1] > recent["unique_customers"].iloc[0] else "DECLINING"

            lines = [
                "[Customer Behavior — Olist]\n",
                f"Total Unique Customers: {total:,}",
                f"One-Time Buyers: {one_time:,} ({one_time/total*100:.1f}%)",
                f"Repeat Buyers: {repeat:,} ({repeat_rate:.1f}%)",
                f"Customer Acquisition Trend: {cust_trend}\n",
                "Monthly Unique Customers (last 6):",
            ]
            for _, r in recent.iterrows():
                lines.append(f"  {r['month']}: {int(r['unique_customers']):,} customers | {int(r['orders']):,} orders")

            return "\n".join(lines)
        except Exception as e:
            return f"Error analyzing customers: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Review Score Analysis (Olist)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_review_scores() -> str:
    sql = """
    SELECT TO_CHAR(o.order_purchase_timestamp, 'YYYY-MM') AS month,
           ROUND(AVG(r.review_score)::numeric, 2) AS avg_score,
           COUNT(*) AS review_count,
           SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS low_reviews,
           SUM(CASE WHEN r.review_score >= 4 THEN 1 ELSE 0 END) AS high_reviews
    FROM olist_order_reviews r
    JOIN olist_orders o ON r.order_id = o.order_id
    WHERE r.review_score IS NOT NULL
    GROUP BY 1 ORDER BY 1
    """
    with _conn() as conn:
        try:
            df = pd.read_sql_query(text(sql), conn)
            if df.empty:
                return "No review data."

            df["low_pct"] = (df["low_reviews"] / df["review_count"] * 100).astype(float).round(1)
            recent = df.tail(6)
            trend = "IMPROVING" if recent["avg_score"].iloc[-1] > recent["avg_score"].iloc[0] else "DECLINING"

            lines = [
                "[Review Score Analysis — Olist]\n",
                f"Score Trend: {trend}",
                f"Current Avg Score: {recent['avg_score'].iloc[-1]}/5",
                f"Low Review Rate (≤2 stars): {recent['low_pct'].mean():.1f}%\n",
                "Monthly Breakdown (last 6):",
            ]
            for _, r in recent.iterrows():
                lines.append(f"  {r['month']}: Avg {r['avg_score']}/5 | Low: {r['low_pct']:.1f}% | Reviews: {int(r['review_count'])}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error analyzing reviews: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Category Performance (Olist)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_category_performance() -> str:
    sql = """
    SELECT t.category_name_en AS category,
           ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
           COUNT(DISTINCT oi.order_id) AS orders,
           ROUND(AVG(r.review_score)::numeric, 2) AS avg_review
    FROM olist_order_items oi
    JOIN olist_orders o ON oi.order_id = o.order_id
    JOIN olist_products p ON oi.product_id = p.product_id
    LEFT JOIN olist_product_category_translation t ON p.category_name_pt = t.category_name_pt
    LEFT JOIN olist_order_reviews r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1
    HAVING COUNT(DISTINCT oi.order_id) >= 10
    ORDER BY revenue DESC
    """
    with _conn() as conn:
        try:
            df = pd.read_sql_query(text(sql), conn)
            if df.empty:
                return "No category data."

            top5 = df.head(5)
            bottom5 = df.tail(5)

            lines = ["[Category Performance — Olist]\nTop 5 Categories by Revenue:"]
            for _, r in top5.iterrows():
                lines.append(f"  {r['category']}: R${r['revenue']:,.2f} | Orders: {int(r['orders'])} | Review: {r['avg_review']}/5")

            lines.append("\nBottom 5 Categories:")
            for _, r in bottom5.iterrows():
                lines.append(f"  {r['category']}: R${r['revenue']:,.2f} | Orders: {int(r['orders'])} | Review: {r['avg_review']}/5")

            return "\n".join(lines)
        except Exception as e:
            return f"Error analyzing categories: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Seller Performance (Olist)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_seller_performance() -> str:
    sql = """
    SELECT s.seller_id, s.city, s.state,
           COUNT(DISTINCT oi.order_id) AS orders,
           ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
           ROUND(AVG(EXTRACT(DAY FROM (o.order_delivered_customer_date - o.order_purchase_timestamp)))::numeric, 1) AS avg_days
    FROM olist_sellers s
    JOIN olist_order_items oi ON s.seller_id = oi.seller_id
    JOIN olist_orders o ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL
    GROUP BY 1, 2, 3
    HAVING COUNT(DISTINCT oi.order_id) >= 5
    ORDER BY revenue DESC
    LIMIT 20
    """
    with _conn() as conn:
        try:
            df = pd.read_sql_query(text(sql), conn)
            if df.empty:
                return "No seller data."

            total_sellers = len(df)
            top10_rev = df.head(10)["revenue"].sum()
            total_rev = df["revenue"].sum()
            concentration = (top10_rev / total_rev * 100) if total_rev > 0 else 0
            slow = df[df["avg_days"] > df["avg_days"].quantile(0.75)]

            lines = [
                "[Seller Performance — Olist]\n",
                f"Active Sellers (≥5 orders): {total_sellers}",
                f"Top 10 Sellers Revenue Share: {concentration:.1f}%",
                f"Avg Delivery Time: {df['avg_days'].mean():.1f} days",
                f"Slow Sellers (>75th pctl): {len(slow)} sellers, avg {slow['avg_days'].mean():.1f} days\n",
                "Top 10 Sellers:",
            ]
            for _, r in df.head(10).iterrows():
                loc = f"{r['city']}, {r['state']}" if r['city'] else '—'
                lines.append(f"  {r['seller_id'][:12]}... ({loc}): R${r['revenue']:,.2f} | {int(r['orders'])} orders")

            return "\n".join(lines)
        except Exception as e:
            return f"Error analyzing sellers: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Geographic Distribution (Olist)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_geographic_distribution() -> str:
    sql = """
    SELECT c.state,
           COUNT(DISTINCT o.order_id) AS orders,
           ROUND(SUM(oi.price + oi.freight_value), 2) AS revenue,
           ROUND(AVG(r.review_score)::numeric, 2) AS avg_review
    FROM olist_customers c
    JOIN olist_orders o ON c.customer_id = o.customer_id
    JOIN olist_order_items oi ON o.order_id = oi.order_id
    LEFT JOIN olist_order_reviews r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1 ORDER BY revenue DESC
    LIMIT 10
    """
    with _conn() as conn:
        try:
            df = pd.read_sql_query(text(sql), conn)
            if df.empty:
                return "No geographic data."

            lines = ["[Geographic Distribution — Olist Top 10 States]\n"]
            for _, r in df.iterrows():
                lines.append(f"  {r['state']}: R${r['revenue']:,.2f} | Orders: {int(r['orders'])} | Review: {r['avg_review']}/5")

            top3_share = (df.head(3)["revenue"].sum() / df["revenue"].sum() * 100) if not df.empty else 0
            lines.append(f"\nTop 3 states account for {top3_share:.1f}% of revenue.")
            return "\n".join(lines)
        except Exception as e:
            return f"Error analyzing geography: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Market Basket Analysis (Olist)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_market_basket() -> str:
    sql = """
    SELECT t1.category_name_en AS cat1,
           t2.category_name_en AS cat2,
           COUNT(*) AS frequency
    FROM olist_order_items oi1
    JOIN olist_order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
    JOIN olist_products p1 ON oi1.product_id = p1.product_id
    JOIN olist_products p2 ON oi2.product_id = p2.product_id
    JOIN olist_product_category_translation t1 ON p1.category_name_pt = t1.category_name_pt
    JOIN olist_product_category_translation t2 ON p2.category_name_pt = t2.category_name_pt
    GROUP BY 1, 2
    ORDER BY frequency DESC
    LIMIT 10
    """
    with _conn() as conn:
        try:
            df = pd.read_sql_query(text(sql), conn)
            if df.empty:
                return "No cross-category purchase patterns found."

            lines = ["[Market Basket Analysis — Olist Category Pairs]\n"]
            for _, r in df.iterrows():
                lines.append(f"  {r['cat1']} + {r['cat2']}: {r['frequency']} times")

            lines.append("\nInsight: Use these pairs for 'Frequently Bought Together' recommendations.")
            return "\n".join(lines)
        except Exception as e:
            return f"Error in market basket: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. CLV by Segment (Olist)
# ─────────────────────────────────────────────────────────────────────────────
def estimate_clv_by_segment() -> str:
    sql = """
    SELECT c.customer_unique_id,
           COUNT(DISTINCT o.order_id) AS frequency,
           SUM(oi.price + oi.freight_value) AS total_spent
    FROM olist_customers c
    JOIN olist_orders o ON c.customer_id = o.customer_id
    JOIN olist_order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1
    """
    with _conn() as conn:
        try:
            df = pd.read_sql_query(text(sql), conn)
            if df.empty:
                return "No CLV data available."

            df['total_spent'] = df['total_spent'].astype(float)
            df['segment'] = pd.qcut(df['total_spent'], 4, labels=['Low-Value', 'Mid-Value', 'High-Value', 'Top-Tier'])

            clv_stats = df.groupby('segment').agg({
                'customer_unique_id': 'count',
                'total_spent': 'mean',
                'frequency': 'mean'
            }).rename(columns={'customer_unique_id': 'n_customers', 'total_spent': 'avg_clv', 'frequency': 'avg_freq'})

            lines = ["[Estimated CLV by Segment — Olist Historical]\n"]
            for seg, r in clv_stats.iterrows():
                lines.append(
                    f"  {seg:<12} | Customers: {int(r['n_customers']):>5} | "
                    f"Avg CLV: R${r['avg_clv']:>7.2f} | Avg Freq: {r['avg_freq']:.2f}"
                )

            overall_avg = df['total_spent'].mean()
            lines.append(f"\nOverall Marketplace Avg CLV: R${overall_avg:.2f}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error estimating CLV: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Order Cancellation Analysis (Olist)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_order_cancellation() -> str:
    sql = """
    SELECT order_status, COUNT(*) AS order_count,
           TO_CHAR(order_purchase_timestamp, 'YYYY-MM') AS month
    FROM olist_orders
    WHERE order_purchase_timestamp IS NOT NULL
    GROUP BY 1, 3
    ORDER BY 3, 2
    """
    with _conn() as conn:
        try:
            df = pd.read_sql_query(text(sql), conn)
            if df.empty:
                return "No cancellation data."

            pivot = df.pivot(index='month', columns='order_status', values='order_count').fillna(0)
            if 'canceled' not in pivot.columns:
                return "No canceled orders found."

            pivot['total'] = pivot.sum(axis=1)
            pivot['cancel_rate'] = (pivot['canceled'] / pivot['total'] * 100).round(2)

            recent = pivot.tail(6)
            avg_rate = recent['cancel_rate'].mean()

            lines = ["[Order Cancellation — Olist]\n"]
            lines.append(f"Avg Recent Cancellation Rate: {avg_rate:.2f}%")
            lines.append("Monthly Cancellation Trend:")
            for m, r in recent.iterrows():
                total = int(r['total'])
                canceled = int(r.get('canceled', 0))
                rate = r.get('cancel_rate', 0)
                lines.append(f"  {m}: {canceled} canceled / {total} total ({rate:.2f}%)")

            return "\n".join(lines)
        except Exception as e:
            return f"Error analyzing cancellations: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 11. Online Retail Overview (UK)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_online_retail_overview() -> str:
    sql_summary = """
    SELECT COUNT(DISTINCT ri.customerid) AS customers,
           COUNT(DISTINCT ri.invoiceno) AS invoices,
           SUM(rii.quantity) AS units_sold,
           SUM(rii.totalprice) AS total_revenue
    FROM retail_invoices ri
    JOIN retail_invoice_items rii ON ri.invoiceno = rii.invoiceno
    WHERE ri.is_cancellation = FALSE
    """
    sql_monthly = """
    SELECT TO_CHAR(ri.invoicedate, 'YYYY-MM') AS month,
           SUM(rii.totalprice) AS revenue,
           COUNT(DISTINCT ri.invoiceno) AS invoices
    FROM retail_invoices ri
    JOIN retail_invoice_items rii ON ri.invoiceno = rii.invoiceno
    WHERE ri.is_cancellation = FALSE AND ri.invoicedate IS NOT NULL
    GROUP BY 1 ORDER BY 1
    """
    sql_country = """
    SELECT rc.country, COUNT(DISTINCT ri.invoiceno) AS invoices, SUM(rii.totalprice) AS revenue
    FROM retail_invoices ri
    JOIN retail_invoice_items rii ON ri.invoiceno = rii.invoiceno
    JOIN retail_customers rc ON ri.customerid = rc.customerid
    WHERE ri.is_cancellation = FALSE
    GROUP BY 1 ORDER BY 3 DESC LIMIT 10
    """
    with _conn() as conn:
        try:
            df_sum = pd.read_sql_query(text(sql_summary), conn)
            df_month = pd.read_sql_query(text(sql_monthly), conn)
            df_country = pd.read_sql_query(text(sql_country), conn)

            r = df_sum.iloc[0]

            lines = [
                "[Online Retail Overview — UK Dataset]\n",
                f"Total Customers: {int(r['customers']):,}",
                f"Total Invoices: {int(r['invoices']):,}",
                f"Units Sold: {int(r['units_sold']):,}",
                f"Total Revenue: £{float(r['total_revenue']):,.2f}\n",
                "Top 10 Countries by Revenue:",
            ]
            for _, row in df_country.iterrows():
                lines.append(f"  {row['country']}: £{float(row['revenue']):,.2f} | {int(row['invoices'])} invoices")

            if not df_month.empty:
                recent = df_month.tail(6)
                lines.append("\nMonthly Revenue (last 6):")
                for _, row in recent.iterrows():
                    lines.append(f"  {row['month']}: £{float(row['revenue']):,.2f} | {int(row['invoices'])} invoices")

            return "\n".join(lines)
        except Exception as e:
            return f"Error analyzing retail: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# 12. Root Cause Aggregator
# ─────────────────────────────────────────────────────────────────────────────
def investigate_root_causes(topic: str = "general") -> str:
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

    if topic in ("general", "retail", "uk"):
        sections.append(analyze_online_retail_overview())

    sections.append(analyze_order_cancellation())

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
# Tool Registry
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
    "analyze_market_basket": {
        "fn": analyze_market_basket,
        "description": "Cross-selling patterns and category correlations",
        "topics": ["cross-sell", "market basket", "recommendations"],
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
    "analyze_online_retail_overview": {
        "fn": analyze_online_retail_overview,
        "description": "Online Retail UK dataset overview — customers, revenue, countries",
        "topics": ["retail", "uk", "online retail", "england"],
    },
    "investigate_root_causes": {
        "fn": investigate_root_causes,
        "description": "Multi-dimensional root cause investigation",
        "topics": ["why", "root cause", "decline", "problem", "issue"],
    },
}