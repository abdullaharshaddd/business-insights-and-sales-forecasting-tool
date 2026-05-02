import os
import asyncio
import json
import re
import sqlite3
from typing import List, TypedDict, Annotated
import operator

import yaml
from dotenv import load_dotenv
from groq import Groq
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.chatbot.business_toolkit import (
    get_model_registry,
    get_kpi_definition,
    get_sales_forecast_summary,
    get_forecast_metrics,
    get_churn_risk_by_segment,
    get_churn_error_analysis,
    search_business_knowledge,
    execute_deterministic_kpi,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Config & Clients
# ─────────────────────────────────────────────────────────────────────────────
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        with open(".env", "r", encoding="utf-8-sig") as f:
            for line in f:
                if line.startswith("GROQ_API_KEY"):
                    os.environ["GROQ_API_KEY"] = line.split("=")[1].strip()
                    GROQ_API_KEY = os.environ["GROQ_API_KEY"]
    except Exception:
        pass

if not GROQ_API_KEY:
    raise ValueError("CRITICAL: GROQ_API_KEY not found in environment. Please add it to your .env file.")

GUARD_MODEL    = "llama-3.1-8b-instant"
REACT_MODEL    = "llama-3.3-70b-versatile"
POLISHER_MODEL = "llama-3.3-70b-versatile"

groq_client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Database
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = cfg["paths"]["olist_db"]
safe_path = DB_PATH.replace("\\", "/")
safe_db_uri = f"sqlite:///file:{safe_path}?mode=ro&uri=true"

db = SQLDatabase.from_uri(
    safe_db_uri,
    sample_rows_in_table_info=3,
    include_tables=[
        "orders", "order_items", "order_payments", "order_reviews",
        "customers", "sellers", "products",
        "product_category_name_translation", "geolocation",
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Deterministic Tools & Rules
# ─────────────────────────────────────────────────────────────────────────────
_SQL_RULES = [
    {
        "check": lambda q: bool(re.search(r"SUM\s*\(\s*[\w\.]*price[\w\.]*\s*\)", q, re.I))
                           and "freight_value" not in q.lower(),
        "error": "BUSINESS RULE VIOLATION: Revenue must use SUM(price + freight_value)."
    },
    {
        "check": lambda q: bool(re.search(
            r"\b(DELETE|UPDATE|INSERT|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE)\b", q, re.I
        )),
        "error": "SECURITY BLOCK: Only SELECT queries are permitted."
    },
]

SYSTEM_PROMPT = """You are an enterprise business intelligence assistant.

DETERMINISTIC KPI LAYER:
- Use execute_deterministic_kpi for ALL core business metrics (Revenue, AOV, etc.).
- NEVER write SQL for these. This ensures consistency with executive dashboards.

EXPLORATORY SQL LAYER:
- Use safe_sql_query for ad-hoc questions not in the KPI registry.
- ALWAYS use tool_search_business_knowledge first to check for related rules.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 4. Deterministic Graph State
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_input: str
    intent: str
    raw_data: str
    sql_query: str
    sql_errors: int
    polisher_output: str

# ─────────────────────────────────────────────────────────────────────────────
# 5. Nodes (Deterministic Pipeline Steps)
# ─────────────────────────────────────────────────────────────────────────────

async def router_node(state: AgentState):
    """Strictly classifies the user query into predefined paths."""
    user_input = state["user_input"]
    prompt = f"""Classify intent into EXACTLY ONE:
- FORECAST (future sales predictions)
- CHURN (customer retention risk)
- KPI (A specific metric like revenue, AOV, or delivery rate)
- SQL (Custom ad-hoc data query)
- OTHER (greetings, off-topic)

Query: "{user_input}"
Respond with ONLY the category name."""

    try:
        res = groq_client.chat.completions.create(
            model=GUARD_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        intent = res.choices[0].message.content.strip().upper()
    except Exception:
        intent = "SQL"

    if intent not in ["FORECAST", "CHURN", "KPI", "SQL", "OTHER"]:
        intent = "SQL"
        
    print(f"\n[Router] Intent: {intent}")
    return {"intent": intent}


async def forecast_node(state: AgentState):
    print("[Worker] Executing Forecast...")
    data = await get_sales_forecast_summary(30)
    return {"raw_data": data}


async def churn_node(state: AgentState):
    print("[Worker] Executing Churn Analysis...")
    data = await get_churn_risk_by_segment()
    return {"raw_data": data}


async def kpi_node(state: AgentState):
    print("[Worker] Executing Deterministic KPI Tool...")
    user_input = state["user_input"].lower()
    from src.chatbot.business_toolkit import kpi_engine
    
    kpis_list = [f"'{kid}': {info['label']} - {info['description']}" for kid, info in kpi_engine.definitions.items()]
    kpis_text = "\n".join(kpis_list)
    
    prompt = f"""
Analyze the user's query and extract the target KPI ID and any relevant filters.
Available KPIs:
{kpis_text}

User Query: "{state['user_input']}"

Respond STRICTLY in JSON format with two keys:
- "kpi_id": The exact string ID of the KPI from the list above, or null if none match.
- "filters": A dictionary of filters (e.g. {{"customer_state": "SP", "order_status": "delivered"}}), or null if no filters. Keep filters simple and matching Olist schema.

Only output valid JSON.
"""
    try:
        res = groq_client.chat.completions.create(
            model=GUARD_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        extraction = json.loads(res.choices[0].message.content)
        target_kpi = extraction.get("kpi_id")
        filters = extraction.get("filters")
    except Exception as e:
        print(f"[kpi_node] Error extracting KPI: {e}")
        target_kpi = None
        filters = None

    if target_kpi and target_kpi in kpi_engine.definitions:
        print(f"[kpi_node] Extracted KPI: {target_kpi}, Filters: {filters}")
        val = await execute_deterministic_kpi.ainvoke({"kpi_id": target_kpi, "filters": filters})
        return {"raw_data": val}
    
    # Fallback to general knowledge search if no direct match
    from src.chatbot.business_toolkit import search_business_knowledge
    data = await search_business_knowledge(state["user_input"], n_results=2)
    return {"raw_data": data}


async def other_node(state: AgentState):
    msg = "I am an enterprise business intelligence assistant. How can I help with your data today?"
    return {"raw_data": msg, "polisher_output": msg}


async def sql_draft_node(state: AgentState):
    print("[Worker] Drafting SQL...")
    context = await search_business_knowledge(state["user_input"], n_results=2)
    schema = db.get_table_info()
    error_feedback = f"PREVIOUS ERROR: {state['raw_data']}" if "Error" in state.get("raw_data", "") else ""

    prompt = f"""Write SQLite SELECT for: {state["user_input"]}
Schema: {schema}
Rules: {context}
{error_feedback}
Respond with ONLY raw SQL."""

    res = groq_client.chat.completions.create(
        model=REACT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    sql = res.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
    return {"sql_query": sql}


async def sql_execute_node(state: AgentState):
    print("[Worker] Validating & Executing...")
    sql = state["sql_query"]
    for rule in _SQL_RULES:
        if rule["check"](sql):
            return {"raw_data": f"Validator Error: {rule['error']}", "sql_errors": state.get("sql_errors", 0) + 1}
    try:
        data = db.run(sql)
        return {"raw_data": f"Results: {data}", "sql_errors": 0}
    except Exception as e:
        return {"raw_data": f"SQL Error: {str(e)}", "sql_errors": state.get("sql_errors", 0) + 1}


async def polish_node(state: AgentState):
    if state.get("polisher_output"): return {}
    messages = [{"role": "system", "content": "You are an executive formatter. Use only retrieved facts. No SQL. No hallucination."}]
    messages.append({"role": "user", "content": f'Q: "{state["user_input"]}"\nData: {state["raw_data"]}'})
    res = groq_client.chat.completions.create(model=POLISHER_MODEL, messages=messages)
    return {"polisher_output": res.choices[0].message.content.strip()}

# ─────────────────────────────────────────────────────────────────────────────
# 6. Build Graph
# ─────────────────────────────────────────────────────────────────────────────
builder = StateGraph(AgentState)
builder.add_node("router", router_node)
builder.add_node("forecast_node", forecast_node)
builder.add_node("churn_node", churn_node)
builder.add_node("kpi_node", kpi_node)
builder.add_node("other_node", other_node)
builder.add_node("sql_draft_node", sql_draft_node)
builder.add_node("sql_execute_node", sql_execute_node)
builder.add_node("polish_node", polish_node)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", lambda s: s["intent"], {
    "FORECAST": "forecast_node", "CHURN": "churn_node", "KPI": "kpi_node", "SQL": "sql_draft_node", "OTHER": "other_node"
})
builder.add_edge("forecast_node", "polish_node")
builder.add_edge("churn_node", "polish_node")
builder.add_edge("kpi_node", "polish_node")
builder.add_edge("other_node", END)
builder.add_edge("sql_draft_node", "sql_execute_node")
builder.add_conditional_edges("sql_execute_node", lambda s: "sql_draft_node" if 0 < s.get("sql_errors", 0) < 3 else "polish_node", {
    "sql_draft_node": "sql_draft_node", "polish_node": "polish_node"
})
builder.add_edge("polish_node", END)

memory = MemorySaver()
compiled_graph = builder.compile(checkpointer=memory)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Interface
# ─────────────────────────────────────────────────────────────────────────────
async def consult_logic_advanced(user_input: str, thread_id: str = "session_1") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"user_input": user_input, "messages": [HumanMessage(content=user_input)], "sql_errors": 0, "raw_data": "", "polisher_output": ""}
    try:
        result = await compiled_graph.ainvoke(initial_state, config)
        return result["polisher_output"]
    except Exception as e:
        return f"System Error: {str(e)}"

if __name__ == "__main__":
    async def main():
        print("BISFT DETERMINISTIC V2 LIVE")
        while True:
            u = input("\n[Owner]: ")
            if u.lower() in ["exit", "quit"]: break
            print(f"\n[AI]: {await consult_logic_advanced(u)}")
    asyncio.run(main())