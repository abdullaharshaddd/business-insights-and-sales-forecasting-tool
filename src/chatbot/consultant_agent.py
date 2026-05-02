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
from langgraph.checkpoint.sqlite import SqliteSaver

from src.chatbot.business_toolkit import (
    get_model_registry,
    get_kpi_definition,
    get_sales_forecast_summary,
    get_forecast_metrics,
    get_churn_risk_by_segment,
    get_churn_error_analysis,
    search_business_knowledge,
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
# 3. SQL Validation Middleware
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

# ─────────────────────────────────────────────────────────────────────────────
# 4. Deterministic Graph State
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # LangGraph standard message appending
    messages: Annotated[list, operator.add]
    
    # Custom deterministic state tracking
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
    prompt = f"""Classify the intent of the following user query into EXACTLY ONE of these categories:
- FORECAST (asking about future sales, revenue predictions)
- CHURN (asking about customer retention, risk, or segment performance)
- KPI (asking for the definition or formula of a metric)
- SQL (asking for live data, counts, or sums requiring a database query)
- OTHER (greetings, completely irrelevant topics)

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

    valid_intents = ["FORECAST", "CHURN", "KPI", "SQL", "OTHER"]
    if intent not in valid_intents:
        intent = "SQL" # Default fallback
        
    print(f"\n[Router] Intent classified as: {intent}")
    return {"intent": intent}


async def forecast_node(state: AgentState):
    print("[Worker] Executing Prophet Model Pipeline...")
    data = await get_sales_forecast_summary(30)
    return {"raw_data": data}


async def churn_node(state: AgentState):
    print("[Worker] Executing DNN Churn Model Pipeline...")
    data = await get_churn_risk_by_segment()
    return {"raw_data": data}


async def kpi_node(state: AgentState):
    print("[Worker] Executing Semantic Knowledge Retrieval...")
    data = await search_business_knowledge(state["user_input"], n_results=2)
    return {"raw_data": data}


async def other_node(state: AgentState):
    print("[Worker] Handling non-business input...")
    msg = "I am an enterprise business intelligence assistant. Please ask me about revenue, orders, customers, forecasts, or churn."
    return {"raw_data": msg, "polisher_output": msg}


async def sql_draft_node(state: AgentState):
    print("[Worker] Drafting SQL Query...")
    # 1. Fetch relevant business rules via RAG
    context = await search_business_knowledge(state["user_input"], n_results=2)
    schema = db.get_table_info()
    
    # 2. Append error feedback if this is a retry
    error_feedback = ""
    if state.get("raw_data") and "Error" in state["raw_data"]:
         error_feedback = f"YOUR PREVIOUS QUERY FAILED WITH THIS ERROR:\n{state['raw_data']}\nFix the query and try again."

    prompt = f"""Write a SQLite SELECT query to answer the following question.
Question: {state["user_input"]}

Database Schema:
{schema}

Business Rules (MUST FOLLOW):
{context}

{error_feedback}

Respond with ONLY the raw SQL query. No markdown, no explanation, no backticks."""

    res = groq_client.chat.completions.create(
        model=REACT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    sql = res.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return {"sql_query": sql}


async def sql_execute_node(state: AgentState):
    print("[Worker] Validating and Executing SQL...")
    sql = state["sql_query"]
    current_errors = state.get("sql_errors", 0)
    
    # 1. Validator Middleware
    for rule in _SQL_RULES:
        if rule["check"](sql):
            return {
                "raw_data": f"Validator Error: {rule['error']}", 
                "sql_errors": current_errors + 1
            }
            
    # 2. Execute
    try:
        data = db.run(sql)
        return {
            "raw_data": f"Executed SQL: {sql}\nResults: {data}",
            "sql_errors": 0 # reset on success
        }
    except Exception as e:
        return {
            "raw_data": f"SQL Execution Error: {str(e)}", 
            "sql_errors": current_errors + 1
        }


async def polish_node(state: AgentState):
    print("[Polisher] Formatting final response...")
    if state.get("polisher_output"):
        return {} # Already handled (e.g. by OTHER node)
        
    messages = [
        {
            "role": "system",
            "content": (
                "You are an executive business formatter.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Only use explicitly retrieved facts from the provided data.\n"
                "2. If information is missing or incomplete, say so explicitly.\n"
                "3. Do not infer trends, causes, or interpretations unless directly supported by the data.\n"
                "4. NEVER show raw SQL, column headers, DataFrame text, or tool names.\n"
            ),
        }
    ]
    
    # Add minimal chat history context if we had a full state
    for msg in state.get("messages", [])[-6:]:
        if isinstance(msg, HumanMessage):
             messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
             messages.append({"role": "assistant", "content": msg.content})
             
    messages.append({
        "role": "user",
        "content": f'Current Question: "{state["user_input"]}"\n\nData retrieved:\n{state["raw_data"]}\n\nFormat this data clearly and professionally.'
    })
    
    res = groq_client.chat.completions.create(model=POLISHER_MODEL, messages=messages)
    return {"polisher_output": res.choices[0].message.content.strip()}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Graph Routing Logic
# ─────────────────────────────────────────────────────────────────────────────
def route_intent(state: AgentState) -> str:
    return state["intent"]

def route_sql_retry(state: AgentState) -> str:
    # If there's an error and we haven't retried 3 times, draft again
    if state.get("sql_errors", 0) > 0 and state.get("sql_errors", 0) < 3:
        print(f"[Router] SQL Error detected. Retrying... (Attempt {state.get('sql_errors')})")
        return "sql_draft_node"
    return "polish_node"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Compile Graph
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

builder.add_conditional_edges(
    "router",
    route_intent,
    {
        "FORECAST": "forecast_node",
        "CHURN": "churn_node",
        "KPI": "kpi_node",
        "SQL": "sql_draft_node",
        "OTHER": "other_node"
    }
)

# Deterministic convergence to Polish node
builder.add_edge("forecast_node", "polish_node")
builder.add_edge("churn_node", "polish_node")
builder.add_edge("kpi_node", "polish_node")
builder.add_edge("other_node", END) # Skip polish for invalid queries

# SQL Subgraph Loop
builder.add_edge("sql_draft_node", "sql_execute_node")
builder.add_conditional_edges(
    "sql_execute_node",
    route_sql_retry,
    {
        "sql_draft_node": "sql_draft_node",
        "polish_node": "polish_node"
    }
)

builder.add_edge("polish_node", END)

# Configure persistent Sqlite Checkpointer
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("data/checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
compiled_graph = builder.compile(checkpointer=memory)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Orchestrator Interface
# ─────────────────────────────────────────────────────────────────────────────
async def consult_logic_advanced(user_input: str, thread_id: str = "session_1") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initialize state
    initial_state = {
        "user_input": user_input,
        "messages": [HumanMessage(content=user_input)],
        "sql_errors": 0,
        "raw_data": "",
        "polisher_output": ""
    }
    
    try:
        result = await compiled_graph.ainvoke(initial_state, config)
        final_output = result["polisher_output"]
        
        # We must manually save the AI's response to the memory stream so the next 
        # turn's state.messages contains it.
        # However, LangGraph's checkpointer automatically saves the state dictionary.
        # But our `messages` reducer requires us to append the AIMessage.
        # We can update the state explicitly:
        compiled_graph.update_state(
            config,
            {"messages": [AIMessage(content=final_output)]}
        )
        
        return final_output
    except Exception as e:
        return f"System Error: {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# 9. CLI Main Loop
# ─────────────────────────────────────────────────────────────────────────────
async def main_loop():
    print("=" * 65)
    print("BISFT: DETERMINISTIC ENTERPRISE RAG — LIVE")
    print("Architecture: Strict StateGraph Routing | SQL Middleware Validation")
    print("=" * 65)

    while True:
        try:
            user_input = input("\n[Owner]: ")
        except EOFError:
            break
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            continue
            
        response = await consult_logic_advanced(user_input, thread_id="cli_session")
        print(f"\n[AI Consultant]: {response}")

if __name__ == "__main__":
    asyncio.run(main_loop())