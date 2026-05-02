"""
Agentic RAG Business Intelligence System
==========================================
Multi-agent LangGraph orchestrator with:
  - Planner Agent (decomposes complex questions into steps)
  - Data Gatherer (executes analytical tools, KPIs, forecasts, churn)
  - RAG Retriever (semantic search over business knowledge)
  - Strategy Synthesizer (produces consultant-grade analysis)
  - Long-term Memory (persists findings across sessions)
"""

import os
import asyncio
import json
import re
from typing import TypedDict, Annotated
import operator

import yaml
from dotenv import load_dotenv
from groq import Groq
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.chatbot.business_toolkit import (
    get_sales_forecast_summary,
    get_forecast_metrics,
    get_churn_risk_by_segment,
    get_churn_error_analysis,
    search_business_knowledge,
    execute_deterministic_kpi,
    kpi_engine,
)
from src.analytics.analytical_tools import ANALYTICAL_TOOLS
from src.chatbot.memory_store import MemoryStore

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
    raise ValueError("CRITICAL: GROQ_API_KEY not found. Add it to .env")

GUARD_MODEL    = "llama-3.1-8b-instant"
REACT_MODEL    = "llama-3.3-70b-versatile"
POLISHER_MODEL = "llama-3.3-70b-versatile"

groq_client = Groq(api_key=GROQ_API_KEY)
memory_store = MemoryStore()

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
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. SQL Safety Rules
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
# 4. Graph State
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_input: str
    thread_id: str
    # Router
    intent: str  # SIMPLE | ANALYTICAL | OTHER
    # Planner
    plan: list
    # Data gathering
    gathered_data: str
    # RAG
    rag_context: str
    # Memory
    memory_context: str
    # SQL fallback
    sql_query: str
    sql_errors: int
    # Output
    final_output: str


# ═════════════════════════════════════════════════════════════════════════════
# 5. AGENT NODES
# ═════════════════════════════════════════════════════════════════════════════

# ── 5a. Router ───────────────────────────────────────────────────────────────
async def router_node(state: AgentState):
    """Classifies: SIMPLE (single-tool), ANALYTICAL (multi-step), or OTHER."""
    user_input = state["user_input"]
    history = state.get("messages", [])
    history_lines = []
    for m in history[-5:-1]:
        role = "User" if getattr(m, "type", "") == "human" else "AI"
        history_lines.append(f"{role}: {m.content[:200]}")
    history_str = "\n".join(history_lines) if history_lines else "None"

    prompt = f"""Classify the user's question into EXACTLY ONE category:

- ANALYTICAL: Complex questions needing multi-step investigation. Examples:
  "Why is revenue declining?", "How can we improve retention?",
  "What are the main business problems?", "Analyze our performance",
  "What should we do about churn?", questions with "why", "how to improve", "root cause", "recommend"

- SIMPLE: Direct single-metric questions. Examples:
  "What is our revenue?", "Show me the forecast for 30 days",
  "What is the AOV?", "Show churn rates", "What is the delivery rate?"

- OTHER: Greetings, off-topic, or non-business questions.
  "Hi", "Hello", "What's the weather?"

Recent Conversation:
{history_str}

Query: "{user_input}"
Respond with ONLY: ANALYTICAL or SIMPLE or OTHER"""

    try:
        res = groq_client.chat.completions.create(
            model=GUARD_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        intent = res.choices[0].message.content.strip().upper()
    except Exception:
        intent = "SIMPLE"

    if intent not in ("ANALYTICAL", "SIMPLE", "OTHER"):
        intent = "SIMPLE"

    print(f"\n[Router] Intent: {intent}")
    return {"intent": intent}


# ── 5b. Greeting Node ────────────────────────────────────────────────────────
async def greeting_node(state: AgentState):
    msg = (
        "Hello! I'm your AI Business Intelligence Consultant. "
        "I can analyze revenue trends, investigate churn, forecast sales, "
        "and provide strategic recommendations. What would you like to explore?"
    )
    return {"final_output": msg, "messages": [AIMessage(content=msg)]}


# ── 5c. Planner Node (CORE OF AGENTIC BEHAVIOR) ─────────────────────────────
async def planner_node(state: AgentState):
    """Decomposes complex questions into a multi-step investigation plan."""
    user_input = state["user_input"]

    available_tools = []
    for tid, info in ANALYTICAL_TOOLS.items():
        available_tools.append(f"  - {tid}: {info['description']}")

    kpi_list = [f"  - kpi:{kid}: {info['label']}" for kid, info in kpi_engine.definitions.items()]

    special_tools = [
        "  - forecast: Sales forecast from Prophet model",
        "  - churn_segments: Churn risk by customer segment",
        "  - churn_errors: Churn model error analysis",
        "  - rag_search: Search business knowledge base",
        "  - sql_query: Custom SQL for ad-hoc data",
    ]

    tools_text = "\n".join(available_tools + kpi_list + special_tools)

    prompt = f"""You are a business intelligence planner. Create an investigation plan to answer:

"{user_input}"

Available tools:
{tools_text}

Create a JSON array of steps. Each step has:
- "tool": the tool name from above
- "purpose": why this step is needed (1 sentence)

Rules:
- Use 3-6 steps for thorough analysis
- Always include rag_search for strategy/knowledge retrieval
- Start with data gathering, end with root cause or comparative analysis
- For "why" questions, include multiple analytical perspectives

Output ONLY a valid JSON array."""

    try:
        res = groq_client.chat.completions.create(
            model=REACT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = res.choices[0].message.content.strip()
        parsed = json.loads(content)
        # Handle both {"steps": [...]} and direct [...]
        if isinstance(parsed, dict):
            plan = parsed.get("steps", parsed.get("plan", list(parsed.values())[0]))
        else:
            plan = parsed
    except Exception as e:
        print(f"[Planner] Error: {e}")
        plan = [
            {"tool": "analyze_revenue_trends", "purpose": "Check revenue health"},
            {"tool": "analyze_customer_behavior", "purpose": "Check customer patterns"},
            {"tool": "rag_search", "purpose": "Find relevant strategies"},
        ]

    print(f"[Planner] Created {len(plan)} step plan:")
    for i, step in enumerate(plan, 1):
        print(f"  Step {i}: {step.get('tool', '?')} — {step.get('purpose', '')}")

    return {"plan": plan}


# ── 5d. Memory Recall Node ──────────────────────────────────────────────────
async def memory_node(state: AgentState):
    """Retrieve relevant past findings and conversation context."""
    user_input = state["user_input"]
    thread_id = state.get("thread_id", "default")

    # Extract topic keywords for memory search
    keywords = [w for w in user_input.lower().split()
                if w not in ("what", "why", "how", "is", "the", "our", "can", "we", "do", "to", "a", "an", "and", "or")]
    topic = " ".join(keywords[:3]) if keywords else "general"

    findings = memory_store.recall_findings(topic=topic, limit=5)
    conv_ctx = memory_store.recall_conversation_context(thread_id, limit=3)

    memory_ctx = ""
    if findings and "No previous" not in findings:
        memory_ctx += findings + "\n\n"
    if conv_ctx:
        memory_ctx += conv_ctx

    return {"memory_context": memory_ctx if memory_ctx else "No prior context."}


# ── 5e. Data Gatherer Node (Executes the Plan) ──────────────────────────────
async def data_gatherer_node(state: AgentState):
    """Execute all planned steps and aggregate results."""
    plan = state.get("plan", [])
    user_input = state["user_input"]
    results = []

    for i, step in enumerate(plan, 1):
        tool_name = step.get("tool", "")
        purpose = step.get("purpose", "")
        print(f"  [Gatherer] Step {i}/{len(plan)}: {tool_name}")

        try:
            if tool_name in ANALYTICAL_TOOLS:
                # Analytical tools
                fn = ANALYTICAL_TOOLS[tool_name]["fn"]
                if tool_name == "investigate_root_causes":
                    # Determine topic from user query
                    topic = "general"
                    for t in ("revenue", "delivery", "customer", "retention", "churn"):
                        if t in user_input.lower():
                            topic = t
                            break
                    result = fn(topic)
                else:
                    result = fn()
                results.append(f"[Step {i}: {tool_name}]\n{result}")

            elif tool_name.startswith("kpi:"):
                # Deterministic KPI
                kpi_id = tool_name.replace("kpi:", "")
                val = await execute_deterministic_kpi.ainvoke({"kpi_id": kpi_id})
                results.append(f"[Step {i}: KPI {kpi_id}]\n{val}")

            elif tool_name == "forecast":
                result = await get_sales_forecast_summary(30)
                results.append(f"[Step {i}: Sales Forecast]\n{result}")

            elif tool_name == "churn_segments":
                result = await get_churn_risk_by_segment()
                results.append(f"[Step {i}: Churn Segments]\n{result}")

            elif tool_name == "churn_errors":
                result = await get_churn_error_analysis()
                results.append(f"[Step {i}: Churn Error Analysis]\n{result}")

            elif tool_name == "rag_search":
                result = await search_business_knowledge(user_input, n_results=4)
                results.append(f"[Step {i}: Knowledge Retrieval]\n{result}")

            elif tool_name == "sql_query":
                # Let the LLM draft a SQL query for this specific purpose
                schema = db.get_table_info()
                sql_prompt = f"Write a SQLite SELECT query for: {purpose}\nSchema: {schema}\nRespond with ONLY raw SQL."
                res = groq_client.chat.completions.create(
                    model=REACT_MODEL,
                    messages=[{"role": "user", "content": sql_prompt}],
                    temperature=0,
                )
                sql = res.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()
                # Validate
                blocked = False
                for rule in _SQL_RULES:
                    if rule["check"](sql):
                        results.append(f"[Step {i}: SQL blocked] {rule['error']}")
                        blocked = True
                        break
                if not blocked:
                    data = db.run(sql)
                    results.append(f"[Step {i}: SQL Query]\n{data}")
            else:
                results.append(f"[Step {i}: {tool_name}] Tool not found, skipped.")

        except Exception as e:
            results.append(f"[Step {i}: {tool_name}] Error: {str(e)}")

    gathered = "\n\n---\n\n".join(results)
    return {"gathered_data": gathered}


# ── 5f. RAG Retrieval Node ───────────────────────────────────────────────────
async def rag_node(state: AgentState):
    """Retrieve business knowledge and strategy docs via vector search."""
    result = await search_business_knowledge(state["user_input"], n_results=4)
    return {"rag_context": result}


# ── 5g. Simple Executor (for direct single-tool questions) ───────────────────
async def simple_executor_node(state: AgentState):
    """Handle simple single-tool questions: KPI, forecast, churn, or SQL."""
    user_input = state["user_input"].lower()

    # 1. Check for forecast intent
    if any(w in user_input for w in ("forecast", "predict", "future", "next days", "projection")):
        days = 30
        for word in user_input.split():
            if word.isdigit():
                days = int(word)
                break
        data = await get_sales_forecast_summary(days)
        return {"gathered_data": data}

    # 2. Check for churn intent
    if any(w in user_input for w in ("churn", "retention", "attrition")):
        data = await get_churn_risk_by_segment()
        return {"gathered_data": data}

    # 3. Check for KPI match
    kpis_list = [f"'{kid}': {info['label']} - {info['description']}"
                 for kid, info in kpi_engine.definitions.items()]
    kpis_text = "\n".join(kpis_list)

    prompt = f"""Extract the KPI ID from the user's query.
Available KPIs:
{kpis_text}

Query: "{state['user_input']}"

Respond in JSON: {{"kpi_id": "the_id_or_null", "filters": null}}"""

    try:
        res = groq_client.chat.completions.create(
            model=GUARD_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        extraction = json.loads(res.choices[0].message.content)
        target_kpi = extraction.get("kpi_id")
        filters = extraction.get("filters")

        if target_kpi and target_kpi in kpi_engine.definitions:
            val = await execute_deterministic_kpi.ainvoke({"kpi_id": target_kpi, "filters": filters})
            return {"gathered_data": val}
    except Exception:
        pass

    # 4. Fallback: SQL
    schema = db.get_table_info()
    context = await search_business_knowledge(state["user_input"], n_results=2)
    sql_prompt = f"""Write SQLite SELECT for: {state["user_input"]}
Schema: {schema}
Rules: {context}
Respond with ONLY raw SQL."""

    res = groq_client.chat.completions.create(
        model=REACT_MODEL,
        messages=[{"role": "user", "content": sql_prompt}],
        temperature=0,
    )
    sql = res.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()

    for rule in _SQL_RULES:
        if rule["check"](sql):
            return {"gathered_data": f"Query blocked: {rule['error']}"}

    try:
        data = db.run(sql)
        return {"gathered_data": f"Results: {data}"}
    except Exception as e:
        return {"gathered_data": f"SQL Error: {str(e)}"}


# ── 5h. Strategy Synthesizer (The Brain) ────────────────────────────────────
async def synthesizer_node(state: AgentState):
    """Produce consultant-grade analysis from all gathered evidence."""
    is_analytical = state.get("intent") == "ANALYTICAL"

    sys_prompt = """You are a Senior Business Intelligence Consultant at a top-tier consulting firm.

ROLE: Synthesize data from multiple analytical tools into a cohesive, actionable business briefing.

RULES:
- Lead with the KEY INSIGHT (1-2 sentences capturing the most important finding)
- Present ROOT CAUSES backed by data (not speculation)
- Provide SPECIFIC, ACTIONABLE RECOMMENDATIONS (not generic advice)
- Use exact numbers from the data provided
- Structure your response with clear headers
- If past findings are available, reference them for continuity
- Keep total response under 400 words
- Never show SQL or technical internals
- If the question is simple, give a concise answer with context"""

    messages = [{"role": "system", "content": sys_prompt}]

    # Add conversation history for continuity
    for m in state.get("messages", [])[:-1]:
        role = "user" if getattr(m, "type", "") == "human" else "assistant"
        messages.append({"role": role, "content": m.content[:300]})

    # Build the evidence package
    evidence_parts = []

    if state.get("gathered_data"):
        evidence_parts.append(f"ANALYTICAL DATA:\n{state['gathered_data']}")

    if is_analytical:
        if state.get("rag_context") and "No relevant" not in state.get("rag_context", ""):
            evidence_parts.append(f"BUSINESS KNOWLEDGE:\n{state['rag_context']}")

        if state.get("memory_context") and "No prior" not in state.get("memory_context", ""):
            evidence_parts.append(f"PAST FINDINGS:\n{state['memory_context']}")

    evidence = "\n\n---\n\n".join(evidence_parts) if evidence_parts else "No data gathered."

    messages.append({
        "role": "user",
        "content": f'Question: "{state["user_input"]}"\n\nEvidence:\n{evidence}',
    })

    res = groq_client.chat.completions.create(model=POLISHER_MODEL, messages=messages)
    content = res.choices[0].message.content.strip()

    # Store key findings in long-term memory
    if is_analytical:
        keywords = [w for w in state["user_input"].lower().split()
                    if len(w) > 3 and w not in ("what", "about", "should", "would", "could")]
        topic = " ".join(keywords[:3]) if keywords else "analysis"
        # Store a condensed finding
        finding_summary = content[:200] + "..." if len(content) > 200 else content
        memory_store.store_finding(topic=topic, finding=finding_summary, importance="high")

    return {"final_output": content, "messages": [AIMessage(content=content)]}


# ═════════════════════════════════════════════════════════════════════════════
# 6. BUILD LANGGRAPH
# ═════════════════════════════════════════════════════════════════════════════
builder = StateGraph(AgentState)

# Register nodes
builder.add_node("router", router_node)
builder.add_node("greeting", greeting_node)
builder.add_node("planner", planner_node)
builder.add_node("memory", memory_node)
builder.add_node("data_gatherer", data_gatherer_node)
builder.add_node("rag_retriever", rag_node)
builder.add_node("simple_executor", simple_executor_node)
builder.add_node("synthesizer", synthesizer_node)

# Edges
builder.add_edge(START, "router")

# Router branches
builder.add_conditional_edges("router", lambda s: s["intent"], {
    "ANALYTICAL": "planner",
    "SIMPLE": "simple_executor",
    "OTHER": "greeting",
})

# Analytical pipeline: planner → parallel(memory, data_gatherer, rag) → synthesizer
builder.add_edge("planner", "memory")
builder.add_edge("memory", "data_gatherer")
builder.add_edge("data_gatherer", "rag_retriever")
builder.add_edge("rag_retriever", "synthesizer")

# Simple pipeline: executor → synthesizer
builder.add_edge("simple_executor", "synthesizer")

# Terminals
builder.add_edge("greeting", END)
builder.add_edge("synthesizer", END)

# Compile with memory checkpointer
memory_checkpointer = MemorySaver()
compiled_graph = builder.compile(checkpointer=memory_checkpointer)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Interface
# ═════════════════════════════════════════════════════════════════════════════
async def consult_logic_advanced(user_input: str, thread_id: str = "session_1") -> str:
    """Main entry point for the agentic BI system."""
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "user_input": user_input,
        "thread_id": thread_id,
        "messages": [HumanMessage(content=user_input)],
        "sql_errors": 0,
        "gathered_data": "",
        "rag_context": "",
        "memory_context": "",
        "final_output": "",
        "plan": [],
    }
    try:
        result = await compiled_graph.ainvoke(initial_state, config)
        return result.get("final_output", "No response generated.")
    except Exception as e:
        return f"System Error: {str(e)}"


if __name__ == "__main__":
    async def main():
        print("=" * 60)
        print("  BISFT AGENTIC RAG v3 — Business Intelligence Consultant")
        print("=" * 60)
        while True:
            u = input("\n[You]: ")
            if u.lower() in ("exit", "quit"):
                break
            print("\n[Thinking...]")
            response = await consult_logic_advanced(u)
            print(f"\n[AI Consultant]:\n{response}")

    asyncio.run(main())