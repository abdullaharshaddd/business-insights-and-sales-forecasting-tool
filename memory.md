# BISFT — Project Memory
### Business Insights & Sales Forecasting Tool
**Last Updated:** 2026-05-08  
**Contributors:** Abdullah Arshad (23L-2531), Sohaib Haider (23L-2519)  
**Institution:** FAST-NUCES (Academic Project)  
**Repo:** https://github.com/abdullaharshaddd/business-insights-and-sales-forecasting-tool

---

## 1. PROJECT OVERVIEW

An **AI-driven Decision Intelligence Platform** for predictive analytics, customer segmentation, and intelligent data querying. Built on two datasets: **Online Retail** (UK e-commerce) and **Olist** (Brazilian marketplace).

### Core Modules

| Module | Algorithm | Dataset | Status |
|---|---|---|---|
| Data Cleaning | Pandas pipeline | Both | ✅ Done |
| Churn Prediction | DNN + Baselines (LR, RF, GB, KNN) | Online Retail | ✅ Done |
| Sales Forecasting (A) | Prophet + Holidays | Online Retail | ✅ Done |
| Sales Forecasting (B) | Multivariate LSTM | Olist | ✅ Done (code ready) |
| Agentic RAG Chatbot | LangGraph + Groq + ChromaDB | Olist | ✅ Done |
| Customer Segmentation | RFM + K-Means | Online Retail | ⏳ Pending |

---

## 2. TECHNOLOGY STACK

- **Language:** Python 3.11+
- **ML/DL:** TensorFlow/Keras, Scikit-learn, imbalanced-learn (SMOTE)
- **Forecasting:** Prophet, LSTM (TensorFlow)
- **LLM:** Groq API (Llama 3.1 8B for routing, Llama 3.3 70B for reasoning/synthesis)
- **Orchestration:** LangGraph (multi-agent state graph)
- **Vector DB:** ChromaDB (persistent, `all-MiniLM-L6-v2` embeddings)
- **Database:** SQLite (Olist data in `data/processed/olist/olist.db`)
- **Caching:** diskcache (tool result caching)
- **API:** FastAPI (stub in `main.py`, not fully connected yet)
- **Config:** YAML (`config/config.yaml`) + JSON (KPI defs, model registry, business knowledge)

---

## 3. PROJECT STRUCTURE (ACTUAL)

```
BISFT/
├── .env                          # GROQ_API_KEY
├── main.py                       # FastAPI stub (not yet connected to agent)
├── run_pipeline.py               # Master pipeline runner (cleaning → features → train → eval)
├── requirements.txt              # All dependencies
├── config/
│   ├── config.yaml               # Master config (paths, HP grid, model arch, forecasting params)
│   ├── kpi_definitions.json      # 9 KPI definitions with SQL formulas
│   ├── model_registry.json       # Churn DNN + Prophet metadata, features, eval results
│   └── business_knowledge.json   # 7 RAG documents (strategies, SOPs, guides)
├── data/
│   ├── raw/
│   │   ├── online_retail/        # online_retail.xlsx
│   │   └── olist/                # 9 Olist CSVs
│   ├── processed/
│   │   ├── online_retail_cleaned.csv   # ~36 MB cleaned
│   │   ├── olist_merged_cleaned.csv    # ~63 MB merged & cleaned
│   │   ├── churn_features.csv          # Customer-level RFM features + churn label
│   │   ├── daily_revenue.csv           # Prophet input (ds, y)
│   │   └── olist/olist.db              # SQLite DB (9 tables for chatbot)
│   ├── vector_db/                # ChromaDB persistent store (bisft_knowledge collection)
│   ├── tool_cache/               # diskcache for tool results
│   ├── agent_memory.db           # SQLite long-term memory (findings + conversation summaries)
│   └── checkpoints.sqlite        # LangGraph checkpointer state
├── scripts/
│   └── data_cleaning.py          # Phase 1: Clean both datasets
├── src/
│   ├── features/
│   │   └── churn_features.py     # Phase 2: RFM + engineered features + churn label
│   ├── models/
│   │   ├── churn_model.py        # DNN architecture (128→64→32→1) + baseline builders
│   │   └── lstm_forecaster.py    # Multivariate LSTM (100→50→Dense(30))
│   ├── training/
│   │   └── train_churn.py        # HP grid search + final training + SMOTE + baselines
│   ├── evaluation/
│   │   └── evaluate_churn.py     # Metrics, ROC, PR, confusion matrix, bias check, error analysis
│   ├── forecasting/
│   │   └── prophet_model.py      # Prophet training + cross-validation + forecast generation
│   ├── analytics/
│   │   ├── kpi_engine.py         # Deterministic KPI calculator (single source of truth)
│   │   └── analytical_tools.py   # 11 analytical tools (revenue, delivery, customer, etc.)
│   ├── chatbot/
│   │   ├── consultant_agent.py   # LangGraph multi-agent orchestrator (THE BRAIN)
│   │   ├── business_toolkit.py   # Tool wrappers (forecast, churn, KPI, RAG, SQL)
│   │   ├── ingest_knowledge.py   # ChromaDB ingestion pipeline (KPIs + models + rules + strategies)
│   │   └── memory_store.py       # SQLite long-term memory for findings & conversation context
│   └── utils/
│       ├── preprocess_olist.py   # CSV → SQLite conversion
│       └── preprocess_forecasting.py  # Online Retail → daily_revenue.csv (Prophet format)
├── models/
│   ├── churn/                    # churn_dnn_best.h5, scaler.pkl, baselines, test sets
│   ├── forecasting/              # prophet_model.pkl
│   └── lstm/                     # sales_lstm_model.h5
├── evaluation/
│   ├── churn/                    # bias_check_by_segment.csv, error_profiles.csv, error_analysis_samples.csv
│   └── forecasting/              # summary_metrics.json, metrics.csv, forecast_results.csv
└── reports/
    ├── figures/                  # All generated plots (ROC, PR, confusion matrices, training curves)
    └── tables/                   # hp_tuning_results.csv, baseline_comparison.csv, final_metrics.csv
```

---

## 4. DATA PIPELINE

### 4a. Online Retail Dataset
- **Source:** `data/raw/online_retail/online_retail.xlsx` (UK e-commerce, 2010-2011)
- **Cleaning (`scripts/data_cleaning.py`):** Remove duplicates, drop missing CustomerIDs, remove cancelled invoices (prefix 'C'), filter positive quantity/price, add `totalprice` column
- **Output:** `data/processed/online_retail_cleaned.csv`
- **Used for:** Churn features, Prophet forecasting (daily revenue)

### 4b. Olist Dataset
- **Source:** `data/raw/olist/` — 9 CSVs (Brazilian marketplace, 2016-2018, ~100K orders)
- **Cleaning (`scripts/data_cleaning.py`):** Merge all 9 tables, filter delivered orders, parse dates, add `delivery_days`, translate categories to English
- **Output:** `data/processed/olist_merged_cleaned.csv` (for LSTM)
- **SQLite (`src/utils/preprocess_olist.py`):** All 9 CSVs → `data/processed/olist/olist.db` with tables: `orders`, `order_items`, `order_payments`, `order_reviews`, `customers`, `sellers`, `products`, `product_category_name_translation`, `geolocation`
- **Used for:** Chatbot SQL queries, KPI engine, analytical tools

---

## 5. CHURN PREDICTION MODULE (✅ COMPLETE)

### 5a. Feature Engineering (`src/features/churn_features.py`)
- **Snapshot date:** 2011-12-10 (last date in Online Retail)
- **Churn definition:** No purchase in 30-day window before snapshot → churned=1
- **Training window:** 12 months before snapshot minus churn window
- **Features (11 total):**
  - `recency` — days since last purchase
  - `frequency` — number of unique invoices
  - `monetary` — total revenue (quantity × unitprice)
  - `avg_basket_size` — avg items per order
  - `product_variety` — distinct product categories
  - `avg_unit_price` — average price per item
  - `r_score`, `f_score`, `m_score` — RFM quintile scores (1-5)
  - `rfm_score` — sum of R+F+M scores (3-15)
  - `country_enc` — label-encoded country
- **Segments:** Low-Value, Mid-Value, High-Value, Champions (based on rfm_score quartiles)
- **Output:** `data/processed/churn_features.csv`

### 5b. Model Architecture (`src/models/churn_model.py`)
- **DNN:** Input → Dense(128, ReLU) → BN → Dropout(0.3) → Dense(64, ReLU) → BN → Dropout(0.3) → Dense(32, ReLU) → BN → Dense(1, Sigmoid)
- **Loss:** Binary Cross-Entropy | **Optimizer:** Adam (lr=0.001) | **Metrics:** AUC-ROC, Accuracy, Precision, Recall
- **Baselines:** Logistic Regression, Random Forest, Gradient Boosting, KNN

### 5c. Training (`src/training/train_churn.py`)
- **Split:** 70/15/15 stratified
- **SMOTE:** Applied to training set only
- **HP Grid Search:** 12 random samples from (lr × batch × dropout × epochs × layers)
- **Final HP:** lr=0.001, batch=64, dropout=0.3, epochs=50, layers=[128,64,32]
- **Saved:** `models/churn/churn_dnn_best.h5`, `scaler.pkl`, `feature_cols.json`, baseline `.pkl` files, test sets as `.npy`

### 5d. Evaluation (`src/evaluation/evaluate_churn.py`)
- Full classification report, confusion matrices, ROC/PR curve comparisons
- **Bias Check:** Per-segment AUC/F1 breakdown
  - Champions: 38.4% churn, AUC 0.6583
  - High-Value: 52.9% churn, AUC 0.6561
  - Mid-Value: 76.4% churn, AUC 0.5523
  - Low-Value: 87.2% churn, AUC 0.5594
- **Error Analysis:** FP/FN feature profiles, top misclassified samples
- **Known Weaknesses:** Struggles with Champions (borderline AUC), FNs from high-frequency borderline-recency customers, FPs from recent 1-time buyers

---

## 6. FORECASTING MODULE (✅ COMPLETE)

### 6a. Prophet (`src/forecasting/prophet_model.py`)
- **Input:** `data/processed/daily_revenue.csv` (ds, y columns — daily total revenue from Online Retail)
- **Preprocessing:** `src/utils/preprocess_forecasting.py` aggregates online_retail_cleaned.csv by day
- **Config:** Linear growth, auto seasonality (yearly + weekly), additive mode, changepoint_prior=0.05, UK holidays, 95% CI
- **Cross-Validation:** initial=180 days, period=30 days, horizon=30 days
- **Performance:** RMSE=R$14,819, MAE=R$10,589, MAPE=36%, Coverage=86.7%
- **Saved:** `models/forecasting/prophet_model.pkl`
- **Outputs:** `evaluation/forecasting/summary_metrics.json`, `forecast_results.csv`, `metrics.csv`

### 6b. LSTM (`src/models/lstm_forecaster.py`)
- **Input:** Olist merged data → daily aggregation (total_sales, avg_freight, avg_review, order_volume, day_of_week, month)
- **Architecture:** Input(60, 6) → LSTM(100, return_seq) → Dropout(0.2) → LSTM(50) → Dropout(0.2) → Dense(30)
- **Lookback:** 60 days → **Horizon:** 30 days
- **Status:** Code complete, model can be trained

---

## 7. AGENTIC RAG CHATBOT (✅ COMPLETE — THE MAIN SYSTEM)

### 7a. Architecture (`src/chatbot/consultant_agent.py`)

**Multi-agent LangGraph pipeline** with 8 nodes:

```
START → Router
  ├── OTHER → Greeting Node → END
  ├── SIMPLE → Simple Executor → Synthesizer → END
  └── ANALYTICAL → Planner → Memory → Data Gatherer → RAG Retriever → Synthesizer → END
```

**LLM Models (via Groq API):**
- `GUARD_MODEL` = `llama-3.1-8b-instant` — Used for: Router classification, KPI extraction, greeting responses
- `REACT_MODEL` = `llama-3.3-70b-versatile` — Used for: Planning, SQL generation
- `POLISHER_MODEL` = `llama-3.3-70b-versatile` — Used for: Final synthesis

**Node Details:**

| Node | Purpose |
|---|---|
| `router_node` | Classifies query as SIMPLE / ANALYTICAL / OTHER using 8B model |
| `greeting_node` | Handles greetings, thanks, off-topic with conversational responses |
| `planner_node` | Decomposes complex questions into 3-6 step investigation plans |
| `memory_node` | Recalls past findings + conversation context from SQLite memory |
| `data_gatherer_node` | Executes all planned steps (analytical tools, KPIs, forecasts, churn, SQL) |
| `rag_node` | Semantic search over ChromaDB business knowledge |
| `simple_executor_node` | Handles single-metric queries (forecast → churn → analytical tool → KPI → SQL fallback) |
| `synthesizer_node` | Produces consultant-grade analysis (executive briefing for ANALYTICAL, direct answer for SIMPLE) |

**State Schema (`AgentState`):**
```python
messages: list          # Conversation history (HumanMessage/AIMessage)
user_input: str         # Current user query
thread_id: str          # Session ID for memory
intent: str             # SIMPLE | ANALYTICAL | OTHER
plan: list              # Planner output (tool + purpose steps)
gathered_data: str      # Aggregated tool results
rag_context: str        # Vector search results
memory_context: str     # Past findings
sql_query: str          # Generated SQL (if any)
sql_errors: int         # Error counter
final_output: str       # Final response to user
```

**Entry Point:** `consult_logic_advanced(user_input, thread_id)` → returns final response string

### 7b. Business Toolkit (`src/chatbot/business_toolkit.py`)

All tool functions the agent can invoke:

| Tool | Function | Caching |
|---|---|---|
| `query_database()` | Raw SQL against Olist SQLite | None |
| `get_model_registry()` | Model metadata from JSON | 24h cache |
| `get_kpi_definition()` | KPI definitions + live calculation | 24h cache |
| `get_sales_forecast_summary(days)` | Live Prophet forecast or cached CSV fallback | 4h cache |
| `get_forecast_metrics()` | Prophet CV metrics (RMSE, MAE, MAPE, coverage) | 24h cache |
| `get_churn_risk_by_segment()` | Segment-level churn rates from eval data | 24h cache |
| `get_churn_error_analysis()` | FP/FN feature profiles | 24h cache |
| `search_business_knowledge(query, n)` | ChromaDB semantic search | 24h cache |
| `execute_deterministic_kpi(kpi_id, filters)` | KPI Engine calculation (LangChain `@tool`) | None |

### 7c. KPI Engine (`src/analytics/kpi_engine.py`)

**Central deterministic KPI calculator** — ensures dashboard and chatbot use identical logic.

**Registered KPIs (9):**
| ID | Label | SQL Formula |
|---|---|---|
| `revenue` | Total Revenue (GMV) | `SUM(price + freight_value)` from order_items |
| `aov` | Average Order Value | `SUM(price + freight_value) / COUNT(DISTINCT order_id)` |
| `on_time_delivery_rate` | On-Time Delivery Rate | `COUNT(delivered ≤ estimated) / COUNT(*) * 100` |
| `avg_review_score` | Average Review Score | `AVG(review_score)` |
| `cancellation_rate` | Cancellation Rate | `COUNT(canceled) / COUNT(*) * 100` |
| `freight_ratio` | Freight-to-Price Ratio | `SUM(freight) / SUM(price) * 100` |
| `avg_payment_installments` | Avg Payment Installments | `AVG(payment_installments)` |
| `top_categories_by_revenue` | Top Categories | Full SELECT with JOINs |
| `customer_geographic_concentration` | Geographic Distribution | Full SELECT with GROUP BY state |

Supports optional `filters` dict for WHERE clause injection.

### 7d. Analytical Tools (`src/analytics/analytical_tools.py`)

**11 deterministic analytical functions** registered in `ANALYTICAL_TOOLS` dict:

| Tool ID | What It Does |
|---|---|
| `analyze_revenue_trends` | Monthly revenue with MoM growth rates |
| `analyze_delivery_performance` | On-time rates, avg delivery days, delay trends |
| `analyze_customer_behavior` | Repeat vs one-time buyers, acquisition trends |
| `analyze_review_scores` | Score distribution, low-review rates by month |
| `analyze_category_performance` | Top/bottom categories by revenue + reviews |
| `analyze_seller_performance` | Seller concentration, slow seller identification |
| `analyze_geographic_distribution` | Revenue by state, top-3 concentration |
| `analyze_market_basket` | Category co-purchase patterns |
| `estimate_clv_by_segment` | Customer lifetime value by spend quartile |
| `analyze_order_cancellation` | Cancellation rates and trends |
| `investigate_root_causes` | Multi-dimensional aggregator (runs multiple tools based on topic) |

Each tool queries the Olist SQLite DB in read-only mode and returns formatted text.

### 7e. Knowledge Base (`src/chatbot/ingest_knowledge.py`)

**ChromaDB vector store** at `data/vector_db/` with collection `bisft_knowledge`.

**Document types ingested:**
1. **KPI Definitions** (9 docs) — from `config/kpi_definitions.json`
2. **Model Registry** (2 docs) — from `config/model_registry.json` (Churn DNN + Prophet)
3. **Business Rules** (5 docs) — hardcoded rules:
   - Revenue = `SUM(price + freight_value)`, never `SUM(price)` alone
   - Churn = no order within 30 days of 2011-12-10
   - Category names require JOIN to translation table
   - Revenue must filter `order_status = 'delivered'`
   - RFM segment definitions (Champions, High-Value, Mid-Value, Low-Value)
4. **Business Strategy Documents** (7 docs) — from `config/business_knowledge.json`:
   - Customer Retention Playbook
   - Revenue Optimization Playbook
   - Delivery Performance SOP
   - Churn Reduction Strategy
   - Marketplace Health Indicators
   - Forecast Interpretation Guide
   - Olist Dataset Documentation

**Embedding model:** `all-MiniLM-L6-v2` (SentenceTransformers)

**Run:** `python -m src.chatbot.ingest_knowledge` (must be run once before chatbot use)

### 7f. Long-Term Memory (`src/chatbot/memory_store.py`)

SQLite-backed persistent memory at `data/agent_memory.db`:

**Tables:**
- `findings` — stores analytical insights with topic, timestamp, importance
- `conversation_summaries` — stores per-thread session summaries

**Operations:**
- `store_finding(topic, finding, importance)` — called after ANALYTICAL synthesis
- `recall_findings(topic, limit)` — fuzzy topic match via LIKE
- `store_conversation_summary(thread_id, summary, key_topics)`
- `recall_conversation_context(thread_id, limit)`

### 7g. SQL Safety

Two validation rules applied to all generated SQL:
1. **Revenue formula check:** If query uses `SUM(price)` without `freight_value`, block it
2. **Write protection:** Block DELETE, UPDATE, INSERT, DROP, ALTER, CREATE, TRUNCATE, REPLACE, MERGE

---

## 8. API LAYER (`main.py`)

**FastAPI stub** — NOT yet connected to the agentic pipeline:
- `POST /chat` — accepts `{message: str}`, currently returns placeholder
- `GET /status` — returns `{status: "online"}`
- **TODO:** Wire `consult_logic_advanced()` into the `/chat` endpoint

---

## 9. KEY CONFIGURATION (`config/config.yaml`)

```yaml
paths:
  olist_db: data/processed/olist/olist.db
  prophet_model: models/forecasting/prophet_model.pkl
  churn_features: data/processed/churn_features.csv
  model_dir: models/churn/
  eval_dir: evaluation/churn/

features:
  snapshot_date: "2011-12-10"
  churn_window_days: 30
  rfm_bins: 5

model:
  layers: [128, 64, 32]
  dropout: 0.3
  loss: binary_crossentropy
  optimizer: adam

final_hp:
  learning_rate: 0.001
  batch_size: 64
  dropout: 0.3
  epochs: 50
  layers: [128, 64, 32]

forecasting.prophet:
  growth: linear
  seasonality_mode: additive
  changepoint_prior_scale: 0.05
  interval_width: 0.95
  # UK holidays added
```

---

## 10. ENVIRONMENT

- **GROQ_API_KEY** in `.env` — required for all LLM calls
- **Python venv** at `.venv/`
- **Run chatbot CLI:** `python -m src.chatbot.consultant_agent`
- **Run full ML pipeline:** `python run_pipeline.py [--skip-cleaning]`
- **Ingest knowledge base:** `python -m src.chatbot.ingest_knowledge`
- **Preprocess Olist to SQLite:** `python -m src.utils.preprocess_olist`
- **Preprocess for Prophet:** `python -m src.utils.preprocess_forecasting`

---

## 11. WHAT HAS BEEN DONE (COMPLETED WORK)

1. ✅ **Data Cleaning Pipeline** — Both Online Retail and Olist datasets cleaned, merged, and stored
2. ✅ **Olist SQLite Database** — 9 CSVs imported into normalized SQLite for chatbot queries
3. ✅ **Churn Feature Engineering** — RFM + 5 additional features + churn label + segment labels
4. ✅ **Churn DNN Training** — HP grid search, final model trained, baselines trained, all saved
5. ✅ **Churn Evaluation** — Full metrics, ROC/PR curves, confusion matrices, bias check per segment, error analysis
6. ✅ **Prophet Forecasting** — Model trained with UK holidays, cross-validated, forecast generated
7. ✅ **LSTM Architecture** — Code complete for multivariate 30-day forecasting
8. ✅ **Deterministic KPI Engine** — 9 KPIs with SQL formulas, shared between dashboard and chatbot
9. ✅ **11 Analytical Tools** — Revenue trends, delivery, customer behavior, reviews, categories, sellers, geography, market basket, CLV, cancellations, root cause
10. ✅ **ChromaDB Knowledge Base** — KPIs, model docs, business rules, strategy playbooks embedded
11. ✅ **Agentic RAG System** — Full LangGraph multi-agent pipeline with Router → Planner → Gatherer → RAG → Synthesizer
12. ✅ **Long-Term Memory** — SQLite-backed findings and conversation persistence
13. ✅ **Tool Caching** — diskcache with TTL for expensive tool calls
14. ✅ **SQL Safety Layer** — Revenue formula validation + write protection

---

## 12. WHAT REMAINS (TODO / NEXT STEPS)

1. ⏳ **Customer Segmentation Module** — RFM + K-Means (pending)
2. ⏳ **FastAPI Integration** — Wire `consult_logic_advanced()` into `main.py /chat` endpoint
3. ⏳ **Frontend Dashboard** — No frontend exists yet; API is stub-only
4. ⏳ **LSTM Training Execution** — Architecture exists but model hasn't been trained on Olist data
5. ⏳ **Colab Runner** — `colab_chatbot_runner.ipynb` exists but needs updating for current architecture
6. 🔧 **Model Improvements** — Churn DNN struggles with Champions segment (AUC ~0.66), could benefit from feature engineering or ensemble approaches

---

## 13. CRITICAL BUSINESS RULES (FOR ANY AI MODIFYING CODE)

1. **Revenue = `SUM(price + freight_value)`** — NEVER use `SUM(price)` alone
2. **Always filter `order_status = 'delivered'`** for financial metrics
3. **Customer identity = `customer_unique_id`** (not `customer_id` which is per-order)
4. **Category names** are in Portuguese → always JOIN `product_category_name_translation`
5. **Churn definition:** No purchase within 30 days of snapshot date 2011-12-10
6. **All KPIs** must flow through `KPIEngine` for consistency between dashboard and chatbot
7. **Olist DB is read-only** — all connections use `?mode=ro&uri=true`
