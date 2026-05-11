# Graph Report - business-insights-and-sales-forecasting-tool  (2026-05-10)

## Corpus Check
- 87 files · ~12,003,844 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 558 nodes · 789 edges · 42 communities (34 shown, 8 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 14 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `76ac21d0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 32|Community 32]]

## God Nodes (most connected - your core abstractions)
1. `PHASE 1: PROJECT ANALYSIS` - 16 edges
2. `BISFT — Project Memory` - 15 edges
3. `parsePagination()` - 11 edges
4. `_conn()` - 11 edges
5. `Business Insights & Sales Forecasting Tool` - 11 edges
6. `MemoryStore` - 10 edges
7. `7b. Table Designs` - 10 edges
8. `InventoryService` - 9 edges
9. `main()` - 9 edges
10. `authenticate()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `chat()` --calls--> `consult_logic_advanced()`  [INFERRED]
  api/routers/chat.py → src/chatbot/consultant_agent.py
- `_get_kpi_engine()` --calls--> `KPIEngine`  [INFERRED]
  api/routers/dashboard.py → src/analytics/kpi_engine.py
- `AgentState` --uses--> `MemoryStore`  [INFERRED]
  src/chatbot/consultant_agent.py → src/chatbot/memory_store.py
- `ProtectedRoute()` --calls--> `useAuth()`  [EXTRACTED]
  frontend/src/App.jsx → frontend/src/context/AuthContext.jsx
- `Sidebar()` --calls--> `useAuth()`  [EXTRACTED]
  frontend/src/components/Sidebar.jsx → frontend/src/context/AuthContext.jsx

## Communities (42 total, 8 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (36): PAGINATION, STOCK_REASONS, InsufficientStockError, ValidationError, router, AddStockInput, addStockSchema, AdjustStockInput (+28 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (28): { page, limit }, router, where, router, LoginInput, loginSchema, RegisterInput, registerSchema (+20 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (21): api, interceptors, inventoryApi, token, NAV_ITEMS, OPS_ITEMS, Sidebar(), AuthContext (+13 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (46): async_cache(), execute_deterministic_kpi(), get_churn_error_analysis(), get_churn_risk_by_segment(), get_churn_risk_overview(), get_forecast_metrics(), get_kpi_definition(), get_model_registry() (+38 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (39): 10. Transaction Management Strategy, 11. Validation & Security Approach, 12. Scalability Considerations, 13. Frontend Integration Guide, 14. Data Seeding Strategy, 15. Implementation Order, 1. Current Backend Architecture, 2. Existing Database Schema (Olist SQLite) (+31 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (35): 10. ENVIRONMENT, 11. WHAT HAS BEEN DONE (COMPLETED WORK), 12. WHAT REMAINS (TODO / NEXT STEPS), 13. CRITICAL BUSINESS RULES (FOR ANY AI MODIFYING CODE), 1. PROJECT OVERVIEW, 2. TECHNOLOGY STACK, 3. PROJECT STRUCTURE (ACTUAL), 4. DATA PIPELINE (+27 more)

### Community 6 - "Community 6"
Cohesion: 0.13
Nodes (24): analyze_category_performance(), analyze_customer_behavior(), analyze_delivery_performance(), analyze_geographic_distribution(), analyze_market_basket(), analyze_order_cancellation(), analyze_revenue_trends(), analyze_review_scores() (+16 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (12): KPIEngine, Calculate a specific KPI by its ID from the registry., Returns a dictionary of all core KPI values for a dashboard summary., Central engine for calculating deterministic business KPIs.     Ensures that Da, get_all_kpis(), _get_kpi_engine(), Dashboard Router — KPI Endpoints Calls the deterministic KPIEngine for all core, Lazy-load KPIEngine to avoid import errors at startup. (+4 more)

### Community 8 - "Community 8"
Cohesion: 0.18
Nodes (15): build_churn_model(), get_baseline_models(), churn_model.py ============== Business Insights & Sales Forecasting Tool Phas, Return a dictionary of baseline Scikit-learn classifiers for comparison.     Th, Build and compile the Customer Churn DNN.      Parameters     ----------, apply_smote(), hp_grid_search(), load_and_preprocess() (+7 more)

### Community 9 - "Community 9"
Cohesion: 0.14
Nodes (14): BaseModel, list_tools(), Analytics Router — 11 Analytical Tools, List all available analytical tools with their descriptions and topics., Execute an analytical tool and return the formatted text result., run_tool(), RunToolRequest, chat() (+6 more)

### Community 10 - "Community 10"
Cohesion: 0.13
Nodes (8): MemoryStore, Long-Term Analytical Memory Store ================================== PostgreSQ, Persistent memory for analytical findings and conversation context., Store an analytical finding for future recall., Retrieve past findings, optionally filtered by topic., Store a conversation summary for long-term recall., Get recent conversation summaries for a thread., Return all distinct topics from findings.

### Community 11 - "Community 11"
Cohesion: 0.12
Nodes (15): 1. Primary Store: PostgreSQL (`bisft_inventory`), 2. AI Intelligence Storage, 3. Legacy / Development Storage, 4. Connection Configuration, 5. Entity-Relationship Highlights, 6. Maintenance Commands, **A. Operational Schema (Inventory Management)**, **A. Vector Database (ChromaDB)** (+7 more)

### Community 12 - "Community 12"
Cohesion: 0.22
Nodes (7): NotFoundError, router, CreateSupplierInput, createSupplierSchema, UpdateSupplierInput, updateSupplierSchema, SupplierService

### Community 13 - "Community 13"
Cohesion: 0.21
Nodes (14): bias_check(), compute_metrics(), error_analysis(), load_dnn(), load_test_data(), main(), plot_confusion_matrix(), plot_pr_curve() (+6 more)

### Community 14 - "Community 14"
Cohesion: 0.14
Nodes (13): AI-Powered Decision Intelligence Platform, Business Insights & Sales Forecasting Tool, code:block1 (BISFT/), code:bash (# 1. Create virtual environment), Contributors, Dataset Strategy, GitHub Repository, License (+5 more)

### Community 15 - "Community 15"
Cohesion: 0.15
Nodes (13): 7. Database Design (PostgreSQL + Prisma), 7a. Entity Relationship Diagram, 7b. Table Designs, code:sql (id              UUID        PK), code:sql (id              UUID        PK), code:sql (id              UUID        PK), code:sql (id              UUID        PK), code:sql (id              UUID        PK) (+5 more)

### Community 16 - "Community 16"
Cohesion: 0.31
Nodes (8): build_lstm_model(), create_windows(), main(), prepare_time_series_data(), lstm_forecaster.py ================== Business Insights & Sales Forecasting Tool, Aggregate transaction data into daily multivariate time series., Create sliding windows for LSTM.     data: scaled numpy array     lookback: numb, Build multivariate LSTM model.

### Community 17 - "Community 17"
Cohesion: 0.25
Nodes (7): get_churn_segments(), get_error_analysis(), get_model_comparison(), Churn Router — Segment Risk & Model Comparison, Return FP/FN feature profiles for model reliability assessment., Return per-segment churn rates and model AUC from evaluation data., Return DNN vs baseline model comparison from evaluation data.

### Community 18 - "Community 18"
Cohesion: 0.36
Nodes (7): clean_olist(), clean_online_retail(), load_olist(), main(), Load and clean the Online Retail (.xlsx) dataset., Load all Olist CSV files into a dictionary of DataFrames., Merge and clean the Olist dataset.

### Community 19 - "Community 19"
Cohesion: 0.33
Nodes (6): _build_summary(), get_forecast(), get_forecast_metrics(), Forecasting Router — Prophet Forecast & Metrics, Return forecast data points for charting.     Tries live Prophet model first, f, Return Prophet cross-validation performance metrics.

### Community 21 - "Community 21"
Cohesion: 0.38
Nodes (5): batchInsert(), OLIST_DB_PATH, prisma, seed(), slugify()

### Community 22 - "Community 22"
Cohesion: 0.33
Nodes (4): fs, path, prisma, { PrismaClient }

### Community 23 - "Community 23"
Cohesion: 0.5
Nodes (4): build_documents(), ingest(), Knowledge Ingestion Pipeline for Enterprise RAG ================================, Convert all structured business knowledge into searchable text chunks.

### Community 24 - "Community 24"
Cohesion: 0.5
Nodes (4): build_churn_features(), main(), churn_features.py ================= Business Insights & Sales Forecasting Tool, Return a customer-level DataFrame with RFM + engineered features and churn label

## Knowledge Gaps
- **206 isolated node(s):** `BISFT — Business Insights & Sales Forecasting Tool FastAPI Backend — Main Entry`, `Analytics Router — 11 Analytical Tools`, `List all available analytical tools with their descriptions and topics.`, `Execute an analytical tool and return the formatted text result.`, `Chat Router — AI Consultant Endpoint Wires consult_logic_advanced() into a REST` (+201 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `app` connect `Community 2` to `Community 1`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Why does `PHASE 1: PROJECT ANALYSIS` connect `Community 4` to `Community 15`?**
  _High betweenness centrality (0.008) - this node is a cross-community bridge._
- **Why does `AgentState` connect `Community 3` to `Community 10`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **What connects `BISFT — Business Insights & Sales Forecasting Tool FastAPI Backend — Main Entry`, `Analytics Router — 11 Analytical Tools`, `List all available analytical tools with their descriptions and topics.` to the rest of the system?**
  _206 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._