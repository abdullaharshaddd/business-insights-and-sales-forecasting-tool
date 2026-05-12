# BISFT — Business Insights & Sales Forecasting Tool
## Complete Project Overview

**Academic Project — FAST-NUCES**
**Contributors:** Abdullah Arshad (23L-2531) · Sohaib Haider (23L-2519)
**Repository:** https://github.com/abdullaharshaddd/business-insights-and-sales-forecasting-tool

---

## What Is This Project?

BISFT is a full-stack AI-powered business intelligence platform. It takes raw e-commerce data (from two real-world datasets) and turns it into actionable insights through:

- **Sales forecasting** using machine learning models
- **Customer churn prediction** using a deep neural network
- **An AI chatbot consultant** that answers business questions in natural language
- **An inventory management system** with suppliers and purchase orders
- **A React dashboard** that visualizes everything in one place

---

## Datasets Used

| Dataset | Format | Purpose |
|---|---|---|
| **Online Retail** | Excel (.xlsx) | UK e-commerce transactions — used for sales forecasting and churn features |
| **Olist** | 9 CSV files → SQLite | Brazilian e-commerce marketplace data — used for the AI chatbot's SQL queries |

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│               React Frontend (Vite)                  │
│  Dashboard / Forecasting / Churn / Chat /            │
│  Inventory / Suppliers / Purchase Orders             │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (CORS-enabled)
          ┌────────────┴────────────┐
          │                         │
┌─────────▼──────────┐   ┌─────────▼──────────────────┐
│  Python FastAPI     │   │  Node.js / Express Backend  │
│  (main.py, port     │   │  (inventory-backend,        │
│   8000)             │   │   TypeScript + Prisma)      │
│                     │   │                             │
│  • /api/dashboard   │   │  • /api/v1/auth             │
│  • /api/forecast    │   │  • /api/v1/products         │
│  • /api/churn       │   │  • /api/v1/inventory        │
│  • /api/chat        │   │  • /api/v1/suppliers        │
│  • /api/analytics   │   │  • /api/v1/purchase-orders  │
└─────────┬──────────┘   └─────────┬──────────────────┘
          │                         │
          │                         │ Prisma ORM
          │                ┌────────▼────────┐
          │                │  PostgreSQL DB   │
          │                │  (bisft_inv.)    │
          │                └─────────────────┘
          │
   ┌──────▼──────────────────────────────────────────┐
   │           Python ML / AI Layer (src/)            │
   │                                                  │
   │  • Prophet / RF models  →  forecasting           │
   │  • DNN + baselines      →  churn prediction      │
   │  • LangGraph agents     →  AI chatbot            │
   │  • ChromaDB vectors     →  RAG knowledge base    │
   │  • SQLite (olist.db)    →  chatbot SQL queries   │
   └──────────────────────────────────────────────────┘
```

---

## Folder Structure (Annotated)

```
BISFT/
│
├── main.py                    ← FastAPI entry point — registers all API routers
├── requirements.txt           ← All Python dependencies
├── package.json               ← Root-level JS config (concurrently scripts)
├── config.yaml (in config/)   ← Master config: paths, model hyperparameters, DB URL
├── run_pipeline.py            ← Script to run the full ML training pipeline
├── run_forecasting_pipeline.bat  ← Windows bat to trigger forecasting pipeline
├── .env                       ← Secrets: GROQ_API_KEY, DATABASE_URL
│
├── api/                       ← FastAPI routers (Python)
│   └── routers/
│       ├── dashboard.py       ← KPI summaries and system status
│       ├── forecasting.py     ← Random Forest forecast endpoint (/api/forecast)
│       ├── churn.py           ← Churn segment risk and model comparison
│       ├── chat.py            ← AI chatbot POST endpoint (/api/chat)
│       └── analytics.py       ← Analytical tools runner
│
├── src/                       ← Core Python ML/AI source code
│   ├── models/
│   │   ├── churn_model.py     ← DNN architecture (TensorFlow/Keras) + sklearn baselines
│   │   ├── lstm_forecaster.py ← LSTM model for time-series (experimental)
│   │   └── rf_forecaster.py   ← Random Forest forecaster
│   │
│   ├── features/
│   │   └── churn_features.py  ← RFM feature engineering for churn
│   │
│   ├── training/
│   │   └── train_churn.py     ← Trains and saves the churn DNN model
│   │
│   ├── forecasting/
│   │   ├── prophet_model.py   ← Prophet model training and evaluation
│   │   ├── rf_feature_engineering.py  ← Feature engineering for RF forecaster
│   │   ├── feature_engineering.py     ← General forecasting features
│   │   └── eda_correlation.py ← EDA helpers for forecasting data
│   │
│   ├── chatbot/
│   │   ├── consultant_agent.py   ← LangGraph multi-agent orchestrator (CORE AI)
│   │   ├── business_toolkit.py   ← Tools used by the AI agent (KPIs, forecast, churn)
│   │   ├── ingest_knowledge.py   ← Loads business docs into ChromaDB vector store
│   │   └── memory_store.py       ← Persistent agent memory (findings + conversation)
│   │
│   ├── analytics/
│   │   ├── analytical_tools.py   ← Pre-built analysis functions (revenue, delivery, CLV…)
│   │   └── kpi_engine.py         ← Computes KPIs from database
│   │
│   ├── evaluation/
│   │   └── evaluate_churn.py     ← Evaluates churn model, produces bias/error reports
│   │
│   └── utils/
│       ├── preprocess_olist.py   ← Cleans/merges the 9 Olist CSVs → olist.db SQLite
│       └── preprocess_forecasting.py  ← Cleans Online Retail → daily_revenue.csv
│
├── frontend/                  ← React frontend (Vite + JSX)
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json           ← React dependencies (react-router, recharts, etc.)
│   └── src/
│       ├── main.jsx           ← React entry point
│       ├── App.jsx            ← Router setup + protected routes
│       ├── index.css          ← Global styles
│       ├── api/
│       │   └── client.js      ← Axios/fetch wrapper for API calls
│       ├── context/
│       │   └── AuthContext.jsx  ← Auth state (login/logout/session)
│       ├── components/
│       │   ├── Sidebar.jsx    ← Navigation sidebar
│       │   ├── PageHeader.jsx ← Page title component
│       │   └── LoadingSpinner.jsx
│       └── pages/
│           ├── Login.jsx          ← Login page (authenticates via inventory-backend)
│           ├── Dashboard.jsx      ← KPI overview cards
│           ├── Forecasting.jsx    ← Sales forecast chart (RF model, 7–90 days)
│           ├── Churn.jsx          ← Churn risk by segment + model comparison table
│           ├── Chat.jsx           ← AI chatbot interface
│           ├── Analytics.jsx      ← Analytical tools runner
│           ├── Inventory.jsx      ← Product inventory levels
│           ├── Suppliers.jsx      ← Supplier management
│           └── PurchaseOrders.jsx ← Purchase order tracking
│
├── inventory-backend/         ← Node.js / TypeScript REST API (Inventory system)
│   ├── src/
│   │   ├── server.ts          ← HTTP server entry point
│   │   ├── app.ts             ← Express app: middleware + route registration
│   │   ├── config/
│   │   │   ├── database.ts    ← Prisma client singleton
│   │   │   ├── env.ts         ← Zod-validated env vars
│   │   │   └── constants.ts
│   │   ├── middleware/
│   │   │   ├── auth.middleware.ts    ← JWT verification
│   │   │   ├── audit.middleware.ts   ← Logs every write action to audit_logs table
│   │   │   ├── error.middleware.ts   ← Global error handler
│   │   │   └── validate.middleware.ts ← Zod request validation
│   │   ├── modules/           ← Feature modules (routes + schema + service)
│   │   │   ├── auth/          ← Login, register, JWT issue
│   │   │   ├── products/      ← CRUD for products
│   │   │   ├── categories/    ← Product category hierarchy
│   │   │   ├── inventory/     ← Stock levels + stock movements
│   │   │   ├── suppliers/     ← Supplier management
│   │   │   ├── purchase-orders/ ← PO lifecycle (draft → received)
│   │   │   └── audit-logs/    ← Read audit trail
│   │   ├── shared/
│   │   │   ├── errors/AppError.ts   ← Custom error class
│   │   │   └── utils/response.ts    ← Standardised JSON response helpers
│   │   └── seed/
│   │       └── seed-from-olist.ts   ← Seeds products/suppliers from Olist CSV data
│   └── prisma/
│       └── schema.prisma      ← Full database schema (see Database section below)
│
├── models/                    ← Saved trained model files
│   ├── churn/
│   │   ├── churn_dnn_best.h5  ← Best Keras DNN weights
│   │   ├── gradient_boosting.pkl
│   │   ├── random_forest.pkl
│   │   ├── logistic_regression.pkl
│   │   ├── k-nearest_neighbors.pkl
│   │   ├── scaler.pkl         ← StandardScaler fitted on training data
│   │   └── feature_cols.json  ← Feature names list
│   ├── forecasting/
│   │   └── prophet_model.pkl  ← Trained Prophet model
│   └── lstm/
│       ├── sales_lstm_model.h5
│       └── forecast_plot.png
│
├── data/                      ← All data files
│   ├── raw/
│   │   ├── online_retail/
│   │   │   └── online_retail.xlsx   ← Original UK Online Retail dataset
│   │   └── olist/             ← 9 raw Olist CSV files (customers, orders, items…)
│   ├── processed/
│   │   ├── online_retail_cleaned.csv
│   │   ├── daily_revenue.csv         ← Aggregated daily sales for forecasting
│   │   ├── churn_features.csv        ← RFM + churn label per customer
│   │   ├── olist_merged_cleaned.csv
│   │   └── olist/
│   │       └── olist.db              ← SQLite database for chatbot SQL queries
│   ├── vector_db/             ← ChromaDB persistent vector store (RAG knowledge)
│   ├── agent_memory.db        ← SQLite: agent findings + conversation summaries
│   ├── checkpoints.sqlite     ← LangGraph memory checkpointer
│   └── tool_cache/            ← Disk cache for expensive tool calls
│
├── evaluation/                ← Model evaluation output files
│   ├── churn/
│   │   ├── bias_check_by_segment.csv   ← AUC/F1 per customer segment
│   │   ├── error_profiles.csv          ← FP/FN feature averages
│   │   └── error_analysis_samples.csv
│   └── forecasting/
│       ├── forecast_results.csv
│       ├── metrics.csv
│       └── summary_metrics.json
│
├── reports/                   ← Generated figures and tables for the paper
│   ├── figures/               ← PNG plots (confusion matrices, ROC curves, forecasts)
│   └── tables/                ← CSV tables (baseline comparison, final metrics)
│
├── config/                    ← Configuration files
│   ├── config.yaml            ← Master config (paths, hyperparameters, DB URL)
│   ├── business_knowledge.json ← Domain knowledge injected into RAG
│   ├── kpi_definitions.json   ← KPI names, formulas, SQL mappings
│   └── model_registry.json    ← Registered model versions
│
├── database/                  ← Database utilities
│   ├── schema.sql             ← Raw SQL schema
│   └── etl.py                 ← ETL pipeline (CSV → PostgreSQL)
│
├── docs/                      ← Academic documentation
│   └── 23L-2519 23L-2531 AI Project.md
│
├── evaluation/                ← (see above)
├── notebooks/                 ← Jupyter notebooks for EDA and experiments
├── scripts/
│   ├── data_cleaning.py       ← Standalone data cleaning script
│   └── delete_old_models.py   ← Utility to prune old model files
├── scratch/                   ← Throwaway debug scripts
│   └── check_db.py
├── report/                    ← LaTeX source for academic report
│   ├── 23L-2519 23L-2531 AI Project.tex
│   └── generate_figures.py
└── graphify-out/              ← Auto-generated code graph visualizations
    ├── graph.html
    ├── GRAPH_REPORT.md
    └── GRAPH_TREE.html
```

---

## Core Modules Explained

### 1. Python FastAPI Backend (`main.py` + `api/`)

The main Python server runs on **port 8000**. It exposes five route groups:

| Router | Prefix | What it does |
|---|---|---|
| `dashboard.py` | `/api/dashboard` | Returns KPI cards and system health status |
| `forecasting.py` | `/api/forecast` | Runs the Random Forest model to predict daily sales for 7–90 days ahead |
| `churn.py` | `/api/churn` | Returns churn risk per customer segment and model performance metrics |
| `chat.py` | `/api/chat` | POST endpoint that receives a user message and returns an AI consultant response |
| `analytics.py` | `/api/analytics` | Runs pre-built analytical tools (revenue trends, delivery analysis, CLV, etc.) |

---

### 2. AI Chatbot — LangGraph Multi-Agent System (`src/chatbot/consultant_agent.py`)

This is the most complex part of the project. It is a **multi-agent RAG system** built with LangGraph and powered by Groq's LLM API (Llama 3 models).

**Agent flow:**

```
User Message
     │
     ▼
[Router Node]  ← classifies intent: ANALYTICAL / SIMPLE / OTHER
     │
     ├─── ANALYTICAL ──► [Planner] ──► [Memory Recall] ──► [Data Gatherer] ──► [RAG Retriever] ──► [Synthesizer]
     │
     ├─── SIMPLE ──────────────────────────────────────────────────────────────► [Simple Executor] ──► [Synthesizer]
     │
     └─── OTHER ────────────────────────────────────────────────────────────────► [Greeting Node] ──► END
```

**Key agents:**
- **Router** — uses `llama-3.1-8b-instant` to classify the query in one token
- **Planner** — uses `llama-3.3-70b-versatile` to decompose complex questions into 3–6 investigation steps
- **Data Gatherer** — executes the plan: runs KPI queries, forecast, churn, SQL, or analytical tools
- **RAG Retriever** — searches the ChromaDB vector store for relevant business knowledge docs
- **Memory Node** — recalls past findings stored in `agent_memory.db`
- **Synthesizer** — produces the final consultant-grade response (executive summary, evidence, recommendations)

**Models used:**
- `llama-3.1-8b-instant` — fast router and simple tasks
- `llama-3.3-70b-versatile` — planner and synthesizer
- ChromaDB + `sentence-transformers` — local vector search (no external embeddings API needed)

---

### 3. Customer Churn Prediction (`src/models/churn_model.py`, `src/training/train_churn.py`)

Predicts whether a customer will churn (stop buying) based on their RFM (Recency, Frequency, Monetary) features.

**Model architecture (DNN):**
```
Input Features → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
              → Dense(64,  ReLU) → BatchNorm → Dropout(0.3)
              → Dense(32,  ReLU) → BatchNorm
              → Dense(1, Sigmoid) → Churn Probability
```

**Baseline models for comparison:** Logistic Regression, Random Forest, Gradient Boosting, KNN

**Class imbalance handling:** SMOTE oversampling

**Best results (DNN):** AUC-ROC ≈ 0.82, F1 ≈ 0.71

---

### 4. Sales Forecasting (`src/forecasting/`, `src/models/`)

Two models were built:

| Model | File | Status | Notes |
|---|---|---|---|
| **Prophet** | `prophet_model.py` | Trained & saved | Facebook's time-series model with UK holiday support |
| **Random Forest** | `rf_forecaster.py` | Active in API | Uses lag features (1-day, 7-day, 30-day) + calendar features |
| **LSTM** | `lstm_forecaster.py` | Experimental | Deep learning forecaster (not in production API) |

The RF forecaster generates predictions iteratively: each predicted day is appended to the history so it can compute the next day's lag features.

---

### 5. Inventory Backend (`inventory-backend/`)

A separate **Node.js/TypeScript** REST API that manages the operational inventory system. It runs independently from the Python backend.

**Tech stack:** Express.js, Prisma ORM, PostgreSQL, JWT auth, Zod validation, Helmet security

**Modules:**
- `auth` — register/login, returns JWT tokens
- `products` — full CRUD with category and supplier relations
- `categories` — hierarchical product categories (parent/child)
- `inventory` — stock levels, stock movements (IN/OUT/ADJUSTMENT/RETURN)
- `suppliers` — supplier contact info and lead times
- `purchase-orders` — PO lifecycle: draft → submitted → confirmed → received
- `audit-logs` — every write action is logged with old/new values and user IP

**Database seeding:** `seed-from-olist.ts` imports Olist products and sellers into the PostgreSQL database.

---

### 6. React Frontend (`frontend/`)

Built with **Vite + React**. All routes are protected by JWT auth (redirects to `/login` if no session).

**Pages:**

| Page | Route | What it shows |
|---|---|---|
| Login | `/login` | Auth form, calls inventory-backend |
| Dashboard | `/dashboard` | KPI summary cards (revenue, orders, churn rate) |
| Forecasting | `/forecasting` | Interactive chart of future daily sales (RF model) |
| Churn | `/churn` | Churn risk table by segment + model comparison |
| Chat | `/chat` | Chat interface to the AI consultant |
| Analytics | `/analytics` | Run analytical tools, view structured output |
| Inventory | `/inventory` | Product stock levels |
| Suppliers | `/suppliers` | Supplier list and details |
| Purchase Orders | `/purchase-orders` | PO status tracker |

---

## Database Schema (PostgreSQL via Prisma)

The unified PostgreSQL database (`bisft_inventory`) holds two categories of tables:

**Operational (inventory system):**
- `users` — system users with roles: admin / manager / staff / viewer
- `categories` — hierarchical product categories
- `products` — SKU, pricing, dimensions, reorder thresholds
- `inventory` — current stock quantity per product
- `stock_movements` — audit trail of every stock change (IN/OUT/ADJUSTMENT/RESERVATION/RETURN/RELEASE)
- `suppliers` — supplier info and lead times
- `purchase_orders` + `purchase_order_items` — PO management
- `audit_logs` — all write actions with before/after JSON values

**Historical Olist data (for AI chatbot SQL queries):**
- `customers`, `orders`, `order_items`, `order_payments`, `order_reviews`, `geolocation`

**Agent memory:**
- `findings` — key analytical findings stored by the AI agent
- `conversation_summaries` — compressed conversation history per thread

---

## Technology Stack Summary

| Layer | Technologies |
|---|---|
| **Frontend** | React 18, Vite, React Router, Recharts |
| **Python API** | FastAPI, Uvicorn, Pydantic v2 |
| **Node.js API** | Express.js, TypeScript, Prisma ORM |
| **Database** | PostgreSQL (main), SQLite (olist.db + agent memory + caching) |
| **ML Models** | TensorFlow/Keras (DNN), Scikit-learn (RF, GB, LR, KNN), Prophet |
| **AI/LLM** | LangGraph, LangChain, Groq API (Llama 3.1 8B + Llama 3.3 70B) |
| **Vector DB** | ChromaDB (local) + sentence-transformers (HuggingFace embeddings) |
| **Data** | Pandas, NumPy, OpenPyXL |
| **Auth** | JWT (inventory-backend), Helmet, bcrypt |
| **Validation** | Pydantic (Python), Zod (TypeScript) |

---

## How to Run

### 1. Prerequisites
```bash
# Python dependencies
pip install -r requirements.txt

# Node.js dependencies (inventory backend)
cd inventory-backend && npm install

# Frontend dependencies
cd frontend && npm install
```

### 2. Environment Setup
Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://postgres:password@localhost:5432/bisft_inventory
```

### 3. Data Pipeline (first time only)
```bash
python -m src.utils.preprocess_forecasting   # Clean Online Retail data
python -m src.utils.preprocess_olist         # Build olist.db SQLite
python -m src.training.train_churn           # Train churn DNN
python run_pipeline.py                        # Train forecasting models
```

### 4. Start Servers
```bash
# Python FastAPI (port 8000)
python main.py

# Node.js inventory backend
cd inventory-backend && npm run dev

# React frontend (port 5173)
cd frontend && npm run dev
```

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| Data Cleaning | Online Retail & Olist preprocessing | ✅ Done |
| Churn Prediction | DNN + LR/RF/KNN/GB baselines | ✅ Done |
| Sales Forecasting | Prophet + Random Forest | ✅ Done |
| AI Chatbot | LangGraph multi-agent RAG system | ✅ Done |
| Inventory System | Node.js backend + Prisma + PostgreSQL | ✅ Done |
| React Dashboard | Full frontend with all pages | ✅ Done |
| LSTM Forecasting | Deep learning time-series | 🔬 Experimental |
| Customer Segmentation | RFM + K-Means | ⏳ Pending |
