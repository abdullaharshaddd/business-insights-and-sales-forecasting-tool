# Business Insights & Sales Forecasting Tool
### AI-Powered Decision Intelligence Platform

---

## GitHub Repository
https://github.com/abdullaharshaddd/business-insights-and-sales-forecasting-tool

---

## Overview

An AI-driven decision intelligence system for predictive analytics, customer segmentation, and intelligent data querying — built on the Online Retail and Olist datasets.

| Module | Algorithm | Dataset |
|---|---|---|
| Sales Forecasting | Prophet / LSTM | Online Retail |
| Customer Segmentation | RFM + K-Means | Online Retail |
| NL2SQL Chatbot | LangChain + SQLite | Olist |

---

## Project Structure

```
BISFT/
├── data/
│   ├── raw/
│   │   ├── online_retail/       ← Online Retail.xlsx
│   │   ├── olist/               ← 9 Olist CSVs
│   │   └── zips/                ← archive.zip, online+retail.zip
│   ├── processed/
│   │   ├── online_retail/       ← daily_revenue.csv, rfm_features.csv, rfm_segmented.csv
│   │   └── olist/               ← olist.db (SQLite)
│   └── external/
│
├── notebooks/
│   ├── 01_eda/                  ← Exploratory data analysis
│   ├── 02_forecasting/          ← Prophet & LSTM experiments
│   ├── 03_segmentation/         ← RFM + K-Means exploration
│   └── 04_chatbot/              ← NL2SQL demos
│
├── src/
│   ├── forecasting/
│   │   ├── prophet_model.py     ← Train & evaluate Prophet
│   │   └── lstm_model.py        ← Train & evaluate LSTM
│   ├── segmentation/
│   │   └── rfm_kmeans.py        ← RFM computation + K-Means
│   ├── chatbot/
│   │   └── nl2sql_agent.py      ← LangChain NL2SQL agent
│   ├── dashboard/               ← FastAPI / Flask routes (future)
│   └── utils/
│       ├── data_cleaning.py     ← Shared cleaning functions
│       ├── preprocess_online_retail.py
│       └── preprocess_olist.py
│
├── models/
│   ├── forecasting/             ← prophet_model.pkl, lstm_model.h5
│   └── segmentation/            ← kmeans_model.pkl
│
├── evaluation/
│   ├── forecasting/             ← prophet_forecast.csv, lstm_forecast.csv, cv_metrics.csv
│   └── segmentation/            ← silhouette_score.txt
│
├── reports/
│   ├── figures/                 ← Saved plots (.png)
│   └── tables/                  ← Saved result tables (.csv)
│
├── config/
│   └── config.yaml              ← All hyperparameters & paths
│
├── tests/
│   ├── test_forecasting/
│   ├── test_segmentation/
│   └── test_chatbot/
│
├── docs/                        ← Academic report, diagrams
├── .env.example                 ← API key template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Preprocess data
python -m src.utils.preprocess_online_retail
python -m src.utils.preprocess_olist

# 4. Train forecasting model
python -m src.forecasting.prophet_model

# 5. Run segmentation
python -m src.segmentation.rfm_kmeans

# 6. Start NL2SQL chatbot
python -m src.chatbot.nl2sql_agent
```

---

## Dataset Strategy

| Dataset | Purpose |
|---|---|
| **Online Retail** (.xlsx) | Daily revenue series for Prophet/LSTM; Customer-level RFM for K-Means |
| **Olist** (9 CSVs → SQLite) | Relational database powering the NL2SQL chatbot |

---

## Technologies

- **Python 3.11+**, Pandas, NumPy, Scikit-learn
- **Prophet** (forecasting), **TensorFlow/Keras** (LSTM)
- **LangChain** + SQLite (NL2SQL chatbot)
- **FastAPI** (dashboard API — upcoming)
- **Matplotlib**, Seaborn, Plotly (visualisation)

---

## Contributors

- Abdullah Arshad (23L-2531)
- Sohaib Haider (23L-2519)

---

## License

Academic Project — FAST-NUCES
