"""
BISFT — Business Insights & Sales Forecasting Tool
FastAPI Backend — Main Entry Point
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from api.routers import dashboard, forecasting, churn, chat, analytics

app = FastAPI(
    title="BISFT Strategic Consultant API",
    description="AI-Powered Decision Intelligence Platform for Business Analytics",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── CORS — allow React dev server ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # fallback
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────────────
app.include_router(dashboard.router)
app.include_router(forecasting.router)
app.include_router(churn.router)
app.include_router(chat.router)
app.include_router(analytics.router)


@app.get("/")
def root():
    return {
        "name": "BISFT Strategic Consultant API",
        "version": "2.0.0",
        "status": "online",
        "docs": "/api/docs",
        "endpoints": {
            "kpis": "/api/dashboard/kpis",
            "system_status": "/api/dashboard/status",
            "forecast": "/api/forecast?days=30",
            "forecast_metrics": "/api/forecast/metrics",
            "churn_segments": "/api/churn/segments",
            "churn_models": "/api/churn/models",
            "chat": "POST /api/chat",
            "analytics_tools": "/api/analytics/tools",
            "analytics_run": "POST /api/analytics/run",
        },
    }


if __name__ == "__main__":
    import uvicorn
    # Auto-reload triggered
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
