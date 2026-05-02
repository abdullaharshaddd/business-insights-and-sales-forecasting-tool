from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.chatbot.business_toolkit import query_database, get_sales_forecast_summary, get_churn_risk_overview
import os

app = FastAPI(title="BISFT Strategic Consultant API")

class ChatRequest(BaseModel):
    message: str

# This is where we will integrate the "Brain"
# For now, it uses the logic from your agent but as a Web API
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Here you would call your consult_logic
        # In a real app, you'd use Groq or OpenAI here for instant speed
        return {"response": f"The consultant received: {request.message}. (Logic ready to be connected)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    return {"status": "online", "database": "connected"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
