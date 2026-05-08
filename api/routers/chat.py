"""
Chat Router — AI Consultant Endpoint
Wires consult_logic_advanced() into a REST POST endpoint.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio
import traceback
import uuid

router = APIRouter(prefix="/api/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    intent: str = "UNKNOWN"
    status: str = "ok"


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the AI Business Consultant (LangGraph multi-agent).
    Returns a consultant-grade response with intent classification.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        from src.chatbot.consultant_agent import consult_logic_advanced, compiled_graph
        response_text = await consult_logic_advanced(request.message, thread_id)

        # Try to get intent from the last run (best effort)
        return ChatResponse(
            response=response_text,
            thread_id=thread_id,
            status="ok",
        )

    except ImportError as e:
        # Agent not available — return informative placeholder
        raise HTTPException(
            status_code=503,
            detail=f"AI Consultant not available: {str(e)}. Make sure GROQ_API_KEY is set and all dependencies are installed."
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Consultant error: {str(e)}")


@router.get("/health")
def chat_health():
    """Check if the AI consultant backend is ready."""
    checks = {}
    try:
        import os
        checks["groq_key"] = bool(os.getenv("GROQ_API_KEY"))
        if not checks["groq_key"]:
            try:
                with open(".env") as f:
                    for line in f:
                        if line.startswith("GROQ_API_KEY"):
                            checks["groq_key"] = True
            except Exception:
                pass
    except Exception:
        checks["groq_key"] = False

    try:
        import os
        checks["olist_db"] = os.path.exists("data/processed/olist/olist.db")
        checks["vector_db"] = os.path.exists("data/vector_db")
    except Exception:
        checks["olist_db"] = False
        checks["vector_db"] = False

    all_ready = all(checks.values())
    return {
        "ready": all_ready,
        "checks": checks,
        "message": "AI Consultant is ready." if all_ready else "Some dependencies are missing. Check the details.",
    }
