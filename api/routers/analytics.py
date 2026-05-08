"""
Analytics Router — 11 Analytical Tools
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import traceback

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/tools")
def list_tools():
    """List all available analytical tools with their descriptions and topics."""
    try:
        from src.analytics.analytical_tools import ANALYTICAL_TOOLS
        tools = []
        for tool_id, info in ANALYTICAL_TOOLS.items():
            tools.append({
                "id": tool_id,
                "description": info.get("description", ""),
                "topics": info.get("topics", []),
            })
        return {"status": "ok", "tools": tools, "count": len(tools)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class RunToolRequest(BaseModel):
    tool_id: str
    topic: str = "general"  # Only used for investigate_root_causes


@router.post("/run")
def run_tool(request: RunToolRequest):
    """Execute an analytical tool and return the formatted text result."""
    try:
        from src.analytics.analytical_tools import ANALYTICAL_TOOLS

        if request.tool_id not in ANALYTICAL_TOOLS:
            available = list(ANALYTICAL_TOOLS.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{request.tool_id}' not found. Available: {available}"
            )

        tool_info = ANALYTICAL_TOOLS[request.tool_id]
        fn = tool_info["fn"]

        # investigate_root_causes accepts a topic argument
        if request.tool_id == "investigate_root_causes":
            result = fn(topic=request.topic)
        else:
            result = fn()

        return {
            "status": "ok",
            "tool_id": request.tool_id,
            "description": tool_info.get("description", ""),
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Tool execution error: {str(e)}")
