"""Chat API routes for Claude Code integration"""
from fastapi import APIRouter, HTTPException

from app.models import ChatRequest, ChatResponse
from app.services.claude import get_claude_service
from app.services.search import get_search_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with Claude using knowledge base context"""
    try:
        claude_service = get_claude_service()

        if not claude_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Claude API key not configured. Please set ANTHROPIC_API_KEY in .env",
            )

        return claude_service.chat(request)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain")
async def explain_code(code: str, language: str = None):
    """Explain a piece of code using Claude"""
    try:
        claude_service = get_claude_service()

        if not claude_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Claude API key not configured. Please set ANTHROPIC_API_KEY in .env",
            )

        explanation = claude_service.explain_code(code, language)
        return {"explanation": explanation}
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
