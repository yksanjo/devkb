"""Search API routes"""
from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.models import SearchRequest, SearchResponse
from app.services.search import get_search_service

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Semantic search across documents"""
    try:
        search_service = get_search_service()
        results = search_service.search(request)

        return SearchResponse(
            results=results,
            query=request.query,
            total=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/keyword", response_model=SearchResponse)
async def keyword_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100),
    content_type: str = None,
    language: str = None,
    category: str = None,
):
    """Keyword-based search"""
    try:
        search_service = get_search_service()

        from app.models import ContentType

        request = SearchRequest(
            query=q,
            limit=limit,
            content_type=ContentType(content_type) if content_type else None,
            language=language,
            category=category,
        )

        results = search_service.keyword_search(
            query=q,
            limit=limit,
            content_type=request.content_type,
            language=language,
            category=category,
        )

        return SearchResponse(
            results=results,
            query=q,
            total=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
