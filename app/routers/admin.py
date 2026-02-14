"""Admin API routes"""
from fastapi import APIRouter, HTTPException

from app.database import db
from app.models import IndexDirectoryRequest, StatsResponse
from app.services.documents import get_document_service
from app.services.embeddings import get_embedding_service

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/index/directory")
async def index_directory(request: IndexDirectoryRequest):
    """Index all files in a directory"""
    try:
        doc_service = get_document_service()
        result = doc_service.index_directory(
            directory=request.path,
            recursive=request.recursive,
            file_extensions=request.file_extensions,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get indexing statistics"""
    try:
        # Get database stats
        db_stats = db.get_stats()

        # Get embedding stats
        embedding_service = get_embedding_service()
        embedding_stats = embedding_service.get_collection_stats()

        return StatsResponse(
            total_documents=db_stats["total_documents"],
            total_snippets=db_stats["total_snippets"],
            total_conversations=db_stats["total_conversations"],
            categories=db_stats["categories"],
            languages=db_stats["languages"],
            tags=db_stats["tags"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/embeddings/stats")
async def get_embedding_stats():
    """Get embedding service statistics"""
    try:
        embedding_service = get_embedding_service()
        return embedding_service.get_collection_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings/clear")
async def clear_embeddings():
    """Clear all embeddings (not documents)"""
    try:
        # This would clear the ChromaDB collection
        # But keep the documents in SQLite
        raise HTTPException(
            status_code=501,
            detail="Clearing embeddings is not yet implemented",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
