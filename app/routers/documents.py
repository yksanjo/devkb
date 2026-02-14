"""Document management API routes"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.database import db
from app.models import (
    DocumentCreate,
    DocumentListResponse,
    DocumentResponse,
    DocumentUpdate,
)
from app.services.documents import get_document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=201)
async def create_document(document: DocumentCreate):
    """Create a new document"""
    try:
        doc_service = get_document_service()
        return doc_service.create_document(document)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    content_type: Optional[str] = None,
    language: Optional[str] = None,
    category: Optional[str] = None,
):
    """List all documents with pagination"""
    items, total = db.list_documents(
        page=page,
        page_size=page_size,
        content_type=content_type,
        language=language,
        category=category,
    )

    # Convert to response format
    import json
    from datetime import datetime

    documents = []
    for item in items:
        tags = None
        if item.get("tags"):
            try:
                tags = json.loads(item["tags"])
            except json.JSONDecodeError:
                pass

        from app.models import CodeSnippet

        snippets_data = db.get_snippets_by_document(item["id"])
        snippets = [
            CodeSnippet(
                id=s["id"],
                document_id=s["document_id"],
                chunk_text=s["chunk_text"],
                start_line=s.get("start_line"),
                end_line=s.get("end_line"),
                language=s.get("language"),
                intent=s.get("intent"),
            )
            for s in snippets_data
        ]

        documents.append(
            DocumentResponse(
                id=item["id"],
                file_path=item["file_path"],
                content_hash=item["content_hash"],
                title=item.get("title"),
                content_type=item.get("content_type"),
                language=item.get("language"),
                created_at=item.get("created_at"),
                updated_at=item.get("updated_at"),
                summary=item.get("summary"),
                tags=tags,
                category=item.get("category"),
                snippets=snippets,
            )
        )

    pages = (total + page_size - 1) // page_size

    return DocumentListResponse(
        items=documents,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: int):
    """Get a document by ID"""
    try:
        doc_service = get_document_service()
        return doc_service.get_document(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{doc_id}", response_model=DocumentResponse)
async def update_document(doc_id: int, update: DocumentUpdate):
    """Update a document"""
    try:
        doc_service = get_document_service()
        return doc_service.update_document(
            doc_id=doc_id,
            title=update.title,
            summary=update.summary,
            tags=update.tags,
            category=update.category,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{doc_id}", status_code=204)
async def delete_document(doc_id: int):
    """Delete a document"""
    try:
        doc_service = get_document_service()
        success = doc_service.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{doc_id}/refresh", response_model=DocumentResponse)
async def refresh_document(doc_id: int):
    """Re-index a document"""
    try:
        # Get existing document
        doc_service = get_document_service()
        existing = doc_service.get_document(doc_id)

        # Delete and recreate
        doc_service.delete_document(doc_id)

        # Re-create with same content (would need to store content)
        # For now, just return error
        raise HTTPException(
            status_code=400,
            detail="Refresh requires re-uploading document content",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tags/list")
async def list_tags():
    """List all unique tags"""
    tags = db.get_all_tags()
    return {"tags": tags}


@router.get("/categories/list")
async def list_categories():
    """List all unique categories"""
    categories = db.get_all_categories()
    return {"categories": categories}
