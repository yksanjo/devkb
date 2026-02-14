"""Search service with semantic and keyword search"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.database import db
from app.models import (
    ContentType,
    SearchRequest,
    SearchResult,
)
from app.services.embeddings import get_embedding_service


class SearchService:
    """Service for searching documents"""

    def __init__(self):
        self.embedding_service = get_embedding_service()

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Perform semantic search using embeddings"""
        # Search vector store
        search_results = self.embedding_service.search(
            query=query,
            limit=limit * 2,  # Get more to filter
            min_similarity=min_similarity,
            filters=filters,
        )

        results = []
        seen_doc_ids = set()

        for result in search_results:
            doc_id = result["metadata"].get("doc_id")
            if not doc_id or doc_id in seen_doc_ids:
                continue

            # Get document from database
            doc_data = db.get_document(doc_id)
            if not doc_data:
                continue

            # Parse tags from JSON
            tags = None
            if doc_data.get("tags"):
                try:
                    tags = json.loads(doc_data["tags"])
                except json.JSONDecodeError:
                    pass

            # Build document response
            doc_response = {
                "id": doc_data["id"],
                "file_path": doc_data["file_path"],
                "content_hash": doc_data["content_hash"],
                "title": doc_data.get("title"),
                "content_type": doc_data.get("content_type"),
                "language": doc_data.get("language"),
                "created_at": doc_data.get("created_at"),
                "updated_at": doc_data.get("updated_at"),
                "summary": doc_data.get("summary"),
                "tags": tags,
                "category": doc_data.get("category"),
                "snippets": [],
            }

            # Get snippets for this document
            snippets_data = db.get_snippets_by_document(doc_id)
            doc_response["snippets"] = [
                {
                    "id": s["id"],
                    "document_id": s["document_id"],
                    "chunk_text": s["chunk_text"],
                    "start_line": s.get("start_line"),
                    "end_line": s.get("end_line"),
                    "language": s.get("language"),
                    "intent": s.get("intent"),
                }
                for s in snippets_data
            ]

            # Create search result
            search_result = SearchResult(
                document=doc_response,
                snippet=result["chunk_text"],
                similarity=result["similarity"],
                highlights=self._extract_highlights(result["chunk_text"], query),
            )

            results.append(search_result)
            seen_doc_ids.add(doc_id)

            if len(results) >= limit:
                break

        return results

    def keyword_search(
        self,
        query: str,
        limit: int = 10,
        content_type: Optional[ContentType] = None,
        language: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform keyword-based search"""
        # Get all documents with filters
        items, _ = db.list_documents(
            page=1,
            page_size=100,  # Get more to filter
            content_type=content_type.value if content_type else None,
            language=language,
            category=category,
        )

        # Simple keyword matching
        query_lower = query.lower()
        results = []

        for doc_data in items:
            # Search in content (would need to load full content)
            # For now, search in title, summary, and path
            search_text = " ".join([
                doc_data.get("file_path", ""),
                doc_data.get("title", ""),
                doc_data.get("summary", ""),
            ]).lower()

            if query_lower in search_text:
                # Parse tags
                tags = None
                if doc_data.get("tags"):
                    try:
                        tags = json.loads(doc_data["tags"])
                    except json.JSONDecodeError:
                        pass

                # Get snippets
                snippets_data = db.get_snippets_by_document(doc_data["id"])

                doc_response = {
                    "id": doc_data["id"],
                    "file_path": doc_data["file_path"],
                    "content_hash": doc_data["content_hash"],
                    "title": doc_data.get("title"),
                    "content_type": doc_data.get("content_type"),
                    "language": doc_data.get("language"),
                    "created_at": doc_data.get("created_at"),
                    "updated_at": doc_data.get("updated_at"),
                    "summary": doc_data.get("summary"),
                    "tags": tags,
                    "category": doc_data.get("category"),
                    "snippets": [
                        {
                            "id": s["id"],
                            "document_id": s["document_id"],
                            "chunk_text": s["chunk_text"],
                            "start_line": s.get("start_line"),
                            "end_line": s.get("end_line"),
                            "language": s.get("language"),
                            "intent": s.get("intent"),
                        }
                        for s in snippets_data
                    ],
                }

                # Use 1.0 similarity for keyword matches (they're exact)
                search_result = SearchResult(
                    document=doc_response,
                    snippet=doc_data.get("summary", "")[:200],
                    similarity=1.0,
                    highlights=[query],
                )

                results.append(search_result)

                if len(results) >= limit:
                    break

        return results

    def search(
        self,
        request: SearchRequest,
    ) -> List[SearchResult]:
        """Perform search (semantic with keyword fallback)"""
        # Build filters
        filters = {}
        if request.language:
            filters["language"] = request.language
        if request.category:
            filters["category"] = request.category

        # Try semantic search first
        results = self.semantic_search(
            query=request.query,
            limit=request.limit,
            min_similarity=request.min_similarity,
            filters=filters if filters else None,
        )

        # If no semantic results, fall back to keyword search
        if not results:
            results = self.keyword_search(
                query=request.query,
                limit=request.limit,
                content_type=request.content_type,
                language=request.language,
                category=request.category,
            )

        # Filter by content_type and tags if specified
        if request.content_type or request.tags:
            filtered = []
            for r in results:
                if request.content_type and r.document.content_type != request.content_type.value:
                    continue
                if request.tags and r.document.tags:
                    if not any(tag in request.tags for tag in r.document.tags):
                        continue
                filtered.append(r)
            results = filtered

        return results

    def _extract_highlights(self, text: str, query: str) -> List[str]:
        """Extract highlight snippets from text"""
        if not text or not query:
            return []

        # Simple highlight extraction - find sentences containing query terms
        query_terms = query.lower().split()
        sentences = re.split(r'[.!?\n]', text)

        highlights = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in query_terms):
                # Truncate if too long
                if len(sentence) > 200:
                    sentence = sentence[:200] + "..."
                highlights.append(sentence)

        return highlights[:3]  # Return up to 3 highlights


# Global search service instance
search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """Get or create search service singleton"""
    global search_service
    if search_service is None:
        search_service = SearchService()
    return search_service
