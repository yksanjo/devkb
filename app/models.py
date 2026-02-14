"""Pydantic models for request/response handling"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Document content types"""
    CODE = "code"
    MARKDOWN = "markdown"
    PLAIN = "plain"


class DocumentCreate(BaseModel):
    """Model for creating a new document"""
    file_path: str
    content: str
    title: Optional[str] = None
    content_type: Optional[ContentType] = None
    language: Optional[str] = None


class DocumentUpdate(BaseModel):
    """Model for updating a document"""
    title: Optional[str] = None
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None


class CodeSnippet(BaseModel):
    """Code snippet model"""
    id: int
    document_id: int
    chunk_text: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    language: Optional[str] = None
    intent: Optional[str] = None

    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    """Document response model"""
    id: int
    file_path: str
    content_hash: str
    title: Optional[str]
    content_type: Optional[str]
    language: Optional[str]
    created_at: datetime
    updated_at: datetime
    summary: Optional[str]
    tags: Optional[List[str]]
    category: Optional[str]
    snippets: List[CodeSnippet] = []

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Paginated document list"""
    items: List[DocumentResponse]
    total: int
    page: int
    page_size: int
    pages: int


class SearchRequest(BaseModel):
    """Semantic search request"""
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    min_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    content_type: Optional[ContentType] = None
    language: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None


class SearchResult(BaseModel):
    """Search result with similarity score"""
    document: DocumentResponse
    snippet: str
    similarity: float
    highlights: List[str] = []


class SearchResponse(BaseModel):
    """Search response"""
    results: List[SearchResult]
    query: str
    total: int


class ChatMessage(BaseModel):
    """Chat message"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat request with Claude"""
    message: str
    conversation_id: Optional[int] = None
    context_limit: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    """Chat response"""
    message: str
    conversation_id: int
    sources: List[SearchResult] = []


class CategorizeRequest(BaseModel):
    """Request to categorize a document"""
    content: str


class CategorizeResponse(BaseModel):
    """Categorization response"""
    category: str
    tags: List[str]
    summary: str
    language: Optional[str] = None


class IndexDirectoryRequest(BaseModel):
    """Request to index a directory"""
    path: str
    recursive: bool = True
    file_extensions: Optional[List[str]] = None


class StatsResponse(BaseModel):
    """Statistics response"""
    total_documents: int
    total_snippets: int
    total_conversations: int
    categories: Dict[str, int]
    languages: Dict[str, int]
    tags: Dict[str, int]
