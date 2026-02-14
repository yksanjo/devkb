"""Document service for managing documents"""
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.database import db
from app.models import ContentType, DocumentCreate, DocumentResponse
from app.services.categorizer import get_categorizer_service
from app.services.embeddings import (
    compute_content_hash,
    get_embedding_service,
)
from app.services.search import get_search_service


class DocumentService:
    """Service for managing documents"""

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        ".md", ".txt",
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".json", ".yaml", ".yml", ".toml",
        ".sql", ".sh", ".bash",
        ".go", ".rs", ".java", ".c", ".cpp", ".h",
        ".html", ".css", ".scss",
    }

    # Language mapping
    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".sql": "sql",
        ".sh": "shell",
        ".bash": "shell",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".md": "markdown",
        ".txt": "plain",
    }

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.search_service = get_search_service()
        self.categorizer = get_categorizer_service()

    def _detect_content_type(self, file_path: str, content: str) -> Tuple[ContentType, str]:
        """Detect content type and language"""
        path = Path(file_path)
        ext = path.suffix.lower()

        # Determine language
        language = self.LANGUAGE_MAP.get(ext, "plain")

        # Determine content type
        if ext == ".md":
            # Check if it's mostly code
            code_blocks = len(re.findall(r'```[\s\S]*?```', content))
            if code_blocks > 2:
                return ContentType.CODE, language
            return ContentType.MARKDOWN, language
        elif ext in {".py", ".js", ".ts", ".go", ".rs", ".java", ".sql", ".sh"}:
            return ContentType.CODE, language
        elif ext in {".json", ".yaml", ".toml"}:
            return ContentType.CODE, language
        else:
            return ContentType.PLAIN, language

    def _chunk_content(
        self,
        content: str,
        content_type: ContentType,
        language: str,
    ) -> List[Dict[str, Any]]:
        """Chunk content into smaller pieces for embedding"""
        chunks = []
        max_size = settings.max_chunk_size
        overlap = settings.chunk_overlap

        if content_type == ContentType.CODE:
            # For code, try to chunk by functions/classes
            chunks = self._chunk_code(content, language, max_size, overlap)
        elif content_type == ContentType.MARKDOWN:
            # For markdown, chunk by headings
            chunks = self._chunk_markdown(content, max_size, overlap)
        else:
            # For plain text, simple chunking
            chunks = self._chunk_text(content, max_size, overlap)

        return chunks

    def _chunk_code(
        self,
        content: str,
        language: str,
        max_size: int,
        overlap: int,
    ) -> List[Dict[str, Any]]:
        """Chunk code content"""
        chunks = []

        # Try to split by function/class definitions
        if language == "python":
            pattern = r'(^def |^class |^async def )'
        elif language in ("javascript", "typescript"):
            pattern = r'(^function |^const |^let |^class |^async )'
        elif language == "go":
            pattern = r'(^func |^type |^package )'
        elif language == "rust":
            pattern = r'(^fn |^struct |^enum |^impl )'
        else:
            pattern = None

        if pattern:
            # Split by function/class definitions
            lines = content.split("\n")
            current_chunk = []
            current_size = 0
            start_line = 1

            for i, line in enumerate(lines, 1):
                # Check if line starts a new definition
                if re.match(pattern, line.strip()) and current_chunk:
                    # Save current chunk
                    chunk_text = "\n".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "start_line": start_line,
                        "end_line": i - 1,
                        "language": language,
                        "intent": self._detect_code_intent(chunk_text, language),
                    })

                    # Start new chunk
                    current_chunk = [line]
                    current_size = len(line)
                    start_line = i
                else:
                    current_chunk.append(line)
                    current_size += len(line)

            # Add remaining chunk
            if current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start_line": start_line,
                    "end_line": len(lines),
                    "language": language,
                    "intent": self._detect_code_intent(chunk_text, language),
                })
        else:
            # Fallback to simple chunking
            chunks = self._chunk_text(content, max_size, overlap)

        # If chunks are too large, split them
        final_chunks = []
        for chunk in chunks:
            if len(chunk["text"]) > max_size:
                sub_chunks = self._chunk_text(chunk["text"], max_size, overlap)
                for sc in sub_chunks:
                    final_chunks.append({
                        "text": sc["text"],
                        "start_line": chunk.get("start_line", 1),
                        "end_line": chunk.get("end_line", 1),
                        "language": chunk.get("language", language),
                        "intent": chunk.get("intent"),
                    })
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _detect_code_intent(self, code: str, language: str) -> str:
        """Detect the intent/purpose of code"""
        code_lower = code.lower()

        if "test" in code_lower or "spec" in code_lower:
            return "test"
        elif "config" in code_lower or "settings" in code_lower:
            return "config"
        elif "class " in code_lower:
            return "class"
        elif "def " in code_lower or "function " in code_lower:
            return "function"
        elif "interface " in code_lower or "type " in code_lower:
            return "type"
        elif "route" in code_lower or "endpoint" in code_lower:
            return "endpoint"
        elif "query" in code_lower or "select" in code_lower:
            return "query"
        else:
            return "utility"

    def _chunk_markdown(
        self,
        content: str,
        max_size: int,
        overlap: int,
    ) -> List[Dict[str, Any]]:
        """Chunk markdown content"""
        chunks = []

        # Split by headings
        sections = re.split(r'(^#+ .+$\n)', content, flags=re.MULTILINE)

        current_section = []
        current_size = 0
        current_heading = ""

        for section in sections:
            if section.startswith("#"):
                # Save previous section
                if current_section:
                    chunk_text = "".join(current_section)
                    if chunk_text.strip():
                        chunks.append({
                            "text": chunk_text,
                            "start_line": 1,
                            "end_line": 1,
                            "language": "markdown",
                            "intent": "section",
                        })

                current_heading = section.strip()
                current_section = [section]
                current_size = len(section)
            else:
                if current_size + len(section) > max_size and current_section:
                    # Save current chunk
                    chunk_text = "".join(current_section)
                    if chunk_text.strip():
                        chunks.append({
                            "text": chunk_text,
                            "start_line": 1,
                            "end_line": 1,
                            "language": "markdown",
                            "intent": "section",
                        })

                    # Keep overlap
                    overlap_text = "".join(current_section[-overlap:])
                    current_section = [overlap_text]
                    current_size = len(overlap_text)

                current_section.append(section)
                current_size += len(section)

        # Add final chunk
        if current_section:
            chunk_text = "".join(current_section)
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "start_line": 1,
                    "end_line": 1,
                    "language": "markdown",
                    "intent": "section",
                })

        return chunks

    def _chunk_text(
        self,
        content: str,
        max_size: int,
        overlap: int,
    ) -> List[Dict[str, Any]]:
        """Simple text chunking"""
        chunks = []
        lines = content.split("\n")
        current_chunk = []
        current_size = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            if current_size + len(line) > max_size and current_chunk:
                chunks.append({
                    "text": "\n".join(current_chunk),
                    "start_line": start_line,
                    "end_line": i - 1,
                    "language": "plain",
                    "intent": "text",
                })

                # Keep overlap
                overlap_lines = current_chunk[-overlap:]
                current_chunk = overlap_lines
                current_size = sum(len(l) for l in overlap_lines)
                start_line = i - len(overlap_lines)

            current_chunk.append(line)
            current_size += len(line)

        if current_chunk:
            chunks.append({
                "text": "\n".join(current_chunk),
                "start_line": start_line,
                "end_line": len(lines),
                "language": "plain",
                "intent": "text",
            })

        return chunks

    def _extract_title(self, file_path: str, content: str) -> str:
        """Extract title from content"""
        path = Path(file_path)

        # Try to get title from markdown heading
        if path.suffix == ".md":
            match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if match:
                return match.group(1).strip()

        # Use filename
        return path.stem.replace("_", " ").replace("-", " ").title()

    def create_document(self, document: DocumentCreate) -> DocumentResponse:
        """Create a new document"""
        # Compute content hash
        content_hash = compute_content_hash(document.content)

        # Check for duplicates
        existing = db.get_document_by_path(document.file_path)
        if existing:
            if existing["content_hash"] == content_hash:
                # Same content, return existing
                return self.get_document(existing["id"])
            else:
                # Content changed, delete old and reindex
                self.delete_document(existing["id"])

        # Detect content type and language
        content_type, language = self._detect_content_type(
            document.file_path,
            document.content
        )

        # Use provided values if available
        if document.content_type:
            content_type = document.content_type
        if document.language:
            language = document.language

        # Extract title
        title = document.title or self._extract_title(
            document.file_path,
            document.content
        )

        # Categorize content
        categorization = self.categorizer.categorize(document.content)

        # Insert document
        doc_id = db.insert_document(
            file_path=document.file_path,
            content_hash=content_hash,
            title=title,
            content_type=content_type.value,
            language=language or categorization.language,
            summary=categorization.summary,
            tags=categorization.tags,
            category=categorization.category,
        )

        # Chunk and index content
        chunks = self._chunk_content(document.content, content_type, language or "plain")

        for chunk in chunks:
            # Insert snippet
            snippet_id = db.insert_snippet(
                document_id=doc_id,
                chunk_text=chunk["text"],
                start_line=chunk.get("start_line"),
                end_line=chunk.get("end_line"),
                language=chunk.get("language"),
                intent=chunk.get("intent"),
            )

            # Add to vector store
            self.embedding_service.index_document(
                doc_id=doc_id,
                chunks=[chunk["text"]],
                metadata={
                    "file_path": document.file_path,
                    "language": chunk.get("language"),
                    "intent": chunk.get("intent"),
                },
            )

        return self.get_document(doc_id)

    def get_document(self, doc_id: int) -> DocumentResponse:
        """Get document by ID"""
        doc_data = db.get_document(doc_id)
        if not doc_data:
            raise ValueError(f"Document {doc_id} not found")

        # Get snippets
        snippets_data = db.get_snippets_by_document(doc_id)

        # Parse tags
        tags = None
        if doc_data.get("tags"):
            import json
            try:
                tags = json.loads(doc_data["tags"])
            except json.JSONDecodeError:
                pass

        return DocumentResponse(
            id=doc_data["id"],
            file_path=doc_data["file_path"],
            content_hash=doc_data["content_hash"],
            title=doc_data.get("title"),
            content_type=doc_data.get("content_type"),
            language=doc_data.get("language"),
            created_at=doc_data.get("created_at"),
            updated_at=doc_data.get("updated_at"),
            summary=doc_data.get("summary"),
            tags=tags,
            category=doc_data.get("category"),
            snippets=[
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
        )

    def update_document(
        self,
        doc_id: int,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
    ) -> DocumentResponse:
        """Update document metadata"""
        db.update_document(
            doc_id=doc_id,
            title=title,
            summary=summary,
            tags=tags,
            category=category,
        )
        return self.get_document(doc_id)

    def delete_document(self, doc_id: int) -> bool:
        """Delete document and its embeddings"""
        # Delete from vector store
        self.embedding_service.delete_document(doc_id)

        # Delete snippets
        db.delete_snippets_by_document(doc_id)

        # Delete document
        return db.delete_document(doc_id)

    def index_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Index all files in a directory"""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")

        # Default extensions
        if file_extensions is None:
            file_extensions = list(self.SUPPORTED_EXTENSIONS)

        # Find files
        files = []
        if recursive:
            for ext in file_extensions:
                files.extend(dir_path.rglob(f"*{ext}"))
        else:
            for ext in file_extensions:
                files.extend(dir_path.glob(f"*{ext}"))

        # Index files
        indexed = 0
        errors = []

        for file_path in files:
            try:
                # Read content
                content = file_path.read_text(encoding="utf-8")

                # Skip empty files
                if not content.strip():
                    continue

                # Create document
                doc = DocumentCreate(
                    file_path=str(file_path),
                    content=content,
                )
                self.create_document(doc)
                indexed += 1

            except Exception as e:
                errors.append({
                    "file": str(file_path),
                    "error": str(e),
                })

        return {
            "indexed": indexed,
            "total": len(files),
            "errors": errors,
        }


# Global document service instance
document_service: Optional[DocumentService] = None


def get_document_service() -> DocumentService:
    """Get or create document service singleton"""
    global document_service
    if document_service is None:
        document_service = DocumentService()
    return document_service
