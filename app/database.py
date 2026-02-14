"""Database setup and management"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from app.config import settings


class Database:
    """SQLite database manager for DevKB"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.get_sqlite_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get SQLite connection"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def get_conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connection"""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema"""
        with self.get_conn() as conn:
            cursor = conn.cursor()

            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    content_hash TEXT NOT NULL,
                    title TEXT,
                    content_type TEXT,
                    language TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    summary TEXT,
                    tags TEXT,
                    category TEXT
                )
            """)

            # Code snippets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS code_snippets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_text TEXT NOT NULL,
                    embedding_id TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    language TEXT,
                    intent TEXT
                )
            """)

            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Tags table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    color TEXT
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_content_type 
                ON documents(content_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_language 
                ON documents(language)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_category 
                ON documents(category)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_snippets_document_id 
                ON code_snippets(document_id)
            """)

    def insert_document(
        self,
        file_path: str,
        content_hash: str,
        title: Optional[str] = None,
        content_type: Optional[str] = None,
        language: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
    ) -> int:
        """Insert a new document"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO documents 
                (file_path, content_hash, title, content_type, language, summary, tags, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_path,
                    content_hash,
                    title,
                    content_type,
                    language,
                    summary,
                    json.dumps(tags) if tags else None,
                    category,
                ),
            )
            return cursor.lastrowid

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT d.*, 
                       GROUP_CONCAT(s.id || ':' || s.chunk_text, '|||') as snippets
                FROM documents d
                LEFT JOIN code_snippets s ON d.id = s.document_id
                WHERE d.id = ?
                GROUP BY d.id
                """,
                (doc_id,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_document_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get document by file path"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE file_path = ?", (file_path,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def update_document(
        self,
        doc_id: int,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
    ) -> bool:
        """Update document"""
        updates = []
        params = []

        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if summary is not None:
            updates.append("summary = ?")
            params.append(summary)
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        if category is not None:
            updates.append("category = ?")
            params.append(category)

        if not updates:
            return False

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(doc_id)

        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE documents SET {', '.join(updates)} WHERE id = ?", params
            )
            return cursor.rowcount > 0

    def delete_document(self, doc_id: int) -> bool:
        """Delete document"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            return cursor.rowcount > 0

    def list_documents(
        self,
        page: int = 1,
        page_size: int = 20,
        content_type: Optional[str] = None,
        language: Optional[str] = None,
        category: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """List documents with pagination"""
        offset = (page - 1) * page_size

        where_clauses = []
        params = []

        if content_type:
            where_clauses.append("content_type = ?")
            params.append(content_type)
        if language:
            where_clauses.append("language = ?")
            params.append(language)
        if category:
            where_clauses.append("category = ?")
            params.append(category)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        with self.get_conn() as conn:
            cursor = conn.cursor()

            # Get total count
            cursor.execute(
                f"SELECT COUNT(*) as total FROM documents WHERE {where_sql}", params
            )
            total = cursor.fetchone()["total"]

            # Get items
            cursor.execute(
                f"""
                SELECT * FROM documents 
                WHERE {where_sql}
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                params + [page_size, offset],
            )
            items = [dict(row) for row in cursor.fetchall()]

        return items, total

    def insert_snippet(
        self,
        document_id: int,
        chunk_text: str,
        embedding_id: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        language: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> int:
        """Insert code snippet"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO code_snippets 
                (document_id, chunk_text, embedding_id, start_line, end_line, language, intent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    chunk_text,
                    embedding_id,
                    start_line,
                    end_line,
                    language,
                    intent,
                ),
            )
            return cursor.lastrowid

    def get_snippets_by_document(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all snippets for a document"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM code_snippets WHERE document_id = ?", (document_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_snippets_by_document(self, document_id: int) -> bool:
        """Delete all snippets for a document"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM code_snippets WHERE document_id = ?", (document_id,)
            )
            return cursor.rowcount > 0

    def insert_conversation(
        self,
        query: str,
        response: str,
        document_id: Optional[int] = None,
    ) -> int:
        """Insert conversation"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversations (query, response, document_id)
                VALUES (?, ?, ?)
                """,
                (query, response, document_id),
            )
            return cursor.lastrowid

    def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Get conversation by ID"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_conn() as conn:
            cursor = conn.cursor()

            # Total documents
            cursor.execute("SELECT COUNT(*) as total FROM documents")
            total_documents = cursor.fetchone()["total"]

            # Total snippets
            cursor.execute("SELECT COUNT(*) as total FROM code_snippets")
            total_snippets = cursor.fetchone()["total"]

            # Total conversations
            cursor.execute("SELECT COUNT(*) as total FROM conversations")
            total_conversations = cursor.fetchone()["total"]

            # Categories
            cursor.execute(
                "SELECT category, COUNT(*) as count FROM documents WHERE category IS NOT NULL GROUP BY category"
            )
            categories = {row["category"]: row["count"] for row in cursor.fetchall()}

            # Languages
            cursor.execute(
                "SELECT language, COUNT(*) as count FROM documents WHERE language IS NOT NULL GROUP BY language"
            )
            languages = {row["language"]: row["count"] for row in cursor.fetchall()}

            # Tags (from JSON)
            cursor.execute("SELECT tags FROM documents WHERE tags IS NOT NULL")
            tags: Dict[str, int] = {}
            for row in cursor.fetchall():
                if row["tags"]:
                    tag_list = json.loads(row["tags"])
                    for tag in tag_list:
                        tags[tag] = tags.get(tag, 0) + 1

            return {
                "total_documents": total_documents,
                "total_snippets": total_snippets,
                "total_conversations": total_conversations,
                "categories": categories,
                "languages": languages,
                "tags": tags,
            }

    def get_all_tags(self) -> List[str]:
        """Get all unique tags"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT tags FROM documents WHERE tags IS NOT NULL")
            tags = set()
            for row in cursor.fetchall():
                if row["tags"]:
                    tag_list = json.loads(row["tags"])
                    tags.update(tag_list)
            return sorted(list(tags))

    def get_all_categories(self) -> List[str]:
        """Get all unique categories"""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT category FROM documents WHERE category IS NOT NULL"
            )
            return [row["category"] for row in cursor.fetchall()]


# Global database instance
db = Database()
