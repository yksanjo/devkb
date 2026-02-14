"""DevKB Configuration"""
import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Keys
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")

    # Database
    sqlite_path: Path = Field(default=Path("./data/devkb.db"), alias="SQLITE_PATH")
    chroma_path: Path = Field(default=Path("./data/chroma"), alias="CHROMA_PATH")

    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")

    # Search
    default_search_limit: int = Field(default=10, alias="DEFAULT_SEARCH_LIMIT")
    min_similarity_threshold: float = Field(default=0.3, alias="MIN_SIMILARITY_THRESHOLD")

    # File watching
    watched_directories: List[str] = Field(default=["./knowledge"], alias="WATCHED_DIRECTORIES")

    # Chunking
    max_chunk_size: int = Field(default=1000, alias="MAX_CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def get_sqlite_path(self) -> Path:
        """Get absolute path for SQLite database"""
        base_dir = Path(__file__).parent.parent
        if self.sqlite_path.is_absolute():
            return self.sqlite_path
        return base_dir / self.sqlite_path

    def get_chroma_path(self) -> Path:
        """Get absolute path for ChromaDB"""
        base_dir = Path(__file__).parent.parent
        if self.chroma_path.is_absolute():
            return self.chroma_path
        return base_dir / self.chroma_path


# Global settings instance
settings = Settings()
