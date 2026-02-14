"""Embedding service using sentence-transformers"""
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    """Service for generating and storing embeddings"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        chroma_path: Optional[Path] = None,
    ):
        self.model_name = model_name or settings.embedding_model
        self.chroma_path = chroma_path or settings.get_chroma_path()
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        # Initialize sentence-transformers model
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("Embedding model loaded successfully")

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(str(self.chroma_path))

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="devkb_embeddings",
            metadata={"hnsw:space": "cosine"},
        )

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model"""
        return self.model.get_sentence_embedding_dimension()

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not text or not text.strip():
            return [0.0] * self.get_embedding_dimension()

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []

        # Filter empty texts
        valid_texts = [t if t and t.strip() else "" for t in texts]

        embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    def index_document(
        self,
        doc_id: int,
        chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Index document chunks with embeddings"""
        if not chunks:
            return []

        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)

        # Prepare metadata for each chunk
        base_metadata = metadata or {}
        base_metadata["doc_id"] = doc_id

        # Add to ChromaDB
        ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"doc_{doc_id}_chunk_{i}"
            chunk_metadata = {**base_metadata, "chunk_index": i}

            self.collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[chunk_metadata],
            )
            ids.append(chunk_id)

        return ids

    def search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not query or not query.strip():
            return []

        # Generate query embedding
        query_embedding = self.generate_embedding(query)

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            min_sim = min_similarity or settings.min_similarity_threshold

            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # Convert distance to similarity (cosine distance)
                similarity = 1 - distance

                if similarity >= min_sim:
                    search_results.append({
                        "chunk_id": chunk_id,
                        "chunk_text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity": similarity,
                    })

        return search_results

    def delete_document(self, doc_id: int) -> bool:
        """Delete all chunks for a document"""
        # Get all chunk IDs for this document
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=["ids"],
        )

        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            return True
        return False

    def get_document_chunks(self, doc_id: int) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas", "ids"],
        )

        if not results or not results["ids"]:
            return []

        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_text": results["documents"][i],
                "metadata": results["metadatas"][i],
            })

        return chunks

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "total_embeddings": self.collection.count(),
            "model_name": self.model_name,
            "dimension": self.get_embedding_dimension(),
        }


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content"""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# Global embedding service instance
embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton"""
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service
