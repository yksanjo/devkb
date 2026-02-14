"""DevKB - FastAPI Application"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import db
from app.routers import admin, chat, documents, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    print("Starting DevKB...")
    print(f"Database: {settings.get_sqlite_path()}")
    print(f"ChromaDB: {settings.get_chroma_path()}")
    print(f"Embedding model: {settings.embedding_model}")

    # Initialize database
    _ = db

    yield

    # Shutdown
    print("Shutting down DevKB...")


app = FastAPI(
    title="DevKB - AI-Powered Developer Knowledge Base",
    description="Local-first knowledge management system for developers with semantic search and LLM-powered features",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "DevKB",
        "version": "0.1.0",
        "description": "AI-Powered Developer Knowledge Base",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "ok",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
