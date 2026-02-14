# DevKB - AI-Powered Developer Knowledge Base

A local-first knowledge management system specifically designed for developers. DevKB provides semantic search across code snippets, documentation, and notes with offline vector embeddings and LLM-powered auto-categorization.

## Features

- **Semantic Search**: Find relevant code and documentation using natural language queries
- **Vector Embeddings**: Offline-capable with sentence-transformers (all-MiniLM-L6-v2)
- **Auto-Categorization**: Automatic tagging and categorization using LLMs
- **Claude Code Integration**: Context-aware assistance with Claude
- **Multiple File Types**: Support for Python, JavaScript, TypeScript, Markdown, and more
- **API-First**: FastAPI-based REST API

## Tech Stack

- **Backend**: FastAPI (Python)
- **Database**: SQLite + ChromaDB
- **Embeddings**: sentence-transformers
- **LLM**: Anthropic Claude API

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and configure:
   ```bash
   cp .env.example .env
   ```
4. Set your Anthropic API key in `.env` (optional, for LLM features):
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

## Usage

### Start the server

```bash
cd devkb
python -m app.main
```

Or with uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Documents
- `POST /documents` - Add a document
- `GET /documents` - List documents (paginated)
- `GET /documents/{id}` - Get document by ID
- `PATCH /documents/{id}` - Update document
- `DELETE /documents/{id}` - Delete document

### Search
- `POST /search` - Semantic search
- `GET /search/keyword` - Keyword search

### Chat
- `POST /chat` - Chat with Claude using knowledge base context
- `POST /chat/explain` - Explain code

### Admin
- `POST /admin/index/directory` - Index a directory
- `GET /admin/stats` - Get statistics

## Example Usage

### Add a document

```bash
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "example.py",
    "content": "def hello_world():\n    print(\"Hello, World!\")",
    "title": "Hello World Example"
  }'
```

### Search

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how to print hello world",
    "limit": 5
  }'
```

### Index a directory

```bash
curl -X POST "http://localhost:8000/admin/index/directory" \
  -H "Content-Type: application/json" \
  -d '{
    "path": "./my-project",
    "recursive": true
  }'
```

## Development

### Run tests

```bash
pytest
```

### Project Structure

```
devkb/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app
│   ├── config.py         # Configuration
│   ├── models.py         # Pydantic models
│   ├── database.py       # SQLite setup
│   ├── services/
│   │   ├── embeddings.py   # Sentence-transformers
│   │   ├── search.py       # Search logic
│   │   ├── categorizer.py  # LLM categorization
│   │   ├── claude.py       # Claude integration
│   │   └── documents.py    # Document management
│   └── routers/
│       ├── documents.py
│       ├── search.py
│       ├── chat.py
│       └── admin.py
├── data/                  # Database storage
├── requirements.txt
└── .env.example
```

## License

MIT
