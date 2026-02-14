"""Claude Code integration service"""
import json
from typing import Any, Dict, List, Optional

import anthropic

from app.config import settings
from app.database import db
from app.models import ChatRequest, ChatResponse
from app.services.search import get_search_service


class ClaudeService:
    """Service for interacting with Claude Code"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.anthropic_api_key
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None

    def is_available(self) -> bool:
        """Check if the service is available (API key configured)"""
        return self.client is not None and self.api_key is not None

    def _build_context_from_sources(self, sources: List[Any], max_chars: int = 8000) -> str:
        """Build context string from search sources"""
        context_parts = []
        total_chars = 0

        for source in sources:
            # Get document info
            doc = source.document
            snippet = source.snippet

            context_part = f"""---
File: {doc.file_path}
Title: {doc.title or 'Untitled'}
Category: {doc.category or 'N/A'}
---

{snippet}
"""

            if total_chars + len(context_part) > max_chars:
                break

            context_parts.append(context_part)
            total_chars += len(context_part)

        return "\n\n".join(context_parts)

    def _build_system_prompt(self) -> str:
        """Build system prompt for Claude"""
        return """You are DevKB Assistant, an AI-powered developer knowledge base assistant. 
Your role is to help developers find and understand information from their codebase and documentation.

When answering questions:
1. Use the provided context from the knowledge base
2. Cite specific files and code snippets when possible
3. If you're unsure about something, say so
4. Keep your answers focused and practical

You have access to:
- Code snippets and documentation from the user's knowledge base
- Semantic search to find relevant information
- The ability to explain code and answer questions"""

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Chat with Claude using knowledge base context"""
        if not self.is_available():
            raise ValueError("Claude API key not configured")

        # Search for relevant context
        search_service = get_search_service()
        from app.services.search import SearchRequest as LocalSearchRequest
        
        search_results = search_service.semantic_search(
            query=request.message,
            limit=request.context_limit,
        )

        # Build context
        context = self._build_context_from_sources(search_results)

        # Build messages
        system_prompt = self._build_system_prompt()
        
        # Get conversation history if provided
        messages = []
        if request.conversation_id:
            # Could load previous messages from DB
            pass

        # Add context to user message
        user_message = f"""Context from knowledge base:
{context}

---

Question: {request.message}"""

        messages.append({"role": "user", "content": user_message})

        # Call Claude
        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                system=system_prompt,
                messages=messages,
            )

            response_text = response.content[0].text.strip()

        except Exception as e:
            raise RuntimeError(f"Error calling Claude API: {e}")

        # Store conversation
        conversation_id = db.insert_conversation(
            query=request.message,
            response=response_text,
            document_id=search_results[0].document.id if search_results else None,
        )

        return ChatResponse(
            message=response_text,
            conversation_id=conversation_id,
            sources=search_results,
        )

    def explain_code(self, code: str, language: Optional[str] = None) -> str:
        """Explain a piece of code"""
        if not self.is_available():
            raise ValueError("Claude API key not configured")

        language_hint = f" (in {language})" if language else ""

        system_prompt = """You are a code explanation assistant. Explain the given code clearly and concisely.
Focus on:
- What the code does
- Key components and their purpose
- Any important patterns or techniques used"""

        user_message = f"""Explain this code{language_hint}:

```{language or ''}
{code}
```"""

        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            return response.content[0].text.strip()

        except Exception as e:
            raise RuntimeError(f"Error calling Claude API: {e}")


# Global Claude service instance
claude_service: Optional[ClaudeService] = None


def get_claude_service() -> ClaudeService:
    """Get or create Claude service singleton"""
    global claude_service
    if claude_service is None:
        claude_service = ClaudeService()
    return claude_service
