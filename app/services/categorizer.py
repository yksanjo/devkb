"""Categorization service using LLM"""
import json
import re
from typing import Any, Dict, List, Optional

import anthropic

from app.config import settings
from app.models import CategorizeResponse


class CategorizerService:
    """Service for auto-categorizing documents using LLM"""

    CATEGORIES = [
        "documentation",
        "configuration",
        "api",
        "database",
        "testing",
        "deployment",
        "authentication",
        "utilities",
        "models",
        "views",
        "controllers",
        "services",
        "middleware",
        "other",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.anthropic_api_key
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None

    def is_available(self) -> bool:
        """Check if the service is available (API key configured)"""
        return self.client is not None and self.api_key is not None

    def _build_categorize_prompt(self, content: str, max_length: int = 2000) -> str:
        """Build prompt for categorization"""
        # Truncate content if too long
        truncated = content[:max_length]

        return f"""Analyze the following code or documentation content and provide:
1. A category (one of: {', '.join(self.CATEGORIES)})
2. Relevant tags (array of strings)
3. A brief summary (1-2 sentences)

Content:
```
{truncated}
```

Provide your response in JSON format:
{{
    "category": "category_name",
    "tags": ["tag1", "tag2", "tag3"],
    "summary": "Brief summary of the content",
    "language": "programming_language_if_code"
}}"""

    def categorize(self, content: str) -> CategorizeResponse:
        """Categorize content using LLM"""
        if not self.is_available():
            # Return default categorization if no API key
            return self._categorize_fallback(content)

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                system="You are a code documentation analyzer. Return only valid JSON.",
                messages=[
                    {"role": "user", "content": self._build_categorize_prompt(content)}
                ],
            )

            # Parse response
            text = response.content[0].text.strip()
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return CategorizeResponse(
                    category=data.get("category", "other"),
                    tags=data.get("tags", []),
                    summary=data.get("summary", ""),
                    language=data.get("language"),
                )

        except Exception as e:
            print(f"Error in LLM categorization: {e}")

        # Fallback to rule-based categorization
        return self._categorize_fallback(content)

    def _categorize_fallback(self, content: str) -> CategorizeResponse:
        """Fallback rule-based categorization"""
        content_lower = content.lower()

        # Detect language
        language = self._detect_language(content)

        # Detect category based on keywords
        category = "other"
        if any(kw in content_lower for kw in ["test", "pytest", "unittest", "spec"]):
            category = "testing"
        elif any(kw in content_lower for kw in ["config", "settings", ".env", "yaml", "toml"]):
            category = "configuration"
        elif any(kw in content_lower for kw in ["def ", "class ", "function", "async"]):
            category = "utilities"
        elif any(kw in content_lower for kw in ["api", "endpoint", "route", "request", "response"]):
            category = "api"
        elif any(kw in content_lower for kw in ["database", "sql", "query", "model", "schema"]):
            category = "database"
        elif any(kw in content_lower for kw in ["deploy", "docker", "kubernetes", "ci/cd"]):
            category = "deployment"
        elif any(kw in content_lower for kw in ["auth", "login", "token", "jwt", "oauth"]):
            category = "authentication"
        elif any(kw in content_lower for kw in ["readme", "doc", "guide", "documentation"]):
            category = "documentation"

        # Extract tags
        tags = []
        if language:
            tags.append(language)
        if category != "other":
            tags.append(category)

        # Generate summary (first meaningful line)
        lines = content.split("\n")
        summary = ""
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("//") and not line.startswith("<!--"):
                summary = line[:100]
                break

        return CategorizeResponse(
            category=category,
            tags=tags,
            summary=summary,
            language=language,
        )

    def _detect_language(self, content: str) -> Optional[str]:
        """Detect programming language from content"""
        content_lower = content.lower()

        # Language patterns
        patterns = {
            "python": [r"def ", r"import ", r"from .* import", r"class .*:", r"if __name__"],
            "javascript": [r"const ", r"let ", r"function ", r"=>", r"require\("],
            "typescript": [r": string", r": number", r"interface ", r"type .*=", r"<T>"],
            "java": [r"public class", r"private ", r"void ", r"import java"],
            "go": [r"func ", r"package ", r"import ", r":="],
            "rust": [r"fn ", r"let mut", r"impl ", r"use "],
            "sql": [r"SELECT ", r"FROM ", r"WHERE ", r"INSERT INTO", r"CREATE TABLE"],
            "yaml": [r"^---", r"^\w+:\s*$", r"^\s+-\s+\w"],
            "json": [r'^\s*\{', r'^\s*\[', r'"[^"]+"\s*:'],
            "shell": [r"#!/bin/bash", r"#!/bin/sh", r"echo ", r"\$\("],
            "html": [r"<html", r"<div", r"<span", r"<!DOCTYPE"],
            "css": [r"\{[^}]*:[^}]*;", r"@media", r"\.[a-z-]+\s*\{"],
        }

        for lang, regexes in patterns.items():
            matches = sum(1 for r in regexes if re.search(r, content, re.IGNORECASE))
            if matches >= 2:
                return lang

        return None


# Global categorizer service instance
categorizer_service: Optional[CategorizerService] = None


def get_categorizer_service() -> CategorizerService:
    """Get or create categorizer service singleton"""
    global categorizer_service
    if categorizer_service is None:
        categorizer_service = CategorizerService()
    return categorizer_service
