"""Domain models for the pipeline service."""

import orjson
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from fastapi import Request
from pydantic import BaseModel, Field

from app.common.models import ClaudeRequest


@dataclass
class TransformationContext:
    """Context passed through transformers containing request metadata."""

    correlation_id: str
    original_request: Request


@dataclass
class ProxyRequest:
    """Internal pipeline request model."""

    claude_request: ClaudeRequest
    headers: Dict[str, str]
    context: TransformationContext

    # HTTP client details (set by request transformers)
    url: Optional[str] = None
    params: Optional[Dict[str, str]] = None

    @classmethod
    def from_claude_request(cls, claude_request: ClaudeRequest, original_request: Request, correlation_id: str) -> 'ProxyRequest':
        """Create ProxyRequest from ClaudeRequest and HTTP request."""

        # Extract headers from FastAPI request
        headers = dict(original_request.headers)

        context = TransformationContext(correlation_id=correlation_id, original_request=original_request)

        return cls(claude_request=claude_request, headers=headers, context=context, params=dict(original_request.query_params))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for HTTP client."""
        return {'url': self.url, 'headers': self.headers, 'json': self.claude_request.model_dump(), 'params': self.params}


@dataclass
class StreamChunk:
    """Individual chunk in a streaming response."""

    data: bytes
    chunk_type: str = 'data'  # data, error, done
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProxyResponse:
    """Response model with content, headers, and status code."""

    content: Union[bytes, Dict[str, Any], str]
    headers: Dict[str, str]
    status_code: int
    metadata: Optional[Dict[str, Any]] = None

    def to_bytes(self) -> bytes:
        """Convert response content to bytes."""
        if isinstance(self.content, bytes):
            return self.content
        elif isinstance(self.content, str):
            return self.content.encode('utf-8')
        else:
            # Assume it's a dict/JSON-serializable object
            return orjson.dumps(self.content)


class ClaudeErrorDetail(BaseModel):
    type: str
    message: str


class ClaudeError(BaseModel):
    type: str = Field(default='error')
    error: ClaudeErrorDetail
    request_id: Optional[str] = None
