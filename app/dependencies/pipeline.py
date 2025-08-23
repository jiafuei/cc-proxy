"""Simplified message pipeline dependency with Anthropic as default."""

from typing import Optional

import httpx

from app.config import get_config
from app.services.pipeline.http_client import HttpClientService
from app.services.pipeline.messages_service import MessagesPipelineService
from app.services.pipeline.request_pipeline import RequestPipeline
from app.services.pipeline.response_pipeline import ResponsePipeline
from app.services.sse_formatter.anthropic_formatter import AnthropicSseFormatter
from app.services.transformers.anthropic.transformers import AnthropicRequestTransformer, AnthropicResponseTransformer, AnthropicStreamTransformer

# Global pipeline instance
_message_pipeline: Optional[MessagesPipelineService] = None
_http_client: Optional[httpx.AsyncClient] = None


def get_message_pipeline() -> MessagesPipelineService:
    """Get the default message pipeline with Anthropic transformers.

    Returns:
        Configured MessagesPipelineService with Anthropic defaults
    """
    global _message_pipeline

    if _message_pipeline is None:
        _message_pipeline = _create_default_pipeline()

    return _message_pipeline


def _create_default_pipeline() -> MessagesPipelineService:
    """Create default pipeline with Anthropic transformers."""
    config = get_config()

    # Create Anthropic transformers (the default pipeline)
    request_transformer = AnthropicRequestTransformer(config)
    response_transformer = AnthropicResponseTransformer()
    stream_transformer = AnthropicStreamTransformer()

    # Create pipelines
    request_pipeline = RequestPipeline([request_transformer])
    response_pipeline = ResponsePipeline([response_transformer], [stream_transformer])

    # Create HTTP client
    http_client = HttpClientService(_get_http_client())

    # Create SSE formatter
    sse_formatter = AnthropicSseFormatter()

    return MessagesPipelineService(request_pipeline=request_pipeline, response_pipeline=response_pipeline, http_client=http_client, sse_formatter=sse_formatter)


def _get_http_client() -> httpx.AsyncClient:
    """Get shared HTTP client instance."""
    global _http_client

    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=60 * 5, http2=True)

    return _http_client


def reset_pipeline() -> None:
    """Reset pipeline instance (primarily for testing)."""
    global _message_pipeline
    _message_pipeline = None
