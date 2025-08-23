"""Interfaces for SSE formatting service."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from app.services.pipeline.models import ProxyResponse, StreamChunk


class SseFormatter(ABC):
    """Interface for formatting responses as Server-Sent Events."""

    @abstractmethod
    async def format_response(self, response: ProxyResponse) -> AsyncIterator[StreamChunk]:
        """Convert a response to SSE format.

        Args:
            response: The response to format

        Yields:
            StreamChunk objects containing SSE-formatted data
        """
        pass

    @abstractmethod
    async def format_error(self, error_data: dict, correlation_id: str = None) -> StreamChunk:
        """Format error data as an SSE event.

        Args:
            error_data: Error information to format
            correlation_id: Optional correlation ID for request tracing

        Returns:
            StreamChunk containing SSE-formatted error
        """
        pass
