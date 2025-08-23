"""Interfaces for error handling service."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import httpx

from .exceptions import PipelineException
from .models import ClaudeError


class ExceptionMapper(ABC):
    """Interface for mapping external exceptions to domain exceptions."""

    @abstractmethod
    def map_httpx_exception(self, exc: httpx.HTTPError, correlation_id: Optional[str] = None) -> PipelineException:
        """Map httpx exceptions to domain exceptions."""
        pass


class ErrorFormatter(ABC):
    """Interface for formatting exceptions into response formats."""

    @abstractmethod
    def format_for_sse(self, exc: PipelineException) -> Tuple[ClaudeError, str]:
        """Format exception for SSE response."""
        pass

    @abstractmethod
    def get_error_type(self, exc: PipelineException) -> str:
        """Get the error type string for an exception."""
        pass
