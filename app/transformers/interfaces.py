"""Unified transformer interfaces used by providers and routers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Tuple


class PreflightTransformer(ABC):
    """Transformers that adjust exchange requests before routing."""

    @abstractmethod
    async def transform(self, exchange_request: Any) -> Any:  # ExchangeRequest without circular import
        """Transform an exchange request before routing occurs."""


class ProviderRequestTransformer(ABC):
    """Interface for transformers that modify outgoing provider requests."""

    def __init__(self, logger):
        self.logger = logger

    def _is_builtin_tool(self, tool: dict) -> bool:
        return isinstance(tool, dict) and 'type' in tool and 'input_schema' not in tool

    def _has_builtin_tools(self, tools: list) -> bool:
        return any(self._is_builtin_tool(tool) for tool in tools if isinstance(tool, dict))

    @abstractmethod
    async def transform(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Transform the outgoing request payload and headers."""


class ProviderResponseTransformer(ABC):
    """Interface for transformers that modify completed provider responses."""

    def __init__(self, logger):
        self.logger = logger

    @abstractmethod
    async def transform_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a complete non-streaming provider response."""


class ProviderStreamTransformer(ABC):
    """Interface for transformers that modify streaming provider responses."""

    def __init__(self, logger):
        self.logger = logger

    @abstractmethod
    async def transform_chunk(self, params: Dict[str, Any]) -> AsyncIterator[bytes]:
        """Transform a streaming response chunk."""
