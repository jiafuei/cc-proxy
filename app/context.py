"""Request context utilities for per-request state management."""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RequestContext:
    """Structured context data attached to each inbound request."""

    # Request identification
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:27] + 'fixed')
    request_id: Optional[str] = None

    # Model routing information (populated after routing decision)
    model_alias: Optional[str] = None
    resolved_model_id: Optional[str] = None
    original_model: Optional[str] = None
    provider_name: Optional[str] = None
    routing_key: Optional[str] = None

    # Routing metadata
    is_direct_routing: bool = False
    is_agent_routing: bool = False
    used_fallback: bool = False

    # Request metadata
    path: Optional[str] = None
    method: Optional[str] = None

    # Arbitrary extras for logging / dumper annotations
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_none: bool = False) -> Dict[str, Any]:
        """Serialize context for structured logging."""
        result: Dict[str, Any] = {}

        for key, value in {
            'correlation_id': self.correlation_id,
            'request_id': self.request_id,
            'model_alias': self.model_alias,
            'resolved_model_id': self.resolved_model_id,
            'original_model': self.original_model,
            'provider_name': self.provider_name,
            'routing_key': self.routing_key,
            'is_direct_routing': self.is_direct_routing if self.is_direct_routing else None,
            'is_agent_routing': self.is_agent_routing if self.is_agent_routing else None,
            'used_fallback': self.used_fallback if self.used_fallback else None,
            'path': self.path,
            'method': self.method,
        }.items():
            if include_none or value is not None:
                result[key] = value

        result.update(self.extra)
        return result

    def update_routing_info(
        self,
        *,
        model_alias: str,
        resolved_model_id: str,
        provider_name: str,
        routing_key: str,
        is_direct_routing: bool = False,
        is_agent_routing: bool = False,
        used_fallback: bool = False,
    ) -> None:
        """Capture routing decision details for downstream logging."""

        self.model_alias = model_alias
        self.resolved_model_id = resolved_model_id
        self.provider_name = provider_name
        self.routing_key = routing_key
        self.is_direct_routing = is_direct_routing
        self.is_agent_routing = is_agent_routing
        self.used_fallback = used_fallback


request_context_var: ContextVar[RequestContext] = ContextVar('request_context', default=RequestContext())


def get_request_context() -> RequestContext:
    """Return the active request context."""

    return request_context_var.get()


def set_request_context(context: RequestContext) -> None:
    """Replace the current request context."""

    request_context_var.set(context)


def get_correlation_id() -> str:
    """Expose the correlation ID for log formatting helpers."""

    return request_context_var.get().correlation_id


__all__ = [
    'RequestContext',
    'get_request_context',
    'set_request_context',
    'get_correlation_id',
    'request_context_var',
]
