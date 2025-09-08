import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RequestContext:
    """Unified request context for structured logging and request metadata."""

    # Request identification
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:27] + 'fixed')
    request_id: Optional[str] = None

    # Model routing information (populated after routing decision)
    model_alias: Optional[str] = None  # Logical model name from routing
    resolved_model_id: Optional[str] = None  # Actual model ID sent to provider
    original_model: Optional[str] = None  # Original model from request
    provider_name: Optional[str] = None  # Provider handling request
    routing_key: Optional[str] = None  # Routing decision made

    # Routing metadata
    is_direct_routing: bool = False  # Using ! suffix
    is_agent_routing: bool = False  # Using /model <model>
    used_fallback: bool = False  # Fallback provider used

    # Request metadata
    path: Optional[str] = None  # Request path
    method: Optional[str] = None  # HTTP method

    # Custom attributes
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_none: bool = False) -> dict:
        """Convert to dictionary for structured logging."""
        result = {}

        # Add all non-None fields
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

        # Add extra attributes
        result.update(self.extra)

        return result

    def update_routing_info(self, model_alias: str, resolved_model_id: str, provider_name: str, routing_key: str, **kwargs):
        """Update routing information after routing decision."""
        self.model_alias = model_alias
        self.resolved_model_id = resolved_model_id
        self.provider_name = provider_name
        self.routing_key = routing_key

        # Update optional flags
        self.is_direct_routing = kwargs.get('is_direct_routing', False)
        self.is_agent_routing = kwargs.get('is_agent_routing', False)
        self.used_fallback = kwargs.get('used_fallback', False)
