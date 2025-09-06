import uuid
from pathlib import Path

from .vars import get_request_context


def generate_correlation_id() -> str:
    """Generate a new correlation ID for request tracing."""
    return uuid.uuid4().hex


def update_routing_context(model_alias: str, resolved_model_id: str, provider_name: str, routing_key: str, **kwargs) -> None:
    """Update routing information in current request context."""
    ctx = get_request_context()
    ctx.update_routing_info(model_alias=model_alias, resolved_model_id=resolved_model_id, provider_name=provider_name, routing_key=routing_key, **kwargs)


def get_app_dir() -> Path:
    return Path.home() / '.cc-proxy'
