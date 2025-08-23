import uuid
from pathlib import Path


def generate_correlation_id() -> str:
    """Generate a new correlation ID for request tracing."""
    return uuid.uuid4().hex


def get_correlation_id() -> str:
    from .vars import correlation_id

    return correlation_id.get()


def get_app_dir() -> Path:
    return Path.home() / '.cc-proxy'
